// Package llmtools implements llm_tools — the keystone for
// ReAct / function-calling agents. It exposes Anthropic's tool_use
// content blocks as N dynamic output ports, one per declared tool.
// Per call, the model either picks a tool (emitting structured args
// on out_<toolname>) or produces a final text response (emitting on
// the `response` port).
//
// The component itself is stateless: each invocation is one
// Anthropic call. The caller supplies the full `messages` history,
// and when a tool is picked the component emits the UPDATED messages
// (including the assistant's tool_use block) so the next invocation
// can append a tool_result and continue. This is how ReAct loops are
// built: tool → handler → next llm_tools call with appended result.
package llmtools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/tiny-systems/module/api/v1alpha1"
	"github.com/tiny-systems/module/module"
	perrors "github.com/tiny-systems/module/pkg/errors"
	"github.com/tiny-systems/module/registry"
)

const (
	ComponentName    = "llm_tools"
	RequestPort      = "request"
	ResponsePort     = "response"
	ErrorPort        = "error"
	anthropicURL     = "https://api.anthropic.com/v1/messages"
	anthropicVersion = "2023-06-01"
	defaultModel     = "claude-haiku-4-5"
	defaultMaxTokens = 1024
	defaultTimeout   = 60 * time.Second

	outPortPrefix = "out_"
)

type Context any

// Tool declares one function the model can call. Mirrors Anthropic's
// tool spec: name + description (the model reads both to decide when
// to invoke), plus an input schema constraining the structured args
// the model emits. inputSchema is a JSON Schema fragment — pass
// `{type: object, properties: {...}, required: [...]}` shape.
type Tool struct {
	Name        string         `json:"name" required:"true" minLength:"1" title:"Name" description:"Becomes output port out_<lowercase(name)>. Used by the model to identify the tool."`
	Description string         `json:"description" required:"true" minLength:"1" title:"Description" description:"Tells the model what the tool does and when to use it. Be specific."`
	InputSchema map[string]any `json:"inputSchema" required:"true" title:"Input schema" description:"JSON Schema for the tool's arguments. Example: {type: object, properties: {query: {type: string}}, required: [query]}."`
}

type Settings struct {
	EnableErrorPort bool    `json:"enableErrorPort" required:"true" title:"Enable Error Port"`
	Tools           []Tool  `json:"tools" required:"true" minItems:"1" uniqueItems:"true" title:"Tools" description:"Tools the model may invoke. At least one."`
	Model           string  `json:"model" required:"true" minLength:"1" default:"claude-haiku-4-5" title:"Model"`
	SystemPrompt    string  `json:"systemPrompt" title:"System Prompt" format:"textarea" description:"Frames the model's behavior across all turns."`
	CacheSystem     bool    `json:"cacheSystem" title:"Cache System Prompt" description:"Mark the system prompt as ephemeral so subsequent identical calls hit the prompt cache."`
	MaxTokens       int     `json:"maxTokens" required:"true" minimum:"1" default:"1024" title:"Max Tokens"`
	Temperature     float64 `json:"temperature" minimum:"0" maximum:"1" title:"Temperature"`
	TimeoutSeconds  int     `json:"timeoutSeconds" minimum:"1" default:"60" title:"Timeout Seconds"`
}

// Message is one turn in the conversation. Content can be a plain
// string (user/assistant text) OR a list of content blocks for
// tool_use and tool_result interactions. The caller is responsible
// for building messages — typically:
//   - first call: [{role: user, content: "<question>"}]
//   - after a tool_use: previous messages + [{role: user, content: [{type: tool_result, tool_use_id, content}]}]
type Message struct {
	Role    string `json:"role" title:"Role" description:"'user' or 'assistant'"`
	Content any    `json:"content" title:"Content" description:"String for plain text, or list of content blocks for tool_use/tool_result."`
}

type Request struct {
	Context  Context   `json:"context,omitempty" configurable:"true" title:"Context" description:"Passthrough emitted on whichever output port fires."`
	APIKey   string    `json:"apiKey" required:"true" minLength:"1" title:"Anthropic API Key" format:"password"`
	Messages []Message `json:"messages" required:"true" minItems:"1" title:"Messages" description:"Full conversation history sent to Claude."`
}

// ToolCall is the payload emitted on a tool's out_<name> port. The
// caller routes this to the matching handler, then feeds the
// handler's output back via a new llm_tools.request call with
// messages updated to include the tool_result.
type ToolCall struct {
	Context   Context   `json:"context"`
	Messages  []Message `json:"messages" description:"Updated history including the model's tool_use response. Append a tool_result and re-call to continue."`
	ToolUseID string    `json:"toolUseId" description:"Anthropic's id for this tool call. Required when building the tool_result for the next turn."`
	Input     any       `json:"input" description:"Structured arguments per the tool's inputSchema."`
}

type Usage struct {
	Input         int `json:"input"`
	Output        int `json:"output"`
	CacheRead     int `json:"cacheRead"`
	CacheCreation int `json:"cacheCreation"`
}

type Response struct {
	Context  Context   `json:"context"`
	Text     string    `json:"text"`
	Messages []Message `json:"messages" description:"Final history including the assistant's text reply. Use as a starting point for future conversation turns."`
	Model    string    `json:"model"`
	Usage    Usage     `json:"usage"`
}

type Error struct {
	Context   Context `json:"context"`
	Error     string  `json:"error"`
	Retryable bool    `json:"retryable"`
}

type Component struct {
	settings Settings
}

func (c *Component) Instance() module.Component {
	return &Component{settings: Settings{
		Model:          defaultModel,
		MaxTokens:      defaultMaxTokens,
		TimeoutSeconds: int(defaultTimeout / time.Second),
	}}
}

func (c *Component) GetInfo() module.ComponentInfo {
	return module.ComponentInfo{
		Name:        ComponentName,
		Description: "LLM Tools",
		Info: "ReAct / function-calling primitive. Declare tools in Settings; each becomes an out_<toolname> source port emitting " +
			"{toolUseId, input, messages} when the model picks it. If the model produces a final text response instead, that fires " +
			"on the response port. Component is stateless — caller supplies full message history. To build a ReAct loop: wire " +
			"out_<tool> → handler → another llm_tools.request with messages updated to append a tool_result for that toolUseId. " +
			"Loop until response port fires. API key per-request like llm_complete.",
		Tags: []string{"LLM", "Anthropic", "Tools", "ReAct", "Agent"},
	}
}

func (c *Component) OnSettings(_ context.Context, msg any) error {
	in, ok := msg.(Settings)
	if !ok {
		return fmt.Errorf("invalid settings")
	}
	if len(in.Tools) == 0 {
		return fmt.Errorf("at least one tool required")
	}
	seen := map[string]bool{}
	for i, t := range in.Tools {
		if strings.TrimSpace(t.Name) == "" {
			return fmt.Errorf("tools[%d]: empty name", i)
		}
		key := strings.ToLower(t.Name)
		if seen[key] {
			return fmt.Errorf("tools[%d]: duplicate name %q (case-insensitive)", i, t.Name)
		}
		seen[key] = true
		if t.InputSchema == nil {
			return fmt.Errorf("tools[%d] (%s): inputSchema required", i, t.Name)
		}
	}
	c.settings = in
	return nil
}

func (c *Component) Handle(ctx context.Context, handler module.Handler, port string, msg any) module.Result {
	if port != RequestPort {
		return module.Fail(fmt.Errorf("unknown port: %s", port))
	}
	in, ok := msg.(Request)
	if !ok {
		return module.Fail(fmt.Errorf("invalid request"))
	}
	return c.invoke(ctx, handler, in)
}

// --- Anthropic API plumbing --------------------------------------

type apiContentBlock struct {
	Type         string         `json:"type"`
	Text         string         `json:"text,omitempty"`
	ID           string         `json:"id,omitempty"`
	Name         string         `json:"name,omitempty"`
	Input        any            `json:"input,omitempty"`
	ToolUseID    string         `json:"tool_use_id,omitempty"`
	Content      any            `json:"content,omitempty"`
	CacheControl *cacheControl  `json:"cache_control,omitempty"`
}

type cacheControl struct {
	Type string `json:"type"`
}

type apiMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // string or []apiContentBlock
}

type apiToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

type apiSystemBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}

type apiRequest struct {
	Model       string           `json:"model"`
	MaxTokens   int              `json:"max_tokens"`
	Temperature float64          `json:"temperature,omitempty"`
	System      []apiSystemBlock `json:"system,omitempty"`
	Messages    []apiMessage     `json:"messages"`
	Tools       []apiToolDef     `json:"tools,omitempty"`
}

type apiResponseUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

type apiResponse struct {
	Model      string            `json:"model"`
	Content    []apiContentBlock `json:"content"`
	StopReason string            `json:"stop_reason"`
	Usage      apiResponseUsage  `json:"usage"`
}

type apiError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type apiErrorResponse struct {
	Error apiError `json:"error"`
}

func (c *Component) invoke(ctx context.Context, handler module.Handler, in Request) module.Result {
	timeout := time.Duration(c.settings.TimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = defaultTimeout
	}
	model := c.settings.Model
	if model == "" {
		model = defaultModel
	}

	body := apiRequest{
		Model:       model,
		MaxTokens:   c.settings.MaxTokens,
		Temperature: c.settings.Temperature,
		Messages:    messagesToAPI(in.Messages),
		Tools:       toolsToAPI(c.settings.Tools),
	}
	if c.settings.SystemPrompt != "" {
		block := apiSystemBlock{Type: "text", Text: c.settings.SystemPrompt}
		if c.settings.CacheSystem {
			block.CacheControl = &cacheControl{Type: "ephemeral"}
		}
		body.System = []apiSystemBlock{block}
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return c.fail(ctx, handler, in.Context, err, false)
	}

	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, anthropicURL, bytes.NewReader(payload))
	if err != nil {
		return c.fail(ctx, handler, in.Context, err, false)
	}
	httpReq.Header.Set("content-type", "application/json")
	httpReq.Header.Set("x-api-key", in.APIKey)
	httpReq.Header.Set("anthropic-version", anthropicVersion)

	resp, err := (&http.Client{}).Do(httpReq)
	if err != nil {
		return c.fail(ctx, handler, in.Context, err, true)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return c.fail(ctx, handler, in.Context, err, false)
	}

	if resp.StatusCode >= 400 {
		retryable := resp.StatusCode == http.StatusTooManyRequests ||
			resp.StatusCode == 529 || resp.StatusCode >= 500
		var apiErr apiErrorResponse
		if jsonErr := json.Unmarshal(respBody, &apiErr); jsonErr == nil && apiErr.Error.Message != "" {
			return c.fail(ctx, handler, in.Context,
				fmt.Errorf("%s: %s", apiErr.Error.Type, apiErr.Error.Message), retryable)
		}
		return c.fail(ctx, handler, in.Context,
			fmt.Errorf("anthropic api: status %d: %s", resp.StatusCode, string(respBody)), retryable)
	}

	var parsed apiResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return c.fail(ctx, handler, in.Context, fmt.Errorf("decode response: %w", err), false)
	}

	// Walk content blocks. First tool_use wins (Anthropic can return
	// multiple in parallel; the common case is one, and we keep the
	// surface simple by routing the first). Text-only responses go on
	// the response port.
	var textBuf strings.Builder
	var toolBlock *apiContentBlock
	for i := range parsed.Content {
		block := &parsed.Content[i]
		switch block.Type {
		case "text":
			textBuf.WriteString(block.Text)
		case "tool_use":
			if toolBlock == nil {
				toolBlock = block
			}
		}
	}

	usage := Usage{
		Input:         parsed.Usage.InputTokens,
		Output:        parsed.Usage.OutputTokens,
		CacheRead:     parsed.Usage.CacheReadInputTokens,
		CacheCreation: parsed.Usage.CacheCreationInputTokens,
	}

	if toolBlock != nil {
		toolName := canonicalToolName(toolBlock.Name, c.settings.Tools)
		if toolName == "" {
			return c.fail(ctx, handler, in.Context,
				fmt.Errorf("model picked unknown tool %q (available: %s)", toolBlock.Name, toolNames(c.settings.Tools)),
				false)
		}
		// Append the assistant's tool_use turn to history so the
		// caller can feed it back with a tool_result for the next
		// llm_tools call.
		updated := append([]Message(nil), in.Messages...)
		updated = append(updated, Message{
			Role:    "assistant",
			Content: parsed.Content, // include text + tool_use blocks verbatim
		})
		return handler(ctx, outPortFor(toolName), ToolCall{
			Context:   in.Context,
			Messages:  updated,
			ToolUseID: toolBlock.ID,
			Input:     toolBlock.Input,
		})
	}

	// No tool picked — model produced final text. Return it plus
	// updated history so the caller can keep chatting in the next turn.
	updated := append([]Message(nil), in.Messages...)
	updated = append(updated, Message{
		Role:    "assistant",
		Content: textBuf.String(),
	})
	return handler(ctx, ResponsePort, Response{
		Context:  in.Context,
		Text:     textBuf.String(),
		Messages: updated,
		Model:    parsed.Model,
		Usage:    usage,
	})
}

// messagesToAPI converts the caller's Message slice into the
// Anthropic API shape. We pass Content through unchanged — Anthropic
// accepts both string content and arrays of content blocks, which
// matches what callers send during ReAct loops.
func messagesToAPI(msgs []Message) []apiMessage {
	out := make([]apiMessage, len(msgs))
	for i, m := range msgs {
		out[i] = apiMessage{Role: m.Role, Content: m.Content}
	}
	return out
}

func toolsToAPI(tools []Tool) []apiToolDef {
	out := make([]apiToolDef, len(tools))
	for i, t := range tools {
		out[i] = apiToolDef{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		}
	}
	return out
}

func canonicalToolName(picked string, tools []Tool) string {
	want := strings.ToLower(strings.TrimSpace(picked))
	for _, t := range tools {
		if strings.ToLower(t.Name) == want {
			return t.Name
		}
	}
	return ""
}

func outPortFor(toolName string) string {
	return outPortPrefix + strings.ToLower(toolName)
}

func toolNames(tools []Tool) string {
	names := make([]string, len(tools))
	for i, t := range tools {
		names[i] = t.Name
	}
	return strings.Join(names, ", ")
}

func (c *Component) fail(ctx context.Context, handler module.Handler, reqCtx Context, err error, retryable bool) module.Result {
	if !retryable {
		err = perrors.NewPermanentError(err)
	}
	if !c.settings.EnableErrorPort {
		return module.Fail(err)
	}
	return handler(ctx, ErrorPort, Error{
		Context:   reqCtx,
		Error:     err.Error(),
		Retryable: retryable,
	})
}

func (c *Component) Ports() []module.Port {
	ports := []module.Port{
		{Name: v1alpha1.SettingsPort, Label: "Settings", Configuration: c.settings},
		{Name: RequestPort, Label: "Request", Configuration: Request{}, Position: module.Left},
		{Name: ResponsePort, Label: "Response", Source: true, Configuration: Response{}, Position: module.Right},
	}
	for _, t := range c.settings.Tools {
		ports = append(ports, module.Port{
			Name:          outPortFor(t.Name),
			Label:         t.Name,
			Source:        true,
			Configuration: ToolCall{},
			Position:      module.Right,
		})
	}
	if c.settings.EnableErrorPort {
		ports = append(ports, module.Port{
			Name: ErrorPort, Label: "Error", Source: true, Configuration: Error{}, Position: module.Bottom,
		})
	}
	return ports
}

var (
	_ module.Component       = (*Component)(nil)
	_ module.SettingsHandler = (*Component)(nil)
)

func init() {
	registry.Register(&Component{})
}
