// Package llmchat implements llm_chat — a stateless multi-turn
// conversation primitive over the Anthropic Messages API. The caller
// supplies the full conversation history per call; this component
// makes the API call, appends the assistant reply, and returns the
// updated history. Persistence is the caller's concern: typically
// document_store or kv loads messages by conversationId before
// llm_chat fires and saves the updated messages after.
//
// Why stateless: keeps the component a pure function from (messages,
// model) → response, swappable across storage backends without code
// changes. The "4 edges per turn" cost of composing get/llm_chat/put
// in the flow is the trade-off for keeping the primitive simple and
// keeping the persistence layer interchangeable. A higher-level
// macro-component can be built later that bundles load + chat + save
// under one node.
package llmchat

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/tiny-systems/module/api/v1alpha1"
	"github.com/tiny-systems/module/module"
	perrors "github.com/tiny-systems/module/pkg/errors"
	"github.com/tiny-systems/module/registry"
)

const (
	ComponentName    = "llm_chat"
	RequestPort      = "request"
	ResponsePort     = "response"
	ErrorPort        = "error"
	anthropicURL     = "https://api.anthropic.com/v1/messages"
	anthropicVersion = "2023-06-01"
	defaultModel     = "claude-haiku-4-5"
	defaultMaxTokens = 1024
	defaultTimeout   = 60 * time.Second
)

type Context any

// Message is one turn in the conversation. Content is a plain string
// for normal user/assistant turns. For tool-use interleavings (when
// composing with llm_tools) Content can be a list of content blocks,
// but llm_chat itself doesn't introspect Content — it forwards
// whatever shape the caller assembled.
type Message struct {
	Role    string `json:"role" title:"Role" description:"'user' or 'assistant'"`
	Content any    `json:"content" title:"Content" description:"String for plain text, or list of content blocks if mixing with tool use."`
}

type Settings struct {
	EnableErrorPort bool    `json:"enableErrorPort" required:"true" title:"Enable Error Port"`
	Model           string  `json:"model" required:"true" minLength:"1" default:"claude-haiku-4-5" title:"Model"`
	SystemPrompt    string  `json:"systemPrompt" title:"System Prompt" format:"textarea" description:"Frames the assistant's role and behaviour across all turns."`
	CacheSystem     bool    `json:"cacheSystem" title:"Cache System Prompt" description:"Mark the system prompt as ephemeral so identical subsequent calls hit Anthropic's prompt cache. Big win for long system prompts."`
	MaxTokens       int     `json:"maxTokens" required:"true" minimum:"1" default:"1024" title:"Max Tokens"`
	Temperature     float64 `json:"temperature" minimum:"0" maximum:"1" title:"Temperature"`
	TimeoutSeconds  int     `json:"timeoutSeconds" minimum:"1" default:"60" title:"Timeout Seconds"`
}

type Request struct {
	Context  Context   `json:"context,omitempty" configurable:"true" title:"Context" description:"Passthrough emitted on whichever output port fires."`
	APIKey   string    `json:"apiKey" required:"true" minLength:"1" title:"Anthropic API Key" format:"password"`
	Messages []Message `json:"messages" required:"true" minItems:"1" title:"Messages" description:"Full conversation history, ending with the new user turn. Load from your storage component before this call; save Response.messages back after."`
}

type Usage struct {
	Input         int `json:"input"`
	Output        int `json:"output"`
	CacheRead     int `json:"cacheRead"`
	CacheCreation int `json:"cacheCreation"`
}

type Response struct {
	Context    Context   `json:"context"`
	Text       string    `json:"text" description:"The assistant's reply text. Convenience field — also present as the last entry in Messages."`
	Messages   []Message `json:"messages" description:"Input messages plus the assistant's reply appended. Save this back to your storage component to continue the conversation later."`
	Model      string    `json:"model"`
	StopReason string    `json:"stopReason"`
	Usage      Usage     `json:"usage"`
}

type Error struct {
	Context   Context `json:"context"`
	Error     string  `json:"error"`
	Retryable bool    `json:"retryable" description:"True for 429 (rate limit), 529 (overloaded), or network errors. Caller may retry with backoff."`
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
		Description: "LLM Chat",
		Info: "Stateless multi-turn conversation primitive. Caller supplies the full Messages history per " +
			"call; component makes the API call and emits the updated history (with the assistant turn appended) " +
			"on Response.Messages. Persist via document_store or kv around llm_chat: load → llm_chat → save. " +
			"For tool-using agents, llm_tools is the right primitive; llm_chat is for pure conversation. " +
			"Caches the system prompt when CacheSystem=true so long system prompts amortise across turns.",
		Tags: []string{"LLM", "Anthropic", "Chat", "Conversation"},
	}
}

func (c *Component) OnSettings(_ context.Context, msg any) error {
	in, ok := msg.(Settings)
	if !ok {
		return fmt.Errorf("invalid settings")
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
	return c.chat(ctx, handler, in)
}

// --- Anthropic API plumbing --------------------------------------

type apiTextBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text,omitempty"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}

type cacheControl struct {
	Type string `json:"type"`
}

type apiMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type apiRequest struct {
	Model       string         `json:"model"`
	MaxTokens   int            `json:"max_tokens"`
	Temperature float64        `json:"temperature,omitempty"`
	System      []apiTextBlock `json:"system,omitempty"`
	Messages    []apiMessage   `json:"messages"`
}

type apiResponseUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

type apiResponseContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type apiResponse struct {
	Model      string               `json:"model"`
	Content    []apiResponseContent `json:"content"`
	StopReason string               `json:"stop_reason"`
	Usage      apiResponseUsage     `json:"usage"`
}

type apiError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type apiErrorResponse struct {
	Error apiError `json:"error"`
}

func (c *Component) chat(ctx context.Context, handler module.Handler, in Request) module.Result {
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
	}
	if c.settings.SystemPrompt != "" {
		block := apiTextBlock{Type: "text", Text: c.settings.SystemPrompt}
		if c.settings.CacheSystem {
			block.CacheControl = &cacheControl{Type: "ephemeral"}
		}
		body.System = []apiTextBlock{block}
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

	// Concat text blocks. llm_chat doesn't support tool-use response
	// blocks intentionally — use llm_tools for that path.
	var textBuf string
	for _, block := range parsed.Content {
		if block.Type == "text" {
			textBuf += block.Text
		}
	}

	// Append the assistant turn so the caller can persist the updated
	// history without having to know about Anthropic's response shape.
	updated := append([]Message(nil), in.Messages...)
	updated = append(updated, Message{Role: "assistant", Content: textBuf})

	return handler(ctx, ResponsePort, Response{
		Context:    in.Context,
		Text:       textBuf,
		Messages:   updated,
		Model:      parsed.Model,
		StopReason: parsed.StopReason,
		Usage: Usage{
			Input:         parsed.Usage.InputTokens,
			Output:        parsed.Usage.OutputTokens,
			CacheRead:     parsed.Usage.CacheReadInputTokens,
			CacheCreation: parsed.Usage.CacheCreationInputTokens,
		},
	})
}

func messagesToAPI(msgs []Message) []apiMessage {
	out := make([]apiMessage, len(msgs))
	for i, m := range msgs {
		out[i] = apiMessage{Role: m.Role, Content: m.Content}
	}
	return out
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
	if !c.settings.EnableErrorPort {
		return ports
	}
	return append(ports, module.Port{
		Name: ErrorPort, Label: "Error", Source: true, Configuration: Error{}, Position: module.Bottom,
	})
}

var (
	_ module.Component       = (*Component)(nil)
	_ module.SettingsHandler = (*Component)(nil)
)

func init() {
	registry.Register(&Component{})
}
