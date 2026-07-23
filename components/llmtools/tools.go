// Package llmtools implements llm_tools — the keystone for
// ReAct / function-calling agents. It exposes the configured LLM
// provider's tool-calling API as N dynamic output ports, one per
// declared tool. Per call, the model either picks a tool (emitting
// structured args on out_<toolname>) or produces a final text response
// (emitting on the `response` port).
//
// Multi-provider since v0.8.0: Anthropic (Messages API tool_use) and
// OpenAI Chat Completions (function calling) share a normalized message
// shape so flows compose the same way regardless of backend. The wire
// shape (tool_use blocks vs tool_calls arrays) is hidden in the
// internal provider package.
//
// The component itself is stateless: each invocation is one API call.
// The caller supplies the full Messages history. When a tool is picked
// the component emits the UPDATED messages (including the assistant
// turn with that tool_use) so the next invocation can append the tool
// result and continue. This is how ReAct loops are built: tool →
// handler → next llm_tools call with appended tool result.
package llmtools

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/tiny-systems/llm-module/internal/provider"
	"github.com/tiny-systems/llm-module/internal/stepcache"
	"github.com/tiny-systems/module/api/v1alpha1"
	"github.com/tiny-systems/module/module"
	perrors "github.com/tiny-systems/module/pkg/errors"
	"github.com/tiny-systems/module/pkg/secret"
	"github.com/tiny-systems/module/registry"
)

const (
	ComponentName    = "llm_tools"
	RequestPort      = "request"
	ResponsePort     = "response"
	ErrorPort        = "error"
	defaultModel     = "claude-haiku-4-5"
	defaultMaxTokens = 1024
	defaultTimeout   = 60 * time.Second

	outPortPrefix = "out_"

	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

type Context any

// Tool declares one function the model can call. Mirrors both
// Anthropic's tool spec (name + description + input_schema) and
// OpenAI's function spec (name + description + parameters) — the
// provider translates as needed.
type Tool struct {
	Name        string         `json:"name" required:"true" minLength:"1" title:"Name" description:"Becomes output port out_<lowercase(name)>. Used by the model to identify the tool."`
	Description string         `json:"description" required:"true" minLength:"1" title:"Description" description:"Tells the model what the tool does and when to use it. Be specific."`
	InputSchema map[string]any `json:"inputSchema" required:"true" title:"Input schema" description:"JSON Schema for the tool's arguments. Example: {type: object, properties: {query: {type: string}}, required: [query]}."`
}

type Settings struct {
	EnableErrorPort bool    `json:"enableErrorPort" required:"true" title:"Enable Error Port"`
	Provider        string  `json:"provider" required:"true" enum:"anthropic,openai" default:"anthropic" title:"Provider" description:"LLM backend. 'anthropic' uses the Messages API tool_use protocol. 'openai' uses Chat Completions function calling and also targets any OpenAI-compatible endpoint via BaseURL."`
	BaseURL         string  `json:"baseURL" title:"Base URL" description:"Optional override for self-hosted or third-party endpoints. For openai-compatible servers, pass the v1 base (e.g. http://ollama:11434/v1). Leave blank for the provider default."`
	APIKey          string  `json:"apiKey" title:"API Key" format:"password" description:"API key for the provider. Overrides Request.apiKey when set. Leave EMPTY when the user supplies the key: the idiomatic shape is a masked field on the flow's trigger widget, carried here on the request edge, so nothing is provisioned per flow."`
	Tools           []Tool  `json:"tools" required:"true" minItems:"1" uniqueItems:"true" title:"Tools" description:"Tools the model may invoke. At least one."`
	Model           string  `json:"model" required:"true" minLength:"1" default:"claude-haiku-4-5" title:"Model"`
	SystemPrompt    string  `json:"systemPrompt" title:"System Prompt" format:"textarea" description:"Frames the model's behaviour across all turns."`
	CacheSystem     bool    `json:"cacheSystem" title:"Cache System Prompt" description:"Anthropic only — mark the system prompt as ephemeral so subsequent identical calls hit the prompt cache. Ignored on openai."`
	MaxTokens       int     `json:"maxTokens" required:"true" minimum:"1" default:"1024" title:"Max Tokens"`
	Temperature     float64 `json:"temperature" minimum:"0" maximum:"1" title:"Temperature"`
	TimeoutSeconds  int     `json:"timeoutSeconds" minimum:"1" default:"60" title:"Timeout Seconds"`
}

// MessageToolUse is one tool invocation the model emitted in an
// assistant turn. The pair (id, name) is what the next tool message
// references via toolCallId.
type MessageToolUse struct {
	ID    string `json:"id" title:"Id" description:"Opaque id the provider assigned. Round-trip unchanged."`
	Name  string `json:"name" title:"Name"`
	Input any    `json:"input" title:"Input" description:"Structured args per the tool's inputSchema."`
}

// Message is provider-agnostic. Three roles:
//
//   - user:      Content holds the user prompt.
//   - assistant: Content holds the model's reply text (may be empty);
//     ToolUses lists any tool invocations the model made.
//   - tool:      ToolCallId names which assistant ToolUses entry this
//     answers; Content holds the tool output as a string.
type Message struct {
	Role       string           `json:"role" title:"Role" description:"'user' | 'assistant' | 'tool'"`
	Content    string           `json:"content,omitempty" title:"Content" description:"Plain text. For 'tool' role, the stringified tool output."`
	ToolUses   []MessageToolUse `json:"toolUses,omitempty" title:"Tool Uses" description:"Assistant turn only — tool invocations the model made."`
	ToolCallID string           `json:"toolCallId,omitempty" title:"Tool Call Id" description:"'tool' role only — which ToolUses entry this is a result for."`
}

type Request struct {
	Context  Context   `json:"context,omitempty" configurable:"true" title:"Context" description:"Passthrough emitted on whichever output port fires."`
	APIKey   string    `json:"apiKey,omitempty" title:"API Key" format:"password" description:"Anthropic x-api-key or OpenAI Bearer token. Usually left empty here and carried per-request from the trigger widget the user fills (map it onto the request edge as apiKey). Settings.APIKey takes precedence if set."`
	Messages []Message `json:"messages" required:"true" minItems:"1" title:"Messages" description:"Full conversation history. Build incrementally: append the prior llm_tools response's Messages, then a {role: tool, toolCallId, content} entry for each tool result, then re-invoke."`
}

// ToolCall is the payload emitted on a tool's out_<name> port. The
// caller routes this to the matching handler, then sends the result
// back via a new llm_tools.request call with a {role:tool, ...} entry
// appended to Messages.
type ToolCall struct {
	Context   Context   `json:"context,omitempty" configurable:"true" title:"Context"`
	Messages  []Message `json:"messages" title:"Messages" description:"Updated history including the assistant turn that called this tool. Append a {role: tool, toolCallId: '<id>', content: '<output>'} and re-call to continue."`
	ToolUseID string    `json:"toolUseId" title:"Tool Use Id" description:"Use this as toolCallId on the next-turn tool message."`
	Input     any       `json:"input" title:"Input" description:"Structured arguments per the tool's inputSchema."`
}

type Usage struct {
	Input         int `json:"input"`
	Output        int `json:"output"`
	CacheRead     int `json:"cacheRead"`
	CacheCreation int `json:"cacheCreation"`
}

type Response struct {
	Context  Context   `json:"context,omitempty" configurable:"true" title:"Context"`
	Text     string    `json:"text" title:"Text"`
	Messages []Message `json:"messages" title:"Messages" description:"Final history including the assistant's text reply. Use as a base for a follow-up call if you want to continue the conversation."`
	Model    string    `json:"model"`
	Usage    Usage     `json:"usage"`
}

type Error struct {
	Context   Context `json:"context,omitempty" configurable:"true" title:"Context"`
	Error     string  `json:"error" title:"Error"`
	Retryable bool    `json:"retryable" title:"Retryable" description:"True for 429/529/5xx and network errors. Caller may retry with backoff."`
}

type Component struct {
	module.Base
	settings Settings
}

func (c *Component) Instance() module.Component {
	return &Component{settings: Settings{
		Provider:       provider.Anthropic,
		Model:          defaultModel,
		MaxTokens:      defaultMaxTokens,
		TimeoutSeconds: int(defaultTimeout / time.Second),
	}}
}

func (c *Component) GetInfo() module.ComponentInfo {
	return module.ComponentInfo{
		Name:        ComponentName,
		Description: "LLM Tools",
		Info: "ReAct / function-calling primitive. Declare tools in Settings; each becomes an out_<toolname> source port " +
			"emitting {toolUseId, input, messages} when the model picks it. Multi-provider — Anthropic Messages tool_use " +
			"(default) or OpenAI Chat Completions function calling via Provider=openai; BaseURL targets any OpenAI-compatible " +
			"endpoint (Ollama, vLLM, OpenRouter). Component is stateless: caller supplies the full Messages history. To build a " +
			"ReAct loop, wire out_<tool> → handler → another llm_tools.request with the previous response's Messages plus a " +
			"{role: tool, toolCallId, content} entry. Loop until the response port fires.",
		Tags: []string{"LLM", "Anthropic", "OpenAI", "Tools", "ReAct", "Agent"},
	}
}

func (c *Component) OnSettings(ctx context.Context, msg any) error {
	in, ok := msg.(Settings)
	if !ok {
		return fmt.Errorf("invalid settings")
	}
	if len(in.Tools) == 0 {
		return fmt.Errorf("at least one tool required")
	}
	if client := c.Client(); client != nil {
		if err := secret.Resolve(ctx, &in, client); err != nil {
			return fmt.Errorf("resolve secrets: %w", err)
		}
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

func (c *Component) invoke(ctx context.Context, handler module.Handler, in Request) module.Result {
	// Durable-run replay guard: a hop re-executed after a pod death reuses
	// the provider response its previous execution already paid for, then
	// re-runs the (deterministic) tool/response dispatch below on it. Without
	// this, a kill between the paid call and the step-ledger write re-bills.
	resp, cached := stepcache.Get[provider.ToolCompletionResponse](ctx, c.State())
	if !cached {
		timeout := time.Duration(c.settings.TimeoutSeconds) * time.Second
		if timeout <= 0 {
			timeout = defaultTimeout
		}
		model := c.settings.Model
		if model == "" {
			model = defaultModel
		}

		p, err := provider.New(c.settings.Provider)
		if err != nil {
			return c.fail(ctx, handler, in.Context, err, false)
		}

		apiKey := c.settings.APIKey
		if apiKey == "" {
			apiKey = in.APIKey
		}
		if apiKey == "" {
			apiKey = provider.EnvAPIKey(c.settings.Provider)
		}
		if apiKey == "" {
			return c.fail(ctx, handler, in.Context, fmt.Errorf("api key missing: set Settings.APIKey, or carry it per request as Request.APIKey (e.g. from the trigger widget the user fills)"), false)
		}

		r, err := p.CompleteWithTools(ctx, provider.ToolCompletionRequest{
			APIKey:       apiKey,
			BaseURL:      c.settings.BaseURL,
			Model:        model,
			SystemPrompt: c.settings.SystemPrompt,
			CacheSystem:  c.settings.CacheSystem,
			Messages:     toProviderMessages(in.Messages),
			Tools:        toProviderTools(c.settings.Tools),
			MaxTokens:    c.settings.MaxTokens,
			Temperature:  c.settings.Temperature,
			Timeout:      timeout,
		})
		if err != nil {
			var perr *provider.Error
			if errors.As(err, &perr) {
				return c.fail(ctx, handler, in.Context, perr.Err, perr.Retryable)
			}
			return c.fail(ctx, handler, in.Context, err, false)
		}
		// Cache the moment the paid call returns — before any downstream dispatch.
		resp = *r
		stepcache.Put(ctx, c.State(), resp)
	}

	usage := Usage{
		Input:         resp.Usage.Input,
		Output:        resp.Usage.Output,
		CacheRead:     resp.Usage.CacheRead,
		CacheCreation: resp.Usage.CacheCreation,
	}

	if len(resp.ToolUses) > 0 {
		// Append the assistant turn to history so the caller can build
		// the next request by appending tool result messages.
		assistantUses := make([]MessageToolUse, len(resp.ToolUses))
		for i, u := range resp.ToolUses {
			assistantUses[i] = MessageToolUse{ID: u.ID, Name: u.Name, Input: u.Input}
		}
		updated := append([]Message(nil), in.Messages...)
		updated = append(updated, Message{
			Role:     RoleAssistant,
			Content:  resp.Text,
			ToolUses: assistantUses,
		})

		// Route the first tool_use to out_<name>. If the model called
		// multiple tools in parallel, the others are surfaced via the
		// updated Messages — the caller can fan out by inspecting
		// ToolUses on the assistant turn. Routing the first keeps the
		// single-port contract intact for the common case.
		first := resp.ToolUses[0]
		toolName := canonicalToolName(first.Name, c.settings.Tools)
		if toolName == "" {
			return c.fail(ctx, handler, in.Context,
				fmt.Errorf("model picked unknown tool %q (available: %s)", first.Name, toolNames(c.settings.Tools)),
				false)
		}
		return handler(ctx, outPortFor(toolName), ToolCall{
			Context:   in.Context,
			Messages:  updated,
			ToolUseID: first.ID,
			Input:     first.Input,
		})
	}

	// No tool picked — final text reply.
	updated := append([]Message(nil), in.Messages...)
	updated = append(updated, Message{
		Role:    RoleAssistant,
		Content: resp.Text,
	})
	return handler(ctx, ResponsePort, Response{
		Context:  in.Context,
		Text:     resp.Text,
		Messages: updated,
		Model:    resp.Model,
		Usage:    usage,
	})
}

func toProviderMessages(msgs []Message) []provider.ToolMessage {
	out := make([]provider.ToolMessage, len(msgs))
	for i, m := range msgs {
		out[i] = provider.ToolMessage{
			Role:       m.Role,
			Text:       m.Content,
			ToolCallID: m.ToolCallID,
		}
		if len(m.ToolUses) > 0 {
			uses := make([]provider.ToolUse, len(m.ToolUses))
			for j, u := range m.ToolUses {
				uses[j] = provider.ToolUse{ID: u.ID, Name: u.Name, Input: u.Input}
			}
			out[i].ToolUses = uses
		}
	}
	return out
}

func toProviderTools(tools []Tool) []provider.ToolDef {
	out := make([]provider.ToolDef, len(tools))
	for i, t := range tools {
		out[i] = provider.ToolDef{
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
