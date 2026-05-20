// Package llmchat implements llm_chat — a stateless multi-turn
// conversation primitive over a configurable LLM provider.
//
// Defaults to Anthropic's Messages API with prompt-cache support on
// the system prompt. Set Settings.Provider="openai" to target the
// OpenAI Chat Completions API (or any OpenAI-compatible endpoint via
// Settings.BaseURL — Ollama, vLLM, OpenRouter, Together, Azure
// OpenAI, etc.). Output shape is unchanged across providers so
// existing flows continue to work.
//
// Stateless by design: caller supplies the full conversation history
// per call; this component makes the API call, appends the assistant
// reply, and returns the updated history. Persistence is the caller's
// concern: typically document_store or kv loads messages by
// conversationId before llm_chat fires and saves the updated messages
// after.
package llmchat

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/tiny-systems/llm-module/internal/provider"
	"github.com/tiny-systems/module/api/v1alpha1"
	"github.com/tiny-systems/module/module"
	perrors "github.com/tiny-systems/module/pkg/errors"
	"github.com/tiny-systems/module/pkg/secret"
	"github.com/tiny-systems/module/registry"
)

const (
	ComponentName    = "llm_chat"
	RequestPort      = "request"
	ResponsePort     = "response"
	ErrorPort        = "error"
	defaultModel     = "claude-haiku-4-5"
	defaultMaxTokens = 1024
	defaultTimeout   = 60 * time.Second
)

type Context any

// Message is one turn in the conversation. Content is a plain string
// for normal user/assistant turns. For tool-use interleavings (when
// composing with llm_tools) Content can be a list of content blocks,
// but llm_chat itself doesn't introspect Content — it forwards
// whatever shape the caller assembled. Tool-use blocks are only
// portable on Anthropic; switching Provider to "openai" with tool
// blocks in history is not supported.
type Message struct {
	Role    string `json:"role" title:"Role" description:"'user' or 'assistant'"`
	Content any    `json:"content" title:"Content" description:"String for plain text, or list of content blocks if mixing with tool use (Anthropic only)."`
}

type Settings struct {
	EnableErrorPort bool    `json:"enableErrorPort" required:"true" title:"Enable Error Port"`
	Provider        string  `json:"provider" required:"true" enum:"anthropic,openai" default:"anthropic" title:"Provider" description:"LLM backend. 'anthropic' uses the Messages API and supports prompt caching on the system prompt. 'openai' uses the Chat Completions API and also works with any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, Azure OpenAI, Together) via BaseURL."`
	BaseURL         string  `json:"baseURL" title:"Base URL" description:"Optional override for self-hosted or third-party endpoints. For openai-compatible servers, pass the v1 base (e.g. http://ollama:11434/v1). Leave blank for the provider default."`
	APIKey          string  `json:"apiKey" title:"API Key" format:"password" description:"Preferred: set [[secret:name/key]] here so the credential is resolved against a Kubernetes Secret in the llm-module pod's namespace and never enters the flow data. Overrides Request.apiKey when set. Requires the helm release to be installed with secrets.enabled=true."`
	Model           string  `json:"model" required:"true" minLength:"1" default:"claude-haiku-4-5" title:"Model"`
	SystemPrompt    string  `json:"systemPrompt" title:"System Prompt" format:"textarea" description:"Frames the assistant's role and behaviour across all turns."`
	CacheSystem     bool    `json:"cacheSystem" title:"Cache System Prompt" description:"Anthropic only: mark the system prompt as ephemeral so identical subsequent calls hit Anthropic's prompt cache. Big win for long system prompts. Ignored on openai."`
	MaxTokens       int     `json:"maxTokens" required:"true" minimum:"1" default:"1024" title:"Max Tokens"`
	Temperature     float64 `json:"temperature" minimum:"0" maximum:"1" title:"Temperature"`
	TimeoutSeconds  int     `json:"timeoutSeconds" minimum:"1" default:"60" title:"Timeout Seconds"`
}

type Request struct {
	Context  Context   `json:"context,omitempty" configurable:"true" title:"Context" description:"Passthrough emitted on whichever output port fires."`
	APIKey   string    `json:"apiKey,omitempty" title:"API Key" format:"password" description:"Anthropic x-api-key or OpenAI Bearer token. Leave empty when Settings.APIKey is set with a secret reference (recommended) — Settings.APIKey takes precedence."`
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
	Retryable bool    `json:"retryable" description:"True for 429 (rate limit), 529 (overloaded), 5xx, or network errors. Caller may retry with backoff."`
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
		Description: "LLM Chat",
		Info: "Stateless multi-turn conversation primitive. Caller supplies the full Messages history per " +
			"call; component makes the API call and emits the updated history (with the assistant turn appended) " +
			"on Response.Messages. Persist via document_store or kv around llm_chat: load → llm_chat → save. " +
			"Defaults to Anthropic's Messages API; switch Provider to 'openai' for OpenAI Chat Completions or " +
			"any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, Azure OpenAI) via BaseURL. " +
			"For tool-using agents, llm_tools is the right primitive; llm_chat is for pure conversation. " +
			"Caches the system prompt when CacheSystem=true (Anthropic only) so long system prompts amortise across turns.",
		Tags: []string{"LLM", "Anthropic", "OpenAI", "Chat", "Conversation"},
	}
}

func (c *Component) OnSettings(ctx context.Context, msg any) error {
	in, ok := msg.(Settings)
	if !ok {
		return fmt.Errorf("invalid settings")
	}
	if client := c.Client(); client != nil {
		if err := secret.Resolve(ctx, &in, client); err != nil {
			return fmt.Errorf("resolve secrets: %w", err)
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
	return c.chat(ctx, handler, in)
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
	maxTokens := c.settings.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
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
		return c.fail(ctx, handler, in.Context, fmt.Errorf("api key missing: set Settings.APIKey (preferred, with [[secret:...]] reference) or Request.APIKey"), false)
	}

	pmsgs := make([]provider.Message, len(in.Messages))
	for i, m := range in.Messages {
		pmsgs[i] = provider.Message{Role: m.Role, Content: m.Content}
	}

	resp, err := p.Complete(ctx, provider.CompletionRequest{
		APIKey:       apiKey,
		BaseURL:      c.settings.BaseURL,
		Model:        model,
		SystemPrompt: c.settings.SystemPrompt,
		CacheSystem:  c.settings.CacheSystem,
		Messages:     pmsgs,
		MaxTokens:    maxTokens,
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

	// Append the assistant turn so the caller can persist the updated
	// history without having to know about provider response shapes.
	updated := append([]Message(nil), in.Messages...)
	updated = append(updated, Message{Role: "assistant", Content: resp.Text})

	return handler(ctx, ResponsePort, Response{
		Context:    in.Context,
		Text:       resp.Text,
		Messages:   updated,
		Model:      resp.Model,
		StopReason: resp.StopReason,
		Usage: Usage{
			Input:         resp.Usage.Input,
			Output:        resp.Usage.Output,
			CacheRead:     resp.Usage.CacheRead,
			CacheCreation: resp.Usage.CacheCreation,
		},
	})
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
