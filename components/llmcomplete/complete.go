// Package llmcomplete implements llm_complete — a single-turn
// completion primitive that targets a configurable LLM provider.
//
// Defaults to Anthropic's Messages API with prompt-cache support on
// the system prompt. Set Settings.Provider="openai" to target the
// OpenAI Chat Completions API (or any OpenAI-compatible endpoint via
// Settings.BaseURL — Ollama, vLLM, OpenRouter, Together, Azure
// OpenAI, etc.). The output shape stays the same regardless of
// provider so downstream edges don't need to know which backend ran.
package llmcomplete

import (
	"context"
	"errors"
	"fmt"
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
	ComponentName    = "llm_complete"
	RequestPort      = "request"
	ResponsePort     = "response"
	ErrorPort        = "error"
	defaultModel     = "claude-haiku-4-5"
	defaultMaxTokens = 1024
	defaultTimeout   = 60 * time.Second
)

type Context any

type Settings struct {
	EnableErrorPort bool    `json:"enableErrorPort" required:"true" title:"Enable Error Port"`
	Provider        string  `json:"provider" required:"true" enum:"anthropic,openai" default:"anthropic" title:"Provider" description:"LLM backend. 'anthropic' uses the Messages API and supports prompt caching on the system prompt. 'openai' uses the Chat Completions API and also works with any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, Azure OpenAI, Together) via BaseURL."`
	BaseURL         string  `json:"baseURL" title:"Base URL" description:"Optional override for self-hosted or third-party endpoints. For openai-compatible servers, pass the v1 base (e.g. http://ollama:11434/v1). Leave blank for the provider default."`
	APIKey          string  `json:"apiKey" title:"API Key" format:"password" description:"Preferred: set [[secret:name/key]] here so the credential is resolved against a Kubernetes Secret in the llm-module pod's namespace and never enters the flow data. Overrides Request.apiKey when set. Requires the helm release to be installed with secrets.enabled=true."`
	Model           string  `json:"model" required:"true" minLength:"1" default:"claude-haiku-4-5" title:"Model" description:"Provider model ID (claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-7 for anthropic; gpt-4o-mini, gpt-4o for openai; llama3.1 etc. for ollama)."`
	SystemPrompt    string  `json:"systemPrompt" title:"System Prompt" format:"textarea" description:"Sent as the system role on every request."`
	CacheSystem     bool    `json:"cacheSystem" title:"Cache System Prompt" description:"Anthropic only: mark the system prompt as ephemeral so subsequent identical calls hit the prompt cache. Ignored on openai."`
	MaxTokens       int     `json:"maxTokens" required:"true" minimum:"1" default:"1024" title:"Max Tokens" description:"Maximum output tokens."`
	Temperature     float64 `json:"temperature" minimum:"0" maximum:"1" title:"Temperature"`
	TimeoutSeconds  int     `json:"timeoutSeconds" minimum:"1" default:"60" title:"Timeout Seconds"`
}

type Request struct {
	Context     Context `json:"context,omitempty" configurable:"true" title:"Context"`
	APIKey      string  `json:"apiKey,omitempty" title:"API Key" format:"password" description:"Anthropic x-api-key or OpenAI Bearer token. Leave empty when Settings.APIKey is set with a secret reference (recommended) — Settings.APIKey takes precedence."`
	UserMessage string  `json:"userMessage" required:"true" minLength:"1" title:"User Message" format:"textarea"`
}

type Usage struct {
	Input         int `json:"input" title:"Input Tokens"`
	Output        int `json:"output" title:"Output Tokens"`
	CacheRead     int `json:"cacheRead" title:"Cache Read Tokens"`
	CacheCreation int `json:"cacheCreation" title:"Cache Creation Tokens"`
}

type Response struct {
	Context    Context `json:"context,omitempty" configurable:"true" title:"Context"`
	Text       string  `json:"text" title:"Text"`
	Model      string  `json:"model" title:"Model"`
	StopReason string  `json:"stopReason" title:"Stop Reason"`
	Usage      Usage   `json:"usage" title:"Usage"`
}

type Error struct {
	Context   Context `json:"context,omitempty" configurable:"true" title:"Context"`
	Error     string  `json:"error" title:"Error"`
	Retryable bool    `json:"retryable" title:"Retryable" description:"True for 429 (rate limit), 529 (overloaded), 5xx, or network errors. Caller may retry with backoff."`
}

type Component struct {
	module.Base
	settings Settings
}

func (c *Component) Instance() module.Component {
	return &Component{
		settings: Settings{
			Provider:       provider.Anthropic,
			Model:          defaultModel,
			MaxTokens:      defaultMaxTokens,
			TimeoutSeconds: int(defaultTimeout / time.Second),
		},
	}
}

func (c *Component) GetInfo() module.ComponentInfo {
	return module.ComponentInfo{
		Name:        ComponentName,
		Description: "LLM Complete",
		Info:        "Single-turn completion. Defaults to Anthropic's Messages API (with prompt caching on the system prompt); switch Provider to 'openai' for OpenAI Chat Completions or any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, Azure OpenAI) via BaseURL. Emits text, model, usage, and stop reason on success; routes 429/529/5xx errors with retryable=true so upstream can decide whether to retry.",
		Tags:        []string{"LLM", "Anthropic", "OpenAI", "Claude"},
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
	return c.complete(ctx, handler, in)
}

func (c *Component) complete(ctx context.Context, handler module.Handler, in Request) module.Result {
	// Durable-run replay guard: a hop re-executed after a pod death reuses
	// the response its previous execution already paid for. Checked before
	// provider/key validation on purpose — a replay needs no credentials.
	if cached, ok := stepcache.Get[Response](ctx, c.State()); ok {
		cached.Context = in.Context
		return handler(ctx, ResponsePort, cached)
	}

	model := c.settings.Model
	if model == "" {
		model = defaultModel
	}
	maxTokens := c.settings.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	timeout := time.Duration(c.settings.TimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = defaultTimeout
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
		return c.fail(ctx, handler, in.Context, fmt.Errorf("api key missing: set Settings.APIKey (preferred, with [[secret:...]] reference) or Request.APIKey"), false)
	}

	resp, err := p.Complete(ctx, provider.CompletionRequest{
		APIKey:       apiKey,
		BaseURL:      c.settings.BaseURL,
		Model:        model,
		SystemPrompt: c.settings.SystemPrompt,
		CacheSystem:  c.settings.CacheSystem,
		Messages: []provider.Message{
			{Role: "user", Content: in.UserMessage},
		},
		MaxTokens:   maxTokens,
		Temperature: c.settings.Temperature,
		Timeout:     timeout,
	})
	if err != nil {
		var perr *provider.Error
		if errors.As(err, &perr) {
			return c.fail(ctx, handler, in.Context, perr.Err, perr.Retryable)
		}
		return c.fail(ctx, handler, in.Context, err, false)
	}

	out := Response{
		Text:       resp.Text,
		Model:      resp.Model,
		StopReason: resp.StopReason,
		Usage: Usage{
			Input:         resp.Usage.Input,
			Output:        resp.Usage.Output,
			CacheRead:     resp.Usage.CacheRead,
			CacheCreation: resp.Usage.CacheCreation,
		},
	}
	// Cache WITHOUT the request context (it re-attaches on replay), the
	// moment the paid call returns — before anything downstream can fail.
	stepcache.Put(ctx, c.State(), out)
	out.Context = in.Context

	return handler(ctx, ResponsePort, out)
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
