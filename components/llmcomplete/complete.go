package llmcomplete

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
	"github.com/tiny-systems/module/registry"
)

const (
	ComponentName    = "llm_complete"
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

type Settings struct {
	EnableErrorPort bool    `json:"enableErrorPort" required:"true" title:"Enable Error Port"`
	Model           string  `json:"model" required:"true" minLength:"1" default:"claude-haiku-4-5" title:"Model" description:"Anthropic model ID (e.g. claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-7)"`
	SystemPrompt    string  `json:"systemPrompt" title:"System Prompt" format:"textarea" description:"Sent as the system role on every request"`
	CacheSystem     bool    `json:"cacheSystem" title:"Cache System Prompt" description:"Mark the system prompt as ephemeral so subsequent identical calls hit the prompt cache"`
	MaxTokens       int     `json:"maxTokens" required:"true" minimum:"1" default:"1024" title:"Max Tokens" description:"Maximum output tokens"`
	Temperature     float64 `json:"temperature" minimum:"0" maximum:"1" title:"Temperature"`
	TimeoutSeconds  int     `json:"timeoutSeconds" minimum:"1" default:"60" title:"Timeout Seconds"`
}

type Request struct {
	Context     Context `json:"context,omitempty" configurable:"true" title:"Context"`
	APIKey      string  `json:"apiKey" required:"true" minLength:"1" title:"Anthropic API Key" format:"password"`
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
	Retryable bool    `json:"retryable" title:"Retryable" description:"True for 429 (rate limit) or 529 (overloaded). Caller may retry with backoff."`
}

type Component struct {
	settings Settings
}

func (c *Component) Instance() module.Component {
	return &Component{
		settings: Settings{
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
		Info:        "Single-turn completion via the Anthropic Messages API. Supports prompt caching on the system prompt for cost-efficient repeated calls. Emits text, model, usage, and stop reason on success; routes 429/529 errors with retryable=true so upstream can decide whether to retry.",
		Tags:        []string{"LLM", "Anthropic", "Claude"},
	}
}

// OnSettings stores the component settings.
func (c *Component) OnSettings(_ context.Context, msg any) error {

	in, ok := msg.(Settings)
	if !ok {
		return fmt.Errorf("invalid settings")
	}
	c.settings = in
	return nil
}

// Handle dispatches the RequestPort. System ports go through capabilities.
func (c *Component) Handle(ctx context.Context, handler module.Handler, port string, msg any) any {
	if port != RequestPort {
		return fmt.Errorf("unknown port: %s", port)
	}

	in, ok := msg.(Request)
	if !ok {
		return fmt.Errorf("invalid request")
	}
	return c.complete(ctx, handler, in)
}

type apiTextBlock struct {
	Type         string         `json:"type"`
	Text         string         `json:"text"`
	CacheControl *cacheControl  `json:"cache_control,omitempty"`
}

type cacheControl struct {
	Type string `json:"type"`
}

type apiMessage struct {
	Role    string         `json:"role"`
	Content []apiTextBlock `json:"content"`
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
	Type  string   `json:"type"`
	Error apiError `json:"error"`
}

func (c *Component) complete(ctx context.Context, handler module.Handler, in Request) any {
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

	body := apiRequest{
		Model:       model,
		MaxTokens:   maxTokens,
		Temperature: c.settings.Temperature,
		Messages: []apiMessage{
			{Role: "user", Content: []apiTextBlock{{Type: "text", Text: in.UserMessage}}},
		},
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
		return c.fail(ctx, handler, in.Context, err, isRetryableNetwork(err))
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return c.fail(ctx, handler, in.Context, err, false)
	}

	if resp.StatusCode >= 400 {
		var apiErr apiErrorResponse
		if jsonErr := json.Unmarshal(respBody, &apiErr); jsonErr == nil && apiErr.Error.Message != "" {
			return c.fail(ctx, handler, in.Context, fmt.Errorf("%s: %s", apiErr.Error.Type, apiErr.Error.Message), isRetryableStatus(resp.StatusCode))
		}
		return c.fail(ctx, handler, in.Context, fmt.Errorf("anthropic api: status %d: %s", resp.StatusCode, string(respBody)), isRetryableStatus(resp.StatusCode))
	}

	var parsed apiResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return c.fail(ctx, handler, in.Context, fmt.Errorf("decode response: %w", err), false)
	}

	text := ""
	for _, block := range parsed.Content {
		if block.Type == "text" {
			text += block.Text
		}
	}

	return handler(ctx, ResponsePort, Response{
		Context:    in.Context,
		Text:       text,
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

func isRetryableStatus(code int) bool {
	return code == http.StatusTooManyRequests || code == 529 || code >= 500
}

func isRetryableNetwork(err error) bool {
	// Timeouts and transient connection errors are retryable.
	// We don't introspect deeply; treat any network-level failure as retryable.
	return err != nil
}

func (c *Component) fail(ctx context.Context, handler module.Handler, reqCtx Context, err error, retryable bool) any {
	if !c.settings.EnableErrorPort {
		return err
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
