// Package llmrouter implements llm_router — an LLM-judged routing
// component. Receives a message, asks Claude which configured route it
// belongs to, emits Context on the matching out_<route> port. Same
// shape as the deterministic router in common-module except the
// branching decision is delegated to a language model instead of a
// boolean conditions array.
//
// Use when:
//   - routing conditions are too fuzzy / numerous to enumerate as booleans
//   - you want intent classification, ticket triage, content moderation
//   - the cost of a per-message LLM call (~300-800ms, ~$0.0001-0.001 on
//     haiku) is acceptable in exchange for not maintaining hand-written
//     conditions
//
// The output ports emit only Context — same as the deterministic
// router — so downstream edges treat llm_router identically. The LLM's
// reasoning + confidence land in the trace span attributes for
// observability, not in the payload.
package llmrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/tiny-systems/llm-module/internal/stepcache"
	"github.com/tiny-systems/module/api/v1alpha1"
	"github.com/tiny-systems/module/module"
	perrors "github.com/tiny-systems/module/pkg/errors"
	"github.com/tiny-systems/module/pkg/secret"
	"github.com/tiny-systems/module/registry"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const (
	ComponentName    = "llm_router"
	RequestPort      = "request"
	DefaultPort      = "default"
	ErrorPort        = "error"
	anthropicURL     = "https://api.anthropic.com/v1/messages"
	anthropicVersion = "2023-06-01"
	defaultModel     = "claude-haiku-4-5"
	defaultMaxTokens = 256
	defaultTimeout   = 30 * time.Second

	outPortPrefix = "out_"
)

type Context any

// Route is one branch the LLM may choose. Name becomes the output port
// (lowercased, prefixed with out_). Description tells the LLM when to
// pick this route — the model literally sees these strings in the
// classification prompt.
type Route struct {
	Name        string `json:"name" required:"true" minLength:"1" title:"Name" description:"Becomes output port out_<lowercase(name)>"`
	Description string `json:"description" required:"true" minLength:"1" title:"Description" description:"Tells the LLM when to pick this route. Be specific."`
}

type Settings struct {
	EnableErrorPort     bool    `json:"enableErrorPort" required:"true" title:"Enable Error Port" description:"Route LLM/API failures to error port instead of failing"`
	EnableDefaultPort   bool    `json:"enableDefaultPort" required:"true" title:"Enable Default Port" description:"Route low-confidence / unmatched messages to a default output port. If false, the top-scoring route always wins regardless of confidence."`
	Routes              []Route `json:"routes" required:"true" minItems:"2" uniqueItems:"true" title:"Routes" description:"Available routes for the LLM to pick from. At least two."`
	APIKey              string  `json:"apiKey" title:"API Key" format:"password" description:"Preferred: set [[secret:name/key]] here so the Anthropic key is resolved against a Kubernetes Secret in the llm-module pod's namespace and never enters the flow data. Overrides Request.apiKey when set. Requires the helm release to be installed with secrets.enabled=true."`
	Model               string  `json:"model" required:"true" minLength:"1" default:"claude-haiku-4-5" title:"Model" description:"Claude model. Default haiku is cheap and fast for classification."`
	SystemPrompt        string  `json:"systemPrompt" title:"System Prompt" format:"textarea" description:"Optional task framing for the LLM (e.g. 'You are triaging support tickets')."`
	ConfidenceThreshold float64 `json:"confidenceThreshold" minimum:"0" maximum:"1" default:"0" title:"Confidence Threshold" description:"Below this, route to default (if enabled). 0 disables the threshold check."`
	TimeoutSeconds      int     `json:"timeoutSeconds" minimum:"1" default:"30" title:"Timeout Seconds"`
}

type Request struct {
	Context Context `json:"context,omitempty" configurable:"true" title:"Context" description:"Passthrough — emitted unchanged on the chosen output port"`
	APIKey  string  `json:"apiKey,omitempty" title:"Anthropic API Key" format:"password" description:"Leave empty when Settings.APIKey is set with a secret reference (recommended) — Settings.APIKey takes precedence."`
	Message string  `json:"message" required:"true" minLength:"1" title:"Message" format:"textarea" description:"Text the LLM judges to pick a route"`
}

type Error struct {
	Context   Context `json:"context,omitempty" configurable:"true" title:"Context"`
	Error     string  `json:"error" title:"Error"`
	Retryable bool    `json:"retryable" title:"Retryable" description:"True for 429 (rate limit) or 529 (overloaded). Caller may retry with backoff."`
}

type Component struct {
	module.Base
	settings Settings
}

func (c *Component) Instance() module.Component {
	return &Component{settings: Settings{
		Model:          defaultModel,
		TimeoutSeconds: int(defaultTimeout / time.Second),
	}}
}

func (c *Component) GetInfo() module.ComponentInfo {
	return module.ComponentInfo{
		Name:        ComponentName,
		Description: "LLM Router",
		Info: "Route a message to one of N output ports based on LLM judgement. Configure Settings.Routes " +
			"with {name, description} pairs — the model picks the best match per incoming message. Each route " +
			"becomes an out_<lowercase(name)> output port. When EnableDefaultPort is true and confidence is " +
			"below ConfidenceThreshold, routes to 'default'. Use for fuzzy intent classification, ticket triage, " +
			"content moderation — anywhere boolean conditions would be too many to enumerate. The reasoning + " +
			"confidence land in trace span attributes for observability. Output ports emit Context only — same " +
			"shape as the deterministic router so downstream edges treat both identically.",
		Tags: []string{"LLM", "Anthropic", "Router", "Classify"},
	}
}

func (c *Component) OnSettings(ctx context.Context, msg any) error {
	in, ok := msg.(Settings)
	if !ok {
		return fmt.Errorf("invalid settings")
	}
	// Catch empty / duplicate route names at settings-time so a bad
	// config surfaces as TinyNode.Status.Error, not as a confused
	// runtime error per message.
	if len(in.Routes) < 2 {
		return fmt.Errorf("at least two routes required")
	}
	seen := map[string]bool{}
	for i, r := range in.Routes {
		if strings.TrimSpace(r.Name) == "" {
			return fmt.Errorf("routes[%d]: empty name", i)
		}
		key := strings.ToLower(r.Name)
		if seen[key] {
			return fmt.Errorf("routes[%d]: duplicate name %q (case-insensitive)", i, r.Name)
		}
		seen[key] = true
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
	return c.route(ctx, handler, in)
}

// decision is what the LLM returns. We constrain it with a JSON
// instruction in the prompt; the parser validates the route name
// against settings to defend against the model hallucinating one.
type decision struct {
	Route      string  `json:"route"`
	Confidence float64 `json:"confidence"`
	Reasoning  string  `json:"reasoning"`
}

func (c *Component) route(ctx context.Context, handler module.Handler, in Request) module.Result {
	span := trace.SpanFromContext(ctx)

	model := c.settings.Model
	if model == "" {
		model = defaultModel
	}

	// Durable-run replay guard: a hop re-executed after a pod death reuses
	// the routing decision its previous execution already paid for; the port
	// dispatch below is deterministic on it. Without this, a kill between the
	// paid call and the step-ledger write re-bills the router's LLM call.
	dec, cached := stepcache.Get[decision](ctx, c.State())
	var usage usageStats
	if !cached {
		timeout := time.Duration(c.settings.TimeoutSeconds) * time.Second
		if timeout <= 0 {
			timeout = defaultTimeout
		}

		prompt := buildRouterPrompt(c.settings.SystemPrompt, c.settings.Routes, in.Message)

		apiKey := c.settings.APIKey
		if apiKey == "" {
			apiKey = in.APIKey
		}
		if apiKey == "" {
			return c.fail(ctx, handler, in.Context, fmt.Errorf("api key missing: set Settings.APIKey (preferred, with [[secret:...]] reference) or Request.APIKey"), false)
		}

		d, u, cerr := callClaudeForDecision(ctx, apiKey, model, timeout, prompt)
		if cerr != nil {
			return c.fail(ctx, handler, in.Context, cerr.err, cerr.retryable)
		}
		dec = d
		usage = u
		// Cache the moment the paid call returns — before any downstream dispatch.
		stepcache.Put(ctx, c.State(), dec)
	}

	chosenName := pickValidRoute(dec.Route, c.settings.Routes)
	if chosenName == "" {
		return c.fail(ctx, handler, in.Context,
			fmt.Errorf("LLM picked unknown route %q (available: %s)", dec.Route, routeNames(c.settings.Routes)),
			false)
	}

	// Surface decision metadata on the trace span so observers can see
	// why the router picked what it picked — not in the output payload
	// since that would change the Context-only shape downstream
	// components rely on.
	span.SetAttributes(
		attribute.String("llm_router.chosen", chosenName),
		attribute.Float64("llm_router.confidence", dec.Confidence),
		attribute.String("llm_router.reasoning", dec.Reasoning),
		attribute.String("llm_router.model", model),
		attribute.Int("llm_router.input_tokens", usage.input),
		attribute.Int("llm_router.output_tokens", usage.output),
	)

	if c.settings.EnableDefaultPort && c.settings.ConfidenceThreshold > 0 &&
		dec.Confidence < c.settings.ConfidenceThreshold {
		return handler(ctx, DefaultPort, in.Context)
	}

	return handler(ctx, outPortFor(chosenName), in.Context)
}

// buildRouterPrompt builds the user-message text we send to Claude.
// System framing goes on a separate role; this is the per-call payload.
func buildRouterPrompt(systemPrompt string, routes []Route, message string) string {
	var b strings.Builder
	b.WriteString("Available routes:\n")
	for _, r := range routes {
		b.WriteString("- ")
		b.WriteString(r.Name)
		b.WriteString(": ")
		b.WriteString(r.Description)
		b.WriteString("\n")
	}
	b.WriteString("\nMessage to route:\n")
	b.WriteString(message)
	b.WriteString("\n\nRespond ONLY with a JSON object — no prose, no markdown — using exactly this shape:\n")
	b.WriteString(`{"route": "<name from list>", "confidence": <0-1>, "reasoning": "<one short sentence>"}`)
	return b.String()
}

// pickValidRoute returns the canonical route name (matching one in
// Settings.Routes) for the LLM's response, or empty if no match.
// Matching is case-insensitive to tolerate model variance.
func pickValidRoute(chosen string, routes []Route) string {
	want := strings.ToLower(strings.TrimSpace(chosen))
	for _, r := range routes {
		if strings.ToLower(r.Name) == want {
			return r.Name
		}
	}
	return ""
}

func outPortFor(routeName string) string {
	return outPortPrefix + strings.ToLower(routeName)
}

func routeNames(routes []Route) string {
	names := make([]string, len(routes))
	for i, r := range routes {
		names[i] = r.Name
	}
	return strings.Join(names, ", ")
}

// --- Anthropic API plumbing (mirrors llmcomplete) -----------------

type apiTextBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type apiMessage struct {
	Role    string         `json:"role"`
	Content []apiTextBlock `json:"content"`
}

type apiRequest struct {
	Model     string         `json:"model"`
	MaxTokens int            `json:"max_tokens"`
	System    []apiTextBlock `json:"system,omitempty"`
	Messages  []apiMessage   `json:"messages"`
}

type apiResponseUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type apiResponseContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type apiResponse struct {
	Content []apiResponseContent `json:"content"`
	Usage   apiResponseUsage     `json:"usage"`
}

type apiError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type apiErrorResponse struct {
	Error apiError `json:"error"`
}

type usageStats struct {
	input  int
	output int
}

type llmCallError struct {
	err       error
	retryable bool
}

func callClaudeForDecision(ctx context.Context, apiKey, model string, timeout time.Duration, userPrompt string) (decision, usageStats, *llmCallError) {
	body := apiRequest{
		Model:     model,
		MaxTokens: defaultMaxTokens,
		Messages:  []apiMessage{{Role: "user", Content: []apiTextBlock{{Type: "text", Text: userPrompt}}}},
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return decision{}, usageStats{}, &llmCallError{err: err}
	}

	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, anthropicURL, bytes.NewReader(payload))
	if err != nil {
		return decision{}, usageStats{}, &llmCallError{err: err}
	}
	httpReq.Header.Set("content-type", "application/json")
	httpReq.Header.Set("x-api-key", apiKey)
	httpReq.Header.Set("anthropic-version", anthropicVersion)

	resp, err := (&http.Client{}).Do(httpReq)
	if err != nil {
		return decision{}, usageStats{}, &llmCallError{err: err, retryable: true}
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return decision{}, usageStats{}, &llmCallError{err: err}
	}

	if resp.StatusCode >= 400 {
		retryable := resp.StatusCode == http.StatusTooManyRequests ||
			resp.StatusCode == 529 || resp.StatusCode >= 500
		var apiErr apiErrorResponse
		if jsonErr := json.Unmarshal(respBody, &apiErr); jsonErr == nil && apiErr.Error.Message != "" {
			return decision{}, usageStats{}, &llmCallError{
				err:       fmt.Errorf("%s: %s", apiErr.Error.Type, apiErr.Error.Message),
				retryable: retryable,
			}
		}
		return decision{}, usageStats{}, &llmCallError{
			err:       fmt.Errorf("anthropic api: status %d: %s", resp.StatusCode, string(respBody)),
			retryable: retryable,
		}
	}

	var parsed apiResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return decision{}, usageStats{}, &llmCallError{err: fmt.Errorf("decode response: %w", err)}
	}

	// Concat all text blocks the model produced. Models usually return
	// one block but the API supports multiple.
	var text strings.Builder
	for _, block := range parsed.Content {
		if block.Type == "text" {
			text.WriteString(block.Text)
		}
	}
	jsonText := extractJSON(text.String())

	var dec decision
	if err := json.Unmarshal([]byte(jsonText), &dec); err != nil {
		return decision{}, usageStats{}, &llmCallError{
			err: fmt.Errorf("LLM did not return valid JSON: %w (raw: %s)", err, text.String()),
		}
	}
	return dec, usageStats{input: parsed.Usage.InputTokens, output: parsed.Usage.OutputTokens}, nil
}

// extractJSON tolerates models that wrap JSON in code fences or add a
// leading sentence despite our instructions. Returns the substring
// from the first { to the matching closing }.
func extractJSON(s string) string {
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start == -1 || end == -1 || end < start {
		return s
	}
	return s[start : end+1]
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
	}
	// One source port per declared route. Matches the deterministic
	// router's pattern so authors get a familiar surface.
	for _, r := range c.settings.Routes {
		ports = append(ports, module.Port{
			Name:          outPortFor(r.Name),
			Label:         r.Name,
			Source:        true,
			Configuration: new(Context),
			Position:      module.Right,
		})
	}
	if c.settings.EnableDefaultPort {
		ports = append(ports, module.Port{
			Name:          DefaultPort,
			Label:         "Default",
			Source:        true,
			Configuration: new(Context),
			Position:      module.Right,
		})
	}
	if c.settings.EnableErrorPort {
		ports = append(ports, module.Port{
			Name:          ErrorPort,
			Label:         "Error",
			Source:        true,
			Configuration: Error{},
			Position:      module.Bottom,
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
