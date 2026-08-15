package mcpcall

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/tiny-systems/llm-module/internal/mcpclient"
	"github.com/tiny-systems/module/api/v1alpha1"
	"github.com/tiny-systems/module/module"
	"github.com/tiny-systems/module/registry"
)

const (
	ComponentName = "mcp_call"
	RequestPort   = "request"
	ResultPort    = "result"
	ErrorPort     = "error"
)

type Context any

type Settings struct {
	ServerURL       string           `json:"serverURL" required:"true" title:"Server URL" description:"MCP server endpoint, e.g. https://api.githubcopilot.com/mcp/ — must speak Streamable HTTP."`
	Headers         []mcpclient.Header `json:"headers,omitempty" title:"Extra Headers" description:"Non-secret headers sent with every request (version pins etc.). Never put credentials here — settings are stored with the flow; carry auth per request via Token or Request.headers."`
	TimeoutSeconds  int              `json:"timeoutSeconds" required:"true" title:"Timeout (seconds)"`
	EnableErrorPort bool             `json:"enableErrorPort" title:"Enable Error Port" description:"Emit failures on the error port instead of failing the flow. A tool that reports its own error still arrives on the result port with isError set."`
}

type Request struct {
	Context   Context `json:"context,omitempty" configurable:"true" title:"Context" description:"Passed through to the result unchanged. Carry loop state here — llm_tools' toolUseId and messages — so the fold step can rebuild the conversation."`
	Tool      string  `json:"tool" required:"true" title:"Tool" description:"Remote tool name, exactly as mcp_tools reported it."`
	Arguments any     `json:"arguments,omitempty" configurable:"true" title:"Arguments" description:"Tool arguments, matching that tool's inputSchema. Map llm_tools' {{$.input}} straight in."`
	Token     string  `json:"token,omitempty" format:"password" title:"Bearer Token" description:"Carried per-request from the trigger widget the user fills, so the credential is not stored in the flow."`
	Headers   []mcpclient.Header `json:"headers,omitempty" title:"Extra Headers" description:"Per-request headers, merged over Settings.Headers (same key wins). The place for custom auth headers, so the secret rides the message instead of being stored in the flow."`
}

type Result struct {
	Context Context `json:"context,omitempty" title:"Context"`
	Tool    string  `json:"tool" title:"Tool"`
	// Text is every text block of the response joined by newlines. This is what
	// goes back into an llm_tools loop, whose tool messages carry a string.
	Text string `json:"text" title:"Text" description:"Text content of the response, blocks joined by newlines. Feed this to the {role: tool} message when continuing an llm_tools loop."`
	// Structured is the machine-readable result when the tool returns one.
	Structured any `json:"structured,omitempty" title:"Structured" description:"Structured content, when the tool returns it. Absent otherwise."`
	// IsError distinguishes a tool that ran and reported a failure from a
	// transport failure. Both are real outcomes but only the latter is worth
	// retrying, so a tool-reported error stays on the result port.
	IsError bool `json:"isError" title:"Is Error" description:"True when the tool itself reported failure. Transport and protocol failures go to the error port instead."`
}

type Component struct {
	settings Settings
}

func (c *Component) Instance() module.Component {
	return &Component{settings: Settings{TimeoutSeconds: 30}}
}

func (c *Component) GetInfo() module.ComponentInfo {
	return module.ComponentInfo{
		Name:        ComponentName,
		Description: "MCP Call Tool",
		Info: "Invokes a tool on a remote MCP server. Request takes {tool, arguments}; Result emits {text, structured, isError} plus the context unchanged. " +
			"To use remote tools in a ReAct loop: declare them in llm_tools' Tools setting (discover them with mcp_tools), wire out_<tool> here mapping tool and {{$.input}} to arguments, " +
			"then fold Result.text back into llm_tools.request as a {role: tool, toolCallId, content} message. A tool that reports its own failure arrives on Result with isError=true; " +
			"transport failures go to the Error port and are marked retryable.",
		Tags: []string{"MCP", "Agent"},
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
		return module.Fail(fmt.Errorf("port %s is not supported", port))
	}
	in, ok := msg.(Request)
	if !ok {
		return module.Fail(fmt.Errorf("invalid message"))
	}
	if strings.TrimSpace(in.Tool) == "" {
		// A missing tool name cannot be fixed by retrying.
		return c.handleError(ctx, handler, in.Context, module.Permanent(fmt.Errorf("tool name is required")))
	}

	sess, err := mcpclient.Open(ctx, mcpclient.Config{
		Endpoint:       c.settings.ServerURL,
		BearerToken:    in.Token,
		Headers:        mcpclient.HeaderMap(append(append([]mcpclient.Header{}, c.settings.Headers...), in.Headers...)),
		TimeoutSeconds: c.settings.TimeoutSeconds,
	})
	if err != nil {
		return c.handleError(ctx, handler, in.Context, err)
	}
	defer sess.Close()

	res, err := sess.CallTool(ctx, &mcp.CallToolParams{
		Name:      in.Tool,
		Arguments: in.Arguments,
	})
	if err != nil {
		// A protocol-level failure: the call never produced a result. Transient
		// causes (timeout, server restart) dominate here, so let retry supervise.
		return c.handleError(ctx, handler, in.Context, module.Retryable(fmt.Errorf("call tool %q: %w", in.Tool, err)))
	}

	return handler(ctx, ResultPort, Result{
		Context:    in.Context,
		Tool:       in.Tool,
		Text:       textOf(res.Content),
		Structured: res.StructuredContent,
		IsError:    res.IsError,
	})
}

// textOf joins the text blocks of a tool response.
//
// Content is an interface over text, image, audio, resource-link and embedded
// resource blocks. Only text has a flow-usable representation today, so
// non-text blocks are rendered as their JSON rather than dropped silently —
// a flow that receives an image at least sees that something arrived.
func textOf(content []mcp.Content) string {
	if len(content) == 0 {
		return ""
	}
	parts := make([]string, 0, len(content))
	for _, block := range content {
		switch b := block.(type) {
		case *mcp.TextContent:
			parts = append(parts, b.Text)
		default:
			if raw, err := json.Marshal(block); err == nil {
				parts = append(parts, string(raw))
			}
		}
	}
	return strings.Join(parts, "\n")
}

func (c *Component) handleError(ctx context.Context, handler module.Handler, reqContext Context, err error) module.Result {
	if !c.settings.EnableErrorPort {
		// Bubble unchanged so the retryability marked at the call site survives.
		return module.Fail(err)
	}
	return handler(ctx, ErrorPort, module.NewError(reqContext, err))
}

func (c *Component) Ports() []module.Port {
	ports := []module.Port{
		{
			Name:          v1alpha1.SettingsPort,
			Label:         "Settings",
			Configuration: c.settings,
		},
		{
			Name:          RequestPort,
			Label:         "Request",
			Position:      module.Left,
			Configuration: Request{},
		},
		{
			Name:     ResultPort,
			Label:    "Result",
			Source:   true,
			Position: module.Right,
			// Concrete example so an edge reading $.text is checkable at build
			// time rather than resolving to null at runtime.
			Configuration: Result{Text: "tool output"},
		},
	}
	if !c.settings.EnableErrorPort {
		return ports
	}
	return append(ports, module.Port{
		Name:          ErrorPort,
		Label:         "Error",
		Source:        true,
		Position:      module.Bottom,
		Configuration: module.ErrorMessage{},
	})
}

var (
	_ module.Component       = (*Component)(nil)
	_ module.SettingsHandler = (*Component)(nil)
)

func init() {
	registry.Register((&Component{}).Instance())
}
