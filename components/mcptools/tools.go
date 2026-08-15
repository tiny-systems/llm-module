package mcptools

import (
	"context"
	"fmt"

	"github.com/tiny-systems/llm-module/internal/mcpclient"
	"github.com/tiny-systems/module/api/v1alpha1"
	"github.com/tiny-systems/module/module"
	"github.com/tiny-systems/module/registry"
)

const (
	ComponentName = "mcp_tools"
	RequestPort   = "request"
	ResultPort    = "result"
	ErrorPort     = "error"
)

type Context any

type Settings struct {
	ServerURL      string           `json:"serverURL" required:"true" title:"Server URL" description:"MCP server endpoint, e.g. https://api.githubcopilot.com/mcp/ — must speak Streamable HTTP."`
	Headers        []mcpclient.Header `json:"headers,omitempty" title:"Extra Headers" description:"Non-secret headers sent with every request (version pins etc.). Never put credentials here — settings are stored with the flow; carry auth per request via Token or Request.headers."`
	TimeoutSeconds int              `json:"timeoutSeconds" required:"true" title:"Timeout (seconds)"`
	EnableErrorPort bool            `json:"enableErrorPort" title:"Enable Error Port" description:"Emit failures on the error port instead of failing the flow."`
}

type Request struct {
	Context Context `json:"context,omitempty" configurable:"true" title:"Context" description:"Passed through to the result unchanged."`
	Token   string  `json:"token,omitempty" format:"password" title:"Bearer Token" description:"Carried per-request from the trigger widget the user fills, so the credential is not stored in the flow."`
	Headers []mcpclient.Header `json:"headers,omitempty" title:"Extra Headers" description:"Per-request headers, merged over Settings.Headers (same key wins). The place for custom auth headers, so the secret rides the message instead of being stored in the flow."`
}

// Tool is deliberately shaped like llm_tools' Tool ({name, description,
// inputSchema}) so a discovered tool can be pasted straight into that
// component's settings. That copy is the bridge between MCP's runtime discovery
// and llm_tools' static tool declaration, which is the one seam this module
// cannot remove: llm_tools derives an out_<tool> port per declared tool, and
// ports come from settings.
type Tool struct {
	Name        string         `json:"name" title:"Name"`
	Description string         `json:"description" title:"Description"`
	InputSchema map[string]any `json:"inputSchema" title:"Input Schema"`
}

type Result struct {
	Context Context `json:"context,omitempty" title:"Context"`
	Server  string  `json:"server" title:"Server" description:"Server name and version reported by the handshake."`
	Tools   []Tool  `json:"tools" title:"Tools" description:"Discovered tools, in llm_tools' declaration shape."`
	Count   int     `json:"count" title:"Count"`
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
		Description: "MCP Discover Tools",
		Info: "Lists the tools a remote MCP server offers. Emits them in llm_tools' declaration shape ({name, description, inputSchema}) " +
			"so they can be copied into that component's Tools setting — llm_tools needs its tools declared in settings to derive out_<tool> ports, " +
			"so discovery is a build-time step. Use mcp_call to actually invoke a tool. Provide the bearer token per-request rather than in settings.",
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

	listed, err := sess.ListTools(ctx, nil)
	if err != nil {
		// The handshake already succeeded, so the server is reachable; a failure
		// here is most likely transient (timeout, restart mid-call).
		return c.handleError(ctx, handler, in.Context, module.Retryable(fmt.Errorf("list tools: %w", err)))
	}

	out := make([]Tool, 0, len(listed.Tools))
	for _, t := range listed.Tools {
		if t == nil {
			continue
		}
		schema, err := schemaMap(t.InputSchema)
		if err != nil {
			return c.handleError(ctx, handler, in.Context,
				module.Permanent(fmt.Errorf("tool %q has an unreadable input schema: %w", t.Name, err)))
		}
		out = append(out, Tool{Name: t.Name, Description: t.Description, InputSchema: schema})
	}

	server := ""
	if init := sess.InitializeResult(); init != nil && init.ServerInfo != nil {
		server = fmt.Sprintf("%s %s", init.ServerInfo.Name, init.ServerInfo.Version)
	}

	return handler(ctx, ResultPort, Result{
		Context: in.Context,
		Server:  server,
		Tools:   out,
		Count:   len(out),
	})
}

func (c *Component) handleError(ctx context.Context, handler module.Handler, reqContext Context, err error) module.Result {
	if !c.settings.EnableErrorPort {
		// Bubble unchanged so the retryability marked above survives.
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
			// A concrete example rather than a zero value: an edge reading
			// $.tools[0].name is then checkable when the flow is built.
			Configuration: Result{
				Tools: []Tool{{
					Name:        "example_tool",
					Description: "what the tool does",
					InputSchema: map[string]any{"type": "object"},
				}},
			},
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
