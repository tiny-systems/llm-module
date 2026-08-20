// Package llmtoolresult closes the tool half of a tool-using loop.
//
// llm_tools emits a ToolCall on out_<tool> and requires a full message
// history back on request. The step between — appending the tool's output to
// that history as a {role:tool} entry — was not expressible anywhere: an edge
// configuration can build a literal array but cannot CONCATENATE one, so every
// loop reached for a js_eval node to do six lines of push().
//
// Ten of them exist across seven flows in one cluster, and llm_tools' own
// README named js_eval as the required workaround. This is that step.
package llmtoolresult

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/tiny-systems/module/api/v1alpha1"
	"github.com/tiny-systems/module/module"
	"github.com/tiny-systems/module/registry"
)

const (
	ComponentName = "llm_tool_result"

	RequestPort  = "request"
	ResponsePort = "response"
	ErrorPort    = "error"

	// RoleTool matches llm_tools' own vocabulary. The two components have to
	// agree on this string or the history is silently malformed.
	RoleTool = "tool"
)

// Context is the passthrough carried across the hop.
type Context any

// MessageToolUse mirrors llm_tools' shape so a history round-trips through
// this component unchanged.
type MessageToolUse struct {
	ID    string `json:"id" title:"Id"`
	Name  string `json:"name" title:"Name"`
	Input any    `json:"input" title:"Input"`
}

// Message is llm_tools' message, repeated rather than imported: a component
// declares its own port shapes, and coupling the two packages would make a
// change to either one a change to both.
type Message struct {
	Role       string           `json:"role" title:"Role" description:"'user' | 'assistant' | 'tool'"`
	Content    string           `json:"content,omitempty" title:"Content"`
	ToolUses   []MessageToolUse `json:"toolUses,omitempty" title:"Tool Uses"`
	ToolCallID string           `json:"toolCallId,omitempty" title:"Tool Call Id"`
}

// Request is what a tool's output is folded into.
type Request struct {
	Context   Context   `json:"context,omitempty" configurable:"true" title:"Context" description:"Passthrough. Carry the apiKey here if the next llm_tools call needs one — this component does not touch it."`
	Messages  []Message `json:"messages" required:"true" title:"Messages" description:"The history exactly as llm_tools emitted it on out_<tool>. Map it straight through: {{$.messages}}."`
	ToolUseID string    `json:"toolUseId" required:"true" title:"Tool Use Id" description:"From the same ToolCall. Identifies which call this result answers."`
	Result    any       `json:"result" title:"Result" description:"Whatever the tool produced. An object or array is JSON-encoded; a string is used as-is, so a tool that already returns text is not double-quoted."`
}

// Response is a history ready to send straight back to llm_tools.
type Response struct {
	Context  Context   `json:"context,omitempty" configurable:"true" title:"Context"`
	Messages []Message `json:"messages" title:"Messages" description:"The input history with the tool result appended. Wire to llm_tools request."`
}

type Error struct {
	Context Context `json:"context,omitempty" configurable:"true" title:"Context"`
	Error   string  `json:"error" title:"Error"`
}

type Settings struct {
	EnableErrorPort bool `json:"enableErrorPort" title:"Enable Error Port" description:"Output errors to the error port instead of failing the run."`
}

type Component struct {
	module.Base
	settings Settings
}

func (c *Component) GetInfo() module.ComponentInfo {
	return module.ComponentInfo{
		Name:        ComponentName,
		Description: "LLM Tool Result",
		Info: "Appends a tool's output to a conversation so a tool-using loop can continue. " +
			"Wire it between a tool's output and llm_tools' request port: take messages and toolUseId from the " +
			"ToolCall that llm_tools emitted, put the tool's output in result, and send the response straight back " +
			"to llm_tools. Objects and arrays are JSON-encoded, a string is passed through unchanged. " +
			"This exists because an edge configuration can build an array but cannot append to one — without it " +
			"every loop needs a js_eval doing the same six lines. Carry an apiKey through context if the next call " +
			"needs one; this component never reads it.",
		Tags: []string{"LLM", "Agent", "Tools"},
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

func (c *Component) Handle(ctx context.Context, handler module.Handler, port string, msg interface{}) module.Result {
	if port != RequestPort {
		return module.Fail(fmt.Errorf("unknown port: %s", port))
	}
	in, ok := msg.(Request)
	if !ok {
		return module.Fail(fmt.Errorf("invalid message"))
	}

	if in.ToolUseID == "" {
		return c.handleError(ctx, handler, in.Context,
			fmt.Errorf("toolUseId is required — take it from the ToolCall llm_tools emitted, or the model cannot tell which call this answers"))
	}
	if len(in.Messages) == 0 {
		return c.handleError(ctx, handler, in.Context,
			fmt.Errorf("messages is required — map the history from the ToolCall ({{$.messages}}); a tool result with no conversation to append to is not a conversation"))
	}

	content, err := renderResult(in.Result)
	if err != nil {
		return c.handleError(ctx, handler, in.Context, fmt.Errorf("tool result could not be encoded: %w", err))
	}

	// Copy rather than append in place: the caller's slice may be shared with
	// another branch of the flow, and growing it under them would corrupt a
	// history nobody else touched.
	updated := make([]Message, len(in.Messages), len(in.Messages)+1)
	copy(updated, in.Messages)
	updated = append(updated, Message{
		Role:       RoleTool,
		Content:    content,
		ToolCallID: in.ToolUseID,
	})

	return handler(ctx, ResponsePort, Response{Context: in.Context, Messages: updated})
}

// renderResult turns a tool's output into the text a model reads.
//
// A string passes through: a tool that already produced prose or JSON text
// would otherwise arrive double-encoded, quoted and escaped, which models read
// badly. Everything else is JSON, which is what a structured result is.
func renderResult(result any) (string, error) {
	switch v := result.(type) {
	case nil:
		// A tool that returned nothing still has to answer, or the model waits
		// for a result that never comes.
		return "(no output)", nil
	case string:
		return v, nil
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

func (c *Component) handleError(ctx context.Context, handler module.Handler, reqCtx Context, err error) module.Result {
	if !c.settings.EnableErrorPort {
		return module.Fail(err)
	}
	return handler(ctx, ErrorPort, Error{Context: reqCtx, Error: err.Error()})
}

func (c *Component) Ports() []module.Port {
	ports := []module.Port{
		{
			Name:          RequestPort,
			Label:         "Request",
			Configuration: Request{},
			Position:      module.Left,
		},
		{
			Name:          ResponsePort,
			Label:         "Response",
			Source:        true,
			Configuration: Response{},
			Position:      module.Right,
		},
		{
			Name:          v1alpha1.SettingsPort,
			Label:         "Settings",
			Configuration: c.settings,
		},
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

func (c *Component) Instance() module.Component {
	return &Component{}
}

var (
	_ module.Component       = (*Component)(nil)
	_ module.SettingsHandler = (*Component)(nil)
)

func init() {
	registry.Register(&Component{})
}
