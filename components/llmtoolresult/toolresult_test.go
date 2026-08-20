package llmtoolresult

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/tiny-systems/module/module"
)

// capture records what a component emitted, so a test asserts on the message
// that actually left rather than on internal state.
func capture(t *testing.T, in Request, settings Settings) (string, interface{}, error) {
	t.Helper()
	c, ok := (&Component{}).Instance().(*Component)
	if !ok {
		t.Fatal("Instance() did not return *Component")
	}
	if err := c.OnSettings(context.Background(), settings); err != nil {
		t.Fatalf("settings: %v", err)
	}

	var gotPort string
	var gotMsg interface{}
	res := c.Handle(context.Background(), func(_ context.Context, port string, msg interface{}) module.Result {
		gotPort, gotMsg = port, msg
		return module.Result{}
	}, RequestPort, in)

	return gotPort, gotMsg, res.Err()
}

// The fold ten js_eval nodes were hand-writing across seven flows.
func TestAppendsTheToolResultToTheHistory(t *testing.T) {
	history := []Message{
		{Role: "user", Content: "which pods are unhealthy?"},
		{Role: "assistant", ToolUses: []MessageToolUse{{ID: "call_1", Name: "pod_list"}}},
	}

	port, msg, err := capture(t, Request{
		Messages:  history,
		ToolUseID: "call_1",
		Result:    map[string]any{"unhealthy": 1},
	}, Settings{})
	if err != nil {
		t.Fatalf("handle: %v", err)
	}
	if port != ResponsePort {
		t.Fatalf("emitted on %q, want %q", port, ResponsePort)
	}

	out, ok := msg.(Response)
	if !ok {
		t.Fatalf("emitted %T", msg)
	}
	if len(out.Messages) != 3 {
		t.Fatalf("history has %d messages, want the original 2 plus the result", len(out.Messages))
	}
	last := out.Messages[2]
	if last.Role != RoleTool {
		t.Errorf("role = %q, want %q", last.Role, RoleTool)
	}
	if last.ToolCallID != "call_1" {
		t.Errorf("toolCallId = %q — the model cannot match the result to its call", last.ToolCallID)
	}
	if last.Content != `{"unhealthy":1}` {
		t.Errorf("content = %q, want the encoded result", last.Content)
	}
}

// The caller's slice can be shared with another branch of the flow. Appending
// in place would grow a history nobody else touched.
func TestDoesNotMutateTheCallersHistory(t *testing.T) {
	history := make([]Message, 1, 8) // spare capacity: append would write in place
	history[0] = Message{Role: "user", Content: "hello"}

	if _, _, err := capture(t, Request{Messages: history, ToolUseID: "call_1", Result: "ok"}, Settings{}); err != nil {
		t.Fatalf("handle: %v", err)
	}
	if len(history) != 1 {
		t.Fatalf("the caller's history grew to %d", len(history))
	}
}

// A tool that already returns text must not arrive double-encoded: quoted and
// escaped output reads badly to a model, and every fold in the cluster took
// care to avoid it.
func TestStringResultIsNotDoubleEncoded(t *testing.T) {
	_, msg, err := capture(t, Request{
		Messages:  []Message{{Role: "user"}},
		ToolUseID: "call_1",
		Result:    "no pods are failing",
	}, Settings{})
	if err != nil {
		t.Fatalf("handle: %v", err)
	}
	got := msg.(Response).Messages[1].Content
	if got != "no pods are failing" {
		t.Fatalf("content = %q, want the string unchanged", got)
	}
}

// A tool that returned nothing still has to answer, or the model waits for a
// result that never arrives.
func TestNilResultStillAnswers(t *testing.T) {
	_, msg, err := capture(t, Request{Messages: []Message{{Role: "user"}}, ToolUseID: "call_1"}, Settings{})
	if err != nil {
		t.Fatalf("handle: %v", err)
	}
	if got := msg.(Response).Messages[1].Content; got == "" {
		t.Fatal("an empty content leaves the model waiting; say something instead")
	}
}

// The passthrough is how an apiKey reaches the next call. Losing it breaks the
// loop one hop later, where the cause is no longer visible.
func TestContextIsCarried(t *testing.T) {
	_, msg, err := capture(t, Request{
		Context:   map[string]any{"apiKey": "sk-test"},
		Messages:  []Message{{Role: "user"}},
		ToolUseID: "call_1",
		Result:    "ok",
	}, Settings{})
	if err != nil {
		t.Fatalf("handle: %v", err)
	}
	carried, _ := msg.(Response).Context.(map[string]any)
	if carried["apiKey"] != "sk-test" {
		t.Fatalf("context = %v, want it carried through", msg.(Response).Context)
	}
}

// Both required fields fail loudly. A tool result with no id cannot be matched
// to its call, and one with no history has no conversation to join.
func TestMissingFieldsAreRefused(t *testing.T) {
	for name, req := range map[string]Request{
		"no toolUseId": {Messages: []Message{{Role: "user"}}, Result: "ok"},
		"no messages":  {ToolUseID: "call_1", Result: "ok"},
	} {
		if _, _, err := capture(t, req, Settings{}); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// With the error port on, the same mistakes route instead of failing the run —
// the recovery-boundary contract every other component follows.
func TestErrorPortRoutesInsteadOfFailing(t *testing.T) {
	port, msg, err := capture(t, Request{Messages: []Message{{Role: "user"}}, Result: "ok"}, Settings{EnableErrorPort: true})
	if err != nil {
		t.Fatalf("with the error port on, the run must not fail: %v", err)
	}
	if port != ErrorPort {
		t.Fatalf("emitted on %q, want %q", port, ErrorPort)
	}
	if msg.(Error).Error == "" {
		t.Error("the error port carried no message")
	}
}

// The history has to round-trip unchanged, or a later turn loses the tool
// calls the model already made.
func TestExistingHistorySurvivesUntouched(t *testing.T) {
	history := []Message{
		{Role: "user", Content: "q"},
		{Role: "assistant", Content: "thinking", ToolUses: []MessageToolUse{{ID: "call_1", Name: "pod_list", Input: map[string]any{"ns": "prod"}}}},
	}
	before, _ := json.Marshal(history)

	_, msg, err := capture(t, Request{Messages: history, ToolUseID: "call_1", Result: "ok"}, Settings{})
	if err != nil {
		t.Fatalf("handle: %v", err)
	}
	after, _ := json.Marshal(msg.(Response).Messages[:2])
	if string(before) != string(after) {
		t.Fatalf("history changed:\n before %s\n after  %s", before, after)
	}
}
