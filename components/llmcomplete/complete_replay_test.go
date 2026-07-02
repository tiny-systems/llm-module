package llmcomplete

import (
	"context"
	"encoding/json"
	"sort"
	"strings"
	"testing"

	"github.com/tiny-systems/module/module"
)

type memState struct{ m map[string][]byte }

func newMem() *memState { return &memState{m: map[string][]byte{}} }
func (s *memState) Get(_ context.Context, k string) ([]byte, bool, error) {
	v, ok := s.m[k]
	return v, ok, nil
}
func (s *memState) Set(_ context.Context, k string, v []byte) error { s.m[k] = v; return nil }
func (s *memState) Delete(_ context.Context, k string) error        { delete(s.m, k); return nil }
func (s *memState) List(_ context.Context, prefix string) ([]string, error) {
	var out []string
	for k := range s.m {
		if strings.HasPrefix(k, prefix) {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out, nil
}
func (s *memState) Scoped(_, _ string) module.State { return s }

// A replayed durable hop must emit the cached response WITHOUT touching the
// provider — proven here by omitting the API key entirely: the pre-cache
// code path would fail with "api key missing" before any provider call.
func TestComplete_DurableReplayUsesCacheWithoutCredentials(t *testing.T) {
	st := newMem()
	cached, _ := json.Marshal(Response{Text: "cached-answer", Model: "claude-test", StopReason: "end_turn"})
	st.m["llm/run-1.step3"] = cached

	c := (&Component{}).Instance().(*Component)
	c.OnState(st)

	ctx := module.WithRun(context.Background(), module.NewRun("run-1", "run-1.step3"))
	var got Response
	res := c.Handle(ctx, func(_ context.Context, port string, data any) module.Result {
		if port == ResponsePort {
			got = data.(Response)
		}
		return module.Ok(nil)
	}, RequestPort, Request{UserMessage: "hi", Context: map[string]any{"k": "v"}})

	if res.Err() != nil {
		t.Fatalf("replay must not fail (no credentials needed): %v", res.Err())
	}
	if got.Text != "cached-answer" || got.Model != "claude-test" {
		t.Fatalf("cached response not reused: %+v", got)
	}
	// The fresh request context re-attaches on replay.
	ctxMap, _ := got.Context.(map[string]any)
	if ctxMap["k"] != "v" {
		t.Fatalf("request context must re-attach on replay: %#v", got.Context)
	}
}

// Outside a durable run the guard is inert: with no API key the component
// fails exactly as before the cache existed.
func TestComplete_NoRunKeepsClassicBehavior(t *testing.T) {
	c := (&Component{}).Instance().(*Component)
	c.OnState(newMem())

	res := c.Handle(context.Background(), func(_ context.Context, _ string, _ any) module.Result {
		return module.Ok(nil)
	}, RequestPort, Request{UserMessage: "hi"})

	if res.Err() == nil || !strings.Contains(res.Err().Error(), "api key missing") {
		t.Fatalf("expected classic api-key failure, got %v", res.Err())
	}
}
