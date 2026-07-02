package stepcache

import (
	"context"
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

type payload struct {
	Text string `json:"text"`
}

func TestStepCache_RoundTripWithinRun(t *testing.T) {
	st := newMem()
	ctx := module.WithRun(context.Background(), module.NewRun("run-1", "run-1.step7"))

	if _, ok := Get[payload](ctx, st); ok {
		t.Fatal("cache must miss before Put")
	}
	Put(ctx, st, payload{Text: "cat"})
	got, ok := Get[payload](ctx, st)
	if !ok || got.Text != "cat" {
		t.Fatalf("round trip failed: %+v %v", got, ok)
	}
}

func TestStepCache_NoRunIsNoop(t *testing.T) {
	st := newMem()
	Put(context.Background(), st, payload{Text: "cat"})
	if len(st.m) != 0 {
		t.Fatalf("Put outside a run must store nothing: %v", st.m)
	}
	if _, ok := Get[payload](context.Background(), st); ok {
		t.Fatal("Get outside a run must miss")
	}
}

func TestStepCache_NilStateIsNoop(t *testing.T) {
	ctx := module.WithRun(context.Background(), module.NewRun("run-1", "run-1.step7"))
	Put[payload](ctx, nil, payload{Text: "cat"}) // must not panic
	if _, ok := Get[payload](ctx, nil); ok {
		t.Fatal("nil state must miss")
	}
}

// Different steps of the same run must not see each other's responses.
func TestStepCache_StepIsolation(t *testing.T) {
	st := newMem()
	ctxA := module.WithRun(context.Background(), module.NewRun("run-1", "run-1.stepA"))
	ctxB := module.WithRun(context.Background(), module.NewRun("run-1", "run-1.stepB"))
	Put(ctxA, st, payload{Text: "A"})
	if _, ok := Get[payload](ctxB, st); ok {
		t.Fatal("step B must not read step A's cache")
	}
}
