// Package stepcache closes the kill-during-call double-spend window for
// provider calls inside durable runs.
//
// The step ledger (SDK v0.11.x) already guarantees a COMPLETED hop is never
// re-executed on redelivery — but a pod that dies after the provider call
// returned and before the step record was written re-executes the hop, and
// the provider call is billed twice. Providers expose no idempotency keys
// for completions, so the guard has to live on our side: cache the provider
// response in the run's execution-scoped state, keyed by the hop's StepKey,
// the moment the call returns. A re-executed hop finds the cached response
// and reuses it instead of re-calling.
//
// The residual window shrinks to the provider HTTP call itself (died
// mid-flight — genuinely unknowable whether it was billed). Replay-by-cache
// is the correct durable-execution semantic even if the component would have
// built a slightly different request on re-execution: the step already ran.
//
// Outside a durable run (no run identity on the context) both functions are
// no-ops, so callers can invoke them unconditionally.
package stepcache

import (
	"context"

	"github.com/goccy/go-json"
	"github.com/tiny-systems/module/module"
)

// keyPrefix namespaces cached provider responses inside the execution scope,
// away from the runtime's step/ ledger records.
const keyPrefix = "llm/"

// Get returns the cached provider response for the current durable hop.
func Get[T any](ctx context.Context, st module.State) (T, bool) {
	var zero T
	run, ok := module.RunFrom(ctx)
	if !ok || run.StepKey == "" || st == nil {
		return zero, false
	}
	exec := st.Scoped(module.ScopeExecution, run.RunID)
	raw, found, err := exec.Get(ctx, keyPrefix+run.StepKey)
	if err != nil || !found {
		return zero, false
	}
	var v T
	if err := json.Unmarshal(raw, &v); err != nil {
		return zero, false
	}
	return v, true
}

// Put stores the provider response for the current durable hop. Best-effort:
// a failed write only means a re-executed hop would re-call the provider —
// the pre-cache behavior.
func Put[T any](ctx context.Context, st module.State, v T) {
	run, ok := module.RunFrom(ctx)
	if !ok || run.StepKey == "" || st == nil {
		return
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return
	}
	exec := st.Scoped(module.ScopeExecution, run.RunID)
	_ = exec.Set(ctx, keyPrefix+run.StepKey, raw)
}
