package provider

import (
	"context"
	"testing"

	m "github.com/tiny-systems/module/module"
)

// Every paid call reports what it consumed, in the units the provider bills in.
// Answering "what did this run cost" used to mean parsing this module's
// response shape from outside — the coupling the SDK exists to prevent.
func TestMeterUsageReportsEveryBilledUnit(t *testing.T) {
	ctx, usage := m.WithUsage(context.Background())

	MeterUsage(ctx, Usage{Input: 1200, Output: 300, CacheRead: 5000, CacheCreation: 900})

	total := usage.Total()
	for unit, want := range map[string]float64{
		"llm_input_tokens":       1200,
		"llm_output_tokens":      300,
		"llm_cache_read_tokens":  5000,
		"llm_cache_write_tokens": 900,
		"llm_calls":              1,
	} {
		if total[unit] != want {
			t.Errorf("%s = %v, want %v", unit, total[unit], want)
		}
	}
}

// Cached reads and cache writes are priced differently. Folding them into
// input tokens would make a cache that is working look like one that is not.
func TestCacheUnitsStaySeparateFromInput(t *testing.T) {
	ctx, usage := m.WithUsage(context.Background())
	MeterUsage(ctx, Usage{Input: 10, CacheRead: 9000})

	if got := usage.Total()["llm_input_tokens"]; got != 10 {
		t.Fatalf("llm_input_tokens = %v — cache reads were folded in", got)
	}
}

// Several calls in one hop — a tool loop making two provider calls — add up
// rather than replacing each other.
func TestCallsInTheSameHopAccumulate(t *testing.T) {
	ctx, usage := m.WithUsage(context.Background())
	MeterUsage(ctx, Usage{Input: 100, Output: 20})
	MeterUsage(ctx, Usage{Input: 150, Output: 30})

	total := usage.Total()
	if total["llm_input_tokens"] != 250 || total["llm_calls"] != 2 {
		t.Fatalf("total = %v, want both calls counted", total)
	}
}

// A provider that reported no tokens must not produce a zero line that reads
// as "measured, and it was free".
func TestUnreportedCountersAreOmitted(t *testing.T) {
	ctx, usage := m.WithUsage(context.Background())
	MeterUsage(ctx, Usage{Input: 100})

	total := usage.Total()
	if _, ok := total["llm_cache_read_tokens"]; ok {
		t.Error("a counter the provider never reported was recorded as zero")
	}
	if total["llm_calls"] != 1 {
		t.Error("the call itself should still be counted")
	}
}

// Metering must never depend on the runtime being present: a direct call in a
// test, or a component invoked outside a hop, has no sink.
func TestMeteringWithoutARuntimeIsSafe(t *testing.T) {
	MeterUsage(context.Background(), Usage{Input: 1, Output: 1})
}
