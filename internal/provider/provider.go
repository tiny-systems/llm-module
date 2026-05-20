// Package provider abstracts the LLM HTTP backend so llm_complete and
// llm_chat can target Anthropic, OpenAI-compatible endpoints (OpenAI,
// Azure OpenAI, Ollama, vLLM, OpenRouter, Together, etc.) without the
// component code branching on shape. Each provider translates a
// normalized CompletionRequest into its native wire format and
// translates the response back into a normalized CompletionResponse.
//
// The normalized shape models the union of features. Fields that are
// provider-specific (CacheSystem for Anthropic prompt caching, the
// CacheRead/CacheCreation usage counters) are ignored by providers
// that don't implement them.
package provider

import (
	"context"
	"fmt"
	"time"
)

type Message struct {
	Role    string
	Content any
}

type CompletionRequest struct {
	APIKey       string
	BaseURL      string
	Model        string
	SystemPrompt string
	CacheSystem  bool
	Messages     []Message
	MaxTokens    int
	Temperature  float64
	Timeout      time.Duration
}

type Usage struct {
	Input         int
	Output        int
	CacheRead     int
	CacheCreation int
}

type CompletionResponse struct {
	Text       string
	Model      string
	StopReason string
	Usage      Usage
}

// Error wraps a provider call failure with the retry intent the
// runner should respect. Retryable=true for 429, 529, 5xx, and
// network-level failures. Non-retryable for 4xx auth/validation and
// for decode errors.
type Error struct {
	Err       error
	Retryable bool
}

func (e *Error) Error() string { return e.Err.Error() }
func (e *Error) Unwrap() error { return e.Err }

type Provider interface {
	Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)
	CompleteWithTools(ctx context.Context, req ToolCompletionRequest) (*ToolCompletionResponse, error)
}

// ToolDef declares one function the model may call.
type ToolDef struct {
	Name        string
	Description string
	InputSchema map[string]any
}

// ToolUse is a single tool invocation by the model. The pair (ID, Name)
// is preserved across the round trip so the caller can match a
// ToolResult message back to the call it answers.
type ToolUse struct {
	ID    string
	Name  string
	Input any
}

// ToolMessage is the provider-agnostic message shape for tool flows.
// Three roles:
//
//   - "user":      Text holds the user prompt.
//   - "assistant": Text holds the model's reply (may be empty when it
//     only called tools); ToolUses holds any tool invocations.
//   - "tool":      ToolCallID names the ToolUse this is a result for;
//     Text holds the tool output (stringified by the caller).
type ToolMessage struct {
	Role       string
	Text       string
	ToolUses   []ToolUse
	ToolCallID string
}

type ToolCompletionRequest struct {
	APIKey       string
	BaseURL      string
	Model        string
	SystemPrompt string
	CacheSystem  bool
	Messages     []ToolMessage
	Tools        []ToolDef
	MaxTokens    int
	Temperature  float64
	Timeout      time.Duration
}

type ToolCompletionResponse struct {
	Text       string
	ToolUses   []ToolUse
	Model      string
	StopReason string
	Usage      Usage
}

const (
	Anthropic = "anthropic"
	OpenAI    = "openai"
)

// New constructs a Provider by name. Empty string defaults to
// Anthropic for backwards compatibility with pre-v0.7.0 flows.
func New(name string) (Provider, error) {
	switch name {
	case "", Anthropic:
		return &anthropicProvider{}, nil
	case OpenAI:
		return &openaiProvider{}, nil
	default:
		return nil, fmt.Errorf("unknown provider %q (supported: anthropic, openai)", name)
	}
}
