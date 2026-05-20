package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const openaiDefaultURL = "https://api.openai.com/v1/chat/completions"

type openaiProvider struct{}

type openaiMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type openaiRequest struct {
	Model       string          `json:"model"`
	Messages    []openaiMessage `json:"messages"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature float64         `json:"temperature,omitempty"`
}

type openaiResponseChoice struct {
	Message      openaiMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type openaiResponseUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type openaiResponse struct {
	Model   string                 `json:"model"`
	Choices []openaiResponseChoice `json:"choices"`
	Usage   openaiResponseUsage    `json:"usage"`
}

type openaiErrorBody struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

type openaiErrorEnvelope struct {
	Error openaiErrorBody `json:"error"`
}

func (p *openaiProvider) Complete(ctx context.Context, in CompletionRequest) (*CompletionResponse, error) {
	url := resolveOpenAIURL(in.BaseURL)

	apiMessages := make([]openaiMessage, 0, len(in.Messages)+1)
	if in.SystemPrompt != "" {
		apiMessages = append(apiMessages, openaiMessage{Role: "system", Content: in.SystemPrompt})
	}
	for _, m := range in.Messages {
		apiMessages = append(apiMessages, openaiMessage{Role: m.Role, Content: m.Content})
	}

	body := openaiRequest{
		Model:       in.Model,
		Messages:    apiMessages,
		MaxTokens:   in.MaxTokens,
		Temperature: in.Temperature,
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, &Error{Err: err}
	}

	timeout := in.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, &Error{Err: err}
	}
	httpReq.Header.Set("content-type", "application/json")
	httpReq.Header.Set("authorization", "Bearer "+in.APIKey)

	resp, err := (&http.Client{}).Do(httpReq)
	if err != nil {
		return nil, &Error{Err: err, Retryable: true}
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &Error{Err: err}
	}

	if resp.StatusCode >= 400 {
		retryable := isRetryableStatus(resp.StatusCode)
		var envelope openaiErrorEnvelope
		if jsonErr := json.Unmarshal(respBody, &envelope); jsonErr == nil && envelope.Error.Message != "" {
			return nil, &Error{
				Err:       fmt.Errorf("%s: %s", envelopeKind(envelope.Error), envelope.Error.Message),
				Retryable: retryable,
			}
		}
		return nil, &Error{
			Err:       fmt.Errorf("openai api: status %d: %s", resp.StatusCode, string(respBody)),
			Retryable: retryable,
		}
	}

	var parsed openaiResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, &Error{Err: fmt.Errorf("decode response: %w", err)}
	}

	if len(parsed.Choices) == 0 {
		return nil, &Error{Err: fmt.Errorf("openai api: empty choices")}
	}

	text, _ := parsed.Choices[0].Message.Content.(string)

	return &CompletionResponse{
		Text:       text,
		Model:      parsed.Model,
		StopReason: parsed.Choices[0].FinishReason,
		Usage: Usage{
			Input:  parsed.Usage.PromptTokens,
			Output: parsed.Usage.CompletionTokens,
		},
	}, nil
}

func resolveOpenAIURL(baseURL string) string {
	if baseURL == "" {
		return openaiDefaultURL
	}
	trimmed := strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(trimmed, "/chat/completions") {
		return trimmed
	}
	return trimmed + "/chat/completions"
}

// --- Tool calling --------------------------------------------------

type openaiFunctionSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

type openaiToolDefWire struct {
	Type     string             `json:"type"`
	Function openaiFunctionSpec `json:"function"`
}

type openaiToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openaiToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function openaiToolCallFunction `json:"function"`
}

type openaiToolMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content,omitempty"`
	ToolCalls  []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type openaiToolRequest struct {
	Model       string              `json:"model"`
	Messages    []openaiToolMessage `json:"messages"`
	Tools       []openaiToolDefWire `json:"tools,omitempty"`
	MaxTokens   int                 `json:"max_tokens,omitempty"`
	Temperature float64             `json:"temperature,omitempty"`
}

type openaiToolResponseMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	ToolCalls []openaiToolCall `json:"tool_calls"`
}

type openaiToolResponseChoice struct {
	Message      openaiToolResponseMessage `json:"message"`
	FinishReason string                    `json:"finish_reason"`
}

type openaiToolResponse struct {
	Model   string                     `json:"model"`
	Choices []openaiToolResponseChoice `json:"choices"`
	Usage   openaiResponseUsage        `json:"usage"`
}

func (p *openaiProvider) CompleteWithTools(ctx context.Context, in ToolCompletionRequest) (*ToolCompletionResponse, error) {
	url := resolveOpenAIURL(in.BaseURL)

	msgs := make([]openaiToolMessage, 0, len(in.Messages)+1)
	if in.SystemPrompt != "" {
		msgs = append(msgs, openaiToolMessage{Role: "system", Content: in.SystemPrompt})
	}
	for _, m := range in.Messages {
		converted, err := toolMessageToOpenAI(m)
		if err != nil {
			return nil, &Error{Err: err}
		}
		msgs = append(msgs, converted)
	}

	body := openaiToolRequest{
		Model:       in.Model,
		Messages:    msgs,
		Tools:       toolDefsToOpenAI(in.Tools),
		MaxTokens:   in.MaxTokens,
		Temperature: in.Temperature,
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, &Error{Err: err}
	}

	timeout := in.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, &Error{Err: err}
	}
	httpReq.Header.Set("content-type", "application/json")
	httpReq.Header.Set("authorization", "Bearer "+in.APIKey)

	resp, err := (&http.Client{}).Do(httpReq)
	if err != nil {
		return nil, &Error{Err: err, Retryable: true}
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &Error{Err: err}
	}

	if resp.StatusCode >= 400 {
		retryable := isRetryableStatus(resp.StatusCode)
		var envelope openaiErrorEnvelope
		if jsonErr := json.Unmarshal(respBody, &envelope); jsonErr == nil && envelope.Error.Message != "" {
			return nil, &Error{
				Err:       fmt.Errorf("%s: %s", envelopeKind(envelope.Error), envelope.Error.Message),
				Retryable: retryable,
			}
		}
		return nil, &Error{
			Err:       fmt.Errorf("openai api: status %d: %s", resp.StatusCode, string(respBody)),
			Retryable: retryable,
		}
	}

	var parsed openaiToolResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, &Error{Err: fmt.Errorf("decode response: %w", err)}
	}
	if len(parsed.Choices) == 0 {
		return nil, &Error{Err: fmt.Errorf("openai api: empty choices")}
	}

	choice := parsed.Choices[0].Message
	var uses []ToolUse
	for _, tc := range choice.ToolCalls {
		var args any
		if tc.Function.Arguments != "" {
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				// Fall back to the raw string so the caller can still
				// see what the model produced.
				args = tc.Function.Arguments
			}
		}
		uses = append(uses, ToolUse{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: args,
		})
	}

	return &ToolCompletionResponse{
		Text:       choice.Content,
		ToolUses:   uses,
		Model:      parsed.Model,
		StopReason: parsed.Choices[0].FinishReason,
		Usage: Usage{
			Input:  parsed.Usage.PromptTokens,
			Output: parsed.Usage.CompletionTokens,
		},
	}, nil
}

func toolDefsToOpenAI(tools []ToolDef) []openaiToolDefWire {
	out := make([]openaiToolDefWire, len(tools))
	for i, t := range tools {
		out[i] = openaiToolDefWire{
			Type: "function",
			Function: openaiFunctionSpec{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		}
	}
	return out
}

func toolMessageToOpenAI(m ToolMessage) (openaiToolMessage, error) {
	switch m.Role {
	case "user":
		return openaiToolMessage{Role: "user", Content: m.Text}, nil
	case "tool":
		return openaiToolMessage{
			Role:       "tool",
			ToolCallID: m.ToolCallID,
			Content:    m.Text,
		}, nil
	case "assistant":
		calls := make([]openaiToolCall, 0, len(m.ToolUses))
		for _, u := range m.ToolUses {
			argsBytes, err := json.Marshal(u.Input)
			if err != nil {
				return openaiToolMessage{}, fmt.Errorf("encode tool input for %s: %w", u.Name, err)
			}
			calls = append(calls, openaiToolCall{
				ID:   u.ID,
				Type: "function",
				Function: openaiToolCallFunction{
					Name:      u.Name,
					Arguments: string(argsBytes),
				},
			})
		}
		return openaiToolMessage{
			Role:      "assistant",
			Content:   m.Text,
			ToolCalls: calls,
		}, nil
	default:
		return openaiToolMessage{}, fmt.Errorf("unknown role %q", m.Role)
	}
}

func envelopeKind(e openaiErrorBody) string {
	if e.Code != "" {
		return e.Code
	}
	if e.Type != "" {
		return e.Type
	}
	return "openai_error"
}
