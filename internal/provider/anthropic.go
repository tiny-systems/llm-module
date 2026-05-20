package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	anthropicDefaultURL = "https://api.anthropic.com/v1/messages"
	anthropicVersion    = "2023-06-01"
)

type anthropicProvider struct{}

type anthropicCacheControl struct {
	Type string `json:"type"`
}

type anthropicTextBlock struct {
	Type         string                 `json:"type"`
	Text         string                 `json:"text,omitempty"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type anthropicRequest struct {
	Model       string               `json:"model"`
	MaxTokens   int                  `json:"max_tokens"`
	Temperature float64              `json:"temperature,omitempty"`
	System      []anthropicTextBlock `json:"system,omitempty"`
	Messages    []anthropicMessage   `json:"messages"`
}

type anthropicResponseUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

type anthropicResponseContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicResponse struct {
	Model      string                     `json:"model"`
	Content    []anthropicResponseContent `json:"content"`
	StopReason string                     `json:"stop_reason"`
	Usage      anthropicResponseUsage     `json:"usage"`
}

type anthropicErrorBody struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type anthropicErrorEnvelope struct {
	Error anthropicErrorBody `json:"error"`
}

func (p *anthropicProvider) Complete(ctx context.Context, in CompletionRequest) (*CompletionResponse, error) {
	url := in.BaseURL
	if url == "" {
		url = anthropicDefaultURL
	}

	apiMessages := make([]anthropicMessage, len(in.Messages))
	for i, m := range in.Messages {
		apiMessages[i] = anthropicMessage{Role: m.Role, Content: m.Content}
	}

	body := anthropicRequest{
		Model:       in.Model,
		MaxTokens:   in.MaxTokens,
		Temperature: in.Temperature,
		Messages:    apiMessages,
	}
	if in.SystemPrompt != "" {
		block := anthropicTextBlock{Type: "text", Text: in.SystemPrompt}
		if in.CacheSystem {
			block.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
		}
		body.System = []anthropicTextBlock{block}
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
	httpReq.Header.Set("x-api-key", in.APIKey)
	httpReq.Header.Set("anthropic-version", anthropicVersion)

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
		var envelope anthropicErrorEnvelope
		if jsonErr := json.Unmarshal(respBody, &envelope); jsonErr == nil && envelope.Error.Message != "" {
			return nil, &Error{
				Err:       fmt.Errorf("%s: %s", envelope.Error.Type, envelope.Error.Message),
				Retryable: retryable,
			}
		}
		return nil, &Error{
			Err:       fmt.Errorf("anthropic api: status %d: %s", resp.StatusCode, string(respBody)),
			Retryable: retryable,
		}
	}

	var parsed anthropicResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, &Error{Err: fmt.Errorf("decode response: %w", err)}
	}

	var text string
	for _, block := range parsed.Content {
		if block.Type == "text" {
			text += block.Text
		}
	}

	return &CompletionResponse{
		Text:       text,
		Model:      parsed.Model,
		StopReason: parsed.StopReason,
		Usage: Usage{
			Input:         parsed.Usage.InputTokens,
			Output:        parsed.Usage.OutputTokens,
			CacheRead:     parsed.Usage.CacheReadInputTokens,
			CacheCreation: parsed.Usage.CacheCreationInputTokens,
		},
	}, nil
}

func isRetryableStatus(code int) bool {
	return code == http.StatusTooManyRequests || code == 529 || code >= 500
}

// --- Tool calling --------------------------------------------------

type anthropicToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

type anthropicToolContentBlock struct {
	Type      string `json:"type"`
	Text      string `json:"text,omitempty"`
	ID        string `json:"id,omitempty"`
	Name      string `json:"name,omitempty"`
	Input     any    `json:"input,omitempty"`
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   any    `json:"content,omitempty"`
}

type anthropicToolRequest struct {
	Model       string               `json:"model"`
	MaxTokens   int                  `json:"max_tokens"`
	Temperature float64              `json:"temperature,omitempty"`
	System      []anthropicTextBlock `json:"system,omitempty"`
	Messages    []anthropicMessage   `json:"messages"`
	Tools       []anthropicToolDef   `json:"tools,omitempty"`
}

type anthropicToolResponse struct {
	Model      string                      `json:"model"`
	Content    []anthropicToolContentBlock `json:"content"`
	StopReason string                      `json:"stop_reason"`
	Usage      anthropicResponseUsage      `json:"usage"`
}

func (p *anthropicProvider) CompleteWithTools(ctx context.Context, in ToolCompletionRequest) (*ToolCompletionResponse, error) {
	url := in.BaseURL
	if url == "" {
		url = anthropicDefaultURL
	}

	body := anthropicToolRequest{
		Model:       in.Model,
		MaxTokens:   in.MaxTokens,
		Temperature: in.Temperature,
		Messages:    toolMessagesToAnthropic(in.Messages),
		Tools:       toolDefsToAnthropic(in.Tools),
	}
	if in.SystemPrompt != "" {
		block := anthropicTextBlock{Type: "text", Text: in.SystemPrompt}
		if in.CacheSystem {
			block.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
		}
		body.System = []anthropicTextBlock{block}
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
	httpReq.Header.Set("x-api-key", in.APIKey)
	httpReq.Header.Set("anthropic-version", anthropicVersion)

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
		var envelope anthropicErrorEnvelope
		if jsonErr := json.Unmarshal(respBody, &envelope); jsonErr == nil && envelope.Error.Message != "" {
			return nil, &Error{
				Err:       fmt.Errorf("%s: %s", envelope.Error.Type, envelope.Error.Message),
				Retryable: retryable,
			}
		}
		return nil, &Error{
			Err:       fmt.Errorf("anthropic api: status %d: %s", resp.StatusCode, string(respBody)),
			Retryable: retryable,
		}
	}

	var parsed anthropicToolResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, &Error{Err: fmt.Errorf("decode response: %w", err)}
	}

	var text string
	var uses []ToolUse
	for _, block := range parsed.Content {
		switch block.Type {
		case "text":
			text += block.Text
		case "tool_use":
			uses = append(uses, ToolUse{
				ID:    block.ID,
				Name:  block.Name,
				Input: block.Input,
			})
		}
	}

	return &ToolCompletionResponse{
		Text:       text,
		ToolUses:   uses,
		Model:      parsed.Model,
		StopReason: parsed.StopReason,
		Usage: Usage{
			Input:         parsed.Usage.InputTokens,
			Output:        parsed.Usage.OutputTokens,
			CacheRead:     parsed.Usage.CacheReadInputTokens,
			CacheCreation: parsed.Usage.CacheCreationInputTokens,
		},
	}, nil
}

func toolDefsToAnthropic(tools []ToolDef) []anthropicToolDef {
	out := make([]anthropicToolDef, len(tools))
	for i, t := range tools {
		out[i] = anthropicToolDef{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		}
	}
	return out
}

// toolMessagesToAnthropic flattens normalized ToolMessage into the
// Anthropic wire shape. Consecutive role:tool messages are coalesced
// into a single user message with multiple tool_result blocks, since
// Anthropic groups parallel tool returns this way.
func toolMessagesToAnthropic(msgs []ToolMessage) []anthropicMessage {
	out := make([]anthropicMessage, 0, len(msgs))
	var pendingResults []anthropicToolContentBlock

	flush := func() {
		if len(pendingResults) == 0 {
			return
		}
		out = append(out, anthropicMessage{
			Role:    "user",
			Content: pendingResults,
		})
		pendingResults = nil
	}

	for _, m := range msgs {
		switch m.Role {
		case "tool":
			pendingResults = append(pendingResults, anthropicToolContentBlock{
				Type:      "tool_result",
				ToolUseID: m.ToolCallID,
				Content:   m.Text,
			})
		case "user":
			flush()
			out = append(out, anthropicMessage{Role: "user", Content: m.Text})
		case "assistant":
			flush()
			var blocks []anthropicToolContentBlock
			if m.Text != "" {
				blocks = append(blocks, anthropicToolContentBlock{Type: "text", Text: m.Text})
			}
			for _, u := range m.ToolUses {
				blocks = append(blocks, anthropicToolContentBlock{
					Type:  "tool_use",
					ID:    u.ID,
					Name:  u.Name,
					Input: u.Input,
				})
			}
			if len(blocks) == 1 && blocks[0].Type == "text" {
				out = append(out, anthropicMessage{Role: "assistant", Content: m.Text})
			} else {
				out = append(out, anthropicMessage{Role: "assistant", Content: blocks})
			}
		}
	}
	flush()
	return out
}
