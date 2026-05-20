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
