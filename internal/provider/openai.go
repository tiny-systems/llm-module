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

func envelopeKind(e openaiErrorBody) string {
	if e.Code != "" {
		return e.Code
	}
	if e.Type != "" {
		return e.Type
	}
	return "openai_error"
}
