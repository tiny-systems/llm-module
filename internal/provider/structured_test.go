package provider

import (
	"encoding/json"
	"testing"
)

var testSchema = map[string]any{
	"type":       "object",
	"properties": map[string]any{"verdict": map[string]any{"type": "string"}},
	"required":   []any{"verdict"},
}

// The Anthropic path expresses structured output as a forced tool call —
// one synthetic tool whose input schema IS the output schema.
func TestAnthropicStructuredRequestShape(t *testing.T) {
	body := anthropicRequest{Model: "m", MaxTokens: 1}
	body.Tools = []anthropicToolDef{{Name: structuredOutputTool, InputSchema: testSchema}}
	body.ToolChoice = &anthropicToolChoice{Type: "tool", Name: structuredOutputTool}

	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	_ = json.Unmarshal(raw, &m)
	tc := m["tool_choice"].(map[string]any)
	if tc["type"] != "tool" || tc["name"] != structuredOutputTool {
		t.Fatalf("tool_choice not forced: %v", tc)
	}
	tools := m["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("want exactly one synthetic tool, got %d", len(tools))
	}
}

func TestOpenAIStructuredRequestShape(t *testing.T) {
	body := openaiRequest{Model: "m"}
	body.ResponseFormat = &openaiResponseFormat{
		Type:       "json_schema",
		JSONSchema: &openaiJSONSchema{Name: "structured_output", Schema: testSchema, Strict: true},
	}
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	_ = json.Unmarshal(raw, &m)
	rf := m["response_format"].(map[string]any)
	if rf["type"] != "json_schema" {
		t.Fatalf("response_format type: %v", rf["type"])
	}
	js := rf["json_schema"].(map[string]any)
	if js["strict"] != true || js["name"] != "structured_output" {
		t.Fatalf("json_schema not strict/named: %v", js)
	}
}
