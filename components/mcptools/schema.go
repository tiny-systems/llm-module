package mcptools

import (
	"encoding/json"
	"fmt"
)

// schemaMap normalises a tool's input schema to a plain map.
//
// The MCP SDK documents this field as "the default JSON marshaling of the
// server's input schema (a map[string]any)" when read from the client side, but
// types it as `any` because a server may set it to anything that marshals to
// valid JSON schema. Round-trip through JSON so a *jsonschema.Schema, a
// json.RawMessage, and a map all arrive here as the same thing — the shape
// llm_tools' Tool.InputSchema wants.
func schemaMap(raw any) (map[string]any, error) {
	if raw == nil {
		// A tool taking no arguments still needs a schema llm_tools can send to
		// a provider; an empty object is the honest translation.
		return map[string]any{"type": "object"}, nil
	}
	if m, ok := raw.(map[string]any); ok {
		return m, nil
	}
	b, err := json.Marshal(raw)
	if err != nil {
		return nil, fmt.Errorf("marshal schema: %w", err)
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, fmt.Errorf("schema is not a JSON object: %w", err)
	}
	return out, nil
}
