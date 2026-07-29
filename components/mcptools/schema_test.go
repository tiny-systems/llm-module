package mcptools

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestSchemaMap(t *testing.T) {
	t.Run("nil becomes an empty object schema", func(t *testing.T) {
		// A no-argument tool still needs a schema llm_tools can send onward.
		got, err := schemaMap(nil)
		if err != nil {
			t.Fatal(err)
		}
		if got["type"] != "object" {
			t.Errorf("got %v, want type:object", got)
		}
	})

	t.Run("a map passes through unchanged", func(t *testing.T) {
		in := map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}
		got, err := schemaMap(in)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, in) {
			t.Errorf("got %v, want %v", got, in)
		}
	})

	t.Run("raw JSON is decoded", func(t *testing.T) {
		got, err := schemaMap(json.RawMessage(`{"type":"object","required":["path"]}`))
		if err != nil {
			t.Fatal(err)
		}
		if got["type"] != "object" {
			t.Errorf("type not preserved: %v", got)
		}
		if _, ok := got["required"]; !ok {
			t.Errorf("required not preserved: %v", got)
		}
	})

	t.Run("a non-object schema is an error, not a silent empty", func(t *testing.T) {
		if _, err := schemaMap([]int{1, 2}); err == nil {
			t.Error("expected an error for a non-object schema")
		}
	})
}
