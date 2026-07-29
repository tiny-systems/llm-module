package mcpcall

import (
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestTextOf(t *testing.T) {
	t.Run("joins text blocks", func(t *testing.T) {
		got := textOf([]mcp.Content{
			&mcp.TextContent{Text: "first"},
			&mcp.TextContent{Text: "second"},
		})
		if got != "first\nsecond" {
			t.Errorf("got %q", got)
		}
	})

	t.Run("empty content is empty string", func(t *testing.T) {
		if got := textOf(nil); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})

	t.Run("a non-text block is rendered, not dropped", func(t *testing.T) {
		// Dropping it silently would make a flow look like it received nothing.
		got := textOf([]mcp.Content{
			&mcp.TextContent{Text: "caption"},
			&mcp.ImageContent{Data: []byte("x"), MIMEType: "image/png"},
		})
		if !strings.Contains(got, "caption") {
			t.Errorf("text block lost: %q", got)
		}
		if !strings.Contains(got, "image/png") {
			t.Errorf("non-text block dropped silently: %q", got)
		}
	})
}
