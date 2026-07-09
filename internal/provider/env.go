package provider

import "os"

// EnvAPIKey returns a fallback API key from the module pod's environment,
// selected by provider. It lets an operator wire a key ONCE at install
// time — e.g. the hosted playground funds every llm node with its own
// funded key, so trial flows run with zero per-node key config, no widget
// to fill, no [[secret:...]] to author — matching the zero-config story
// the pgvector bundle gives the memory components.
//
// Returns "" when no env key is set, so callers still require an explicit
// Settings.APIKey / Request.APIKey exactly as before (non-breaking): the
// fallback only activates where an operator deliberately provided one.
func EnvAPIKey(providerName string) string {
	switch providerName {
	case "openai":
		return os.Getenv("OPENAI_API_KEY")
	default: // anthropic (and anything unspecified)
		return os.Getenv("ANTHROPIC_API_KEY")
	}
}
