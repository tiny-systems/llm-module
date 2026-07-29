// Package session opens a client connection to a remote MCP server.
//
// Both components in this module are stateless: they connect, make one call, and
// close. A node instance is a goroutine that may be scheduled on any replica and
// re-executed on redelivery, so holding a session across invocations would tie a
// flow to one pod and leak connections when that pod dies. The cost is the
// initialize handshake per call; correctness first, pooling later if a hot loop
// makes it worth the complexity.
package mcpclient

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/tiny-systems/module/module"
)

// clientName identifies this implementation to the remote server. Servers log
// it and some gate behaviour on it, so it names the platform rather than the
// component.
const clientName = "tiny-systems"

// Config is everything needed to reach a server. Kept separate from either
// component's settings so both build it the same way.
type Config struct {
	Endpoint       string
	BearerToken    string
	Headers        map[string]string
	TimeoutSeconds int
}

// bearerTransport attaches static headers to every request. The MCP SDK takes
// an *http.Client rather than per-request options, so auth rides here.
type bearerTransport struct {
	base    http.RoundTripper
	token   string
	headers map[string]string
}

func (t *bearerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Clone: RoundTrippers must not mutate the caller's request.
	r := req.Clone(req.Context())
	if t.token != "" {
		r.Header.Set("Authorization", "Bearer "+t.token)
	}
	for k, v := range t.headers {
		r.Header.Set(k, v)
	}
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	return base.RoundTrip(r)
}

// Open connects and completes the MCP initialize handshake. The caller must
// Close the returned session.
//
// Connection failures are marked retryable: an unreachable or overloaded server
// is transient, so the retry component can supervise the call. A rejected
// handshake is not — that is a bad endpoint or bad credentials, and retrying
// only repeats it.
func Open(ctx context.Context, cfg Config) (*mcp.ClientSession, error) {
	endpoint := strings.TrimSpace(cfg.Endpoint)
	if endpoint == "" {
		return nil, fmt.Errorf("server URL is required")
	}
	if !strings.HasPrefix(endpoint, "http://") && !strings.HasPrefix(endpoint, "https://") {
		return nil, fmt.Errorf("server URL must be http:// or https://, got %q", endpoint)
	}

	timeout := time.Duration(cfg.TimeoutSeconds) * time.Second
	if cfg.TimeoutSeconds <= 0 {
		timeout = 30 * time.Second
	}

	httpClient := &http.Client{
		Timeout: timeout,
		Transport: &bearerTransport{
			token:   strings.TrimSpace(cfg.BearerToken),
			headers: cfg.Headers,
		},
	}

	client := mcp.NewClient(&mcp.Implementation{
		Name:    clientName,
		Version: "v1",
	}, nil)

	transport := &mcp.StreamableClientTransport{
		Endpoint:   endpoint,
		HTTPClient: httpClient,
		// Nothing here consumes server-initiated messages — a stateless
		// connect-call-close never outlives the call, so a persistent GET
		// stream would be opened and abandoned on every invocation.
		DisableStandaloneSSE: true,
	}

	sess, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, module.Retryable(fmt.Errorf("connect to MCP server %s: %w", endpoint, err))
	}
	return sess, nil
}

// HeaderMap converts the flow-facing header list into the map Open wants.
func HeaderMap(headers []Header) map[string]string {
	if len(headers) == 0 {
		return nil
	}
	out := make(map[string]string, len(headers))
	for _, h := range headers {
		if h.Key == "" {
			continue
		}
		out[h.Key] = h.Value
	}
	return out
}

// Header is one extra HTTP header to send with every request. Some servers
// authenticate with a custom header rather than a bearer token.
type Header struct {
	Key   string `json:"key" title:"Key"`
	Value string `json:"value" title:"Value"`
}
