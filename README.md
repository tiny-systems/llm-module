# Tiny Systems LLM Module

LLM components for Tiny Systems flows — completion and agentic routing.

## Components

| Name | Purpose |
|---|---|
| `llm_complete` | Single-turn completion via Anthropic Messages API (default) or OpenAI Chat Completions (and any OpenAI-compatible endpoint — Ollama, vLLM, OpenRouter, Azure OpenAI). Supports prompt caching on Anthropic. Emits `text`, `model`, `stopReason`, and detailed `usage`. |
| `llm_router` | Route a message to one of N output ports based on LLM judgement. Configure routes as `{name, description}` pairs; each becomes an `out_<name>` source port. Emits Context-only (same shape as deterministic router). Decision metadata (chosen route, confidence, reasoning, token usage) lands on the trace span as attributes. |
| `llm_tools` | ReAct / function-calling primitive — multi-provider. Declare tools as `{name, description, inputSchema}` triples; each becomes an `out_<name>` source port that fires when the model picks that tool. Anthropic Messages tool_use and OpenAI Chat Completions function-calling share a single normalized `Message` shape: `{role, content, toolUses?, toolCallId?}`. Caller loop: take the response's `messages`, append `{role: "tool", toolCallId: "<id>", content: "<output>"}` for each tool result, re-call. Stateless. |
| `llm_chat` | Stateless multi-turn conversation, Anthropic or OpenAI-compatible. Caller supplies the full `messages` history per call; component emits the updated history (with the assistant turn appended) plus the response text on the `response` port. Persist with `document_store` (or `kv`/`postgres_exec`) around the call: load → llm_chat → save. Use for pure conversation; use `llm_tools` when the agent needs to call tools. |
| `mcp_tools` | List the tools a remote [MCP](https://modelcontextprotocol.io) server offers, emitted in `llm_tools`' declaration shape (`{name, description, inputSchema}`) so the entries paste straight into its `Tools` setting. |
| `mcp_call` | Invoke one tool on a remote MCP server. Emits `{text, structured?, isError}` with the request context passed through, so the result folds back into an `llm_tools` loop. |

## MCP: using a remote server's tools

`mcp_tools` and `mcp_call` let an agent use any MCP server's tools — GitHub,
Sentry, your own — without a component being written for each one. Streamable
HTTP via the official Go SDK, so servers on current and older protocol
revisions both work.

**Discovery is a build-time step, by necessity.** `llm_tools` derives an
`out_<tool>` port per tool declared in its **settings**, and settings are
static, so a runtime tool list cannot create ports:

1. Call `mcp_tools` once; read `result.tools`.
2. Paste those entries into `llm_tools`' `Tools` setting — the shapes match, no
   translation needed.
3. Wire `llm_tools` `out_<tool>` → `mcp_call.request`, mapping the tool's name
   to `tool` and `{{$.input}}` to `arguments`.
4. Fold the result back: `mcp_call.result` → `js_eval` appending
   `{role: "tool", toolCallId: <the toolUseId carried in context>, content: <result.text>}`
   to `messages` → `llm_tools.request`. Loop until `llm_tools`' `response` fires.

Carry loop state (`toolUseId`, `messages`, `apiKey`) through `mcp_call`'s
`context`, which passes through untouched.

**Set `disableParallelToolUse` on `llm_tools`.** It routes only the *first* tool
call of a turn to a port; if the model calls several and the loop returns one
result, the next provider call fails with `tool_use ids without tool_result`.

**Ports.** `mcp_tools`: `request` ← `{context?, token?}`, `result` →
`{context, server, tools[], count}`. `mcp_call`: `request` ←
`{context?, tool, arguments?, token?}`, `result` →
`{context, tool, text, structured?, isError}`. Both take `serverURL`,
`headers[]`, `timeoutSeconds`, `enableErrorPort` in settings, and both emit
`{context, error, retryable}` on `error` when it is enabled.

`text` is every text block joined by newlines — what a `{role: tool}` message
wants. Leave `token` empty in settings and map it per-request from the trigger
widget, so the credential is not stored in the flow.

A tool that runs and reports its own failure arrives on `result` with
`isError: true`, not on the `error` port; only transport and protocol failures go
there, because only those are worth retrying. Connection and call failures are
marked retryable and can be wired into `retry`.

## `llm_router`

Use when boolean routing conditions would be too many or too fuzzy to enumerate. Common cases: ticket triage, intent classification, content moderation, support escalation.

| Setting | Default | Notes |
|---|---|---|
| `routes` | required | `[{name, description}]`. Description tells the LLM when to pick this route. |
| `model` | `claude-haiku-4-5` | Haiku is cheap and fast for classification (~$0.0001/call). |
| `systemPrompt` | *(empty)* | Optional task framing ("You are triaging support tickets"). |
| `confidenceThreshold` | `0` | If `enableDefaultPort=true`, routes below this go to `default`. |
| `enableDefaultPort` | `false` | Exposes a `default` source port for low-confidence routes. |
| `enableErrorPort` | `false` | Routes LLM/API errors to an `error` source port instead of failing. |
| `timeoutSeconds` | `30` | Per-request HTTP timeout. |

API key is supplied per-message via `Request.apiKey` (same pattern as `llm_complete`).

Decision metadata lands on the trace span via these attributes:
- `llm_router.chosen` — picked route name
- `llm_router.confidence` — 0-1
- `llm_router.reasoning` — one sentence
- `llm_router.input_tokens` / `llm_router.output_tokens`

## Settings (`llm_complete`, `llm_chat`)

| Field | Default | Notes |
|---|---|---|
| `provider` | `anthropic` | `anthropic` or `openai`. Determines wire format, auth header, and which provider defaults apply. |
| `baseURL` | *(empty)* | Optional override. For `openai`-compatible servers pass the v1 base (e.g. `http://ollama:11434/v1` or `https://openrouter.ai/api/v1`). Leave blank for the provider default. |
| `model` | `claude-haiku-4-5` | Provider-specific model id. Anthropic: `claude-haiku-4-5`, `claude-sonnet-5`, `claude-opus-5`. OpenAI: `gpt-4o-mini`, `gpt-4o`. Ollama: whatever you have pulled (`llama3.1`, `qwen2.5`, …). |
| `systemPrompt` | *(empty)* | Sent as system role on every call. |
| `cacheSystem` | `false` | Anthropic only. Mark the system prompt as `ephemeral` so identical subsequent calls hit the prompt cache. Ignored on `openai`. |
| `maxTokens` | `1024` | Output token cap. |
| `outputSchema` | _empty_ | Optional JSON Schema; when set the reply is forced to conform (Anthropic: forced tool call, OpenAI: `response_format` json_schema strict) and the parsed object is emitted on `response.structured`. |
| `temperature` | `0` | Sent explicitly (0 = deterministic-ish). Set higher for sampling diversity. |
| `timeoutSeconds` | `60` | Per-request HTTP timeout. |

The API key flows in via the input message (`apiKey`), not via component settings — same context-passthrough pattern other Tiny Systems modules use for credentials. The same field carries Anthropic `x-api-key` or OpenAI `Bearer` tokens depending on `provider`.

### Switching providers

The output shape (`text`, `model`, `stopReason`, `usage`) is identical across providers, so downstream edges don't have to change when you flip `provider`. Differences:

- **Auth**: Anthropic sends `x-api-key` + `anthropic-version`; OpenAI sends `Authorization: Bearer …`.
- **Prompt caching**: Anthropic-only via `cacheSystem`; OpenAI ignores the field.
- **Usage**: `cacheRead` / `cacheCreation` are zero outside Anthropic.
- **Tool use**: `llm_tools` works on both providers since v0.8.0 (Anthropic tool_use and OpenAI function calling).

### Self-hosted via Ollama

```
provider: openai
baseURL:  http://ollama.ollama.svc.cluster.local:11434/v1
model:    llama3.1
apiKey:   any-non-empty-string
```

Ollama exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so the `openai` provider works against it directly. The `apiKey` is still required by the request shape but Ollama doesn't validate it — pass `ollama` or similar.

## Retryable errors

The error port emits `retryable=true` on:
- `429` (rate limit)
- `529` (Anthropic overloaded)
- `5xx` (server errors)
- network-level failures

Wire the error port to a delay→retry loop or a router that distinguishes retryable vs permanent failures.

## Pattern: scoring loop

```
prefilter:matched → llm_complete (system: "You score posts 0-100…", cacheSystem: true)
                       ├── response → json_decode → router (score > threshold) → alert
                       └── error    → router (retryable) → delay → loop back to llm_complete
```

The system prompt + product descriptions stay cached across all calls in the same minute, so per-call cost drops significantly.

## Run Locally

```shell
go run cmd/main.go run \
  --name=tiny-systems/llm-module-v0 \
  --namespace=tinysystems \
  --version=0.1.0
```

## License

MIT for this module's source. Depends on [Tiny Systems Module SDK](https://github.com/tiny-systems/module) (BSL 1.1).
