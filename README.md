# Tiny Systems LLM Module

LLM components for Tiny Systems flows — completion and agentic routing.

## Components

| Name | Purpose |
|---|---|
| `llm_complete` | Single-turn completion via the Anthropic Messages API. Supports prompt caching on the system prompt. Emits `text`, `model`, `stopReason`, and detailed `usage` (input/output/cache_read/cache_creation tokens). |
| `llm_router` | Route a message to one of N output ports based on LLM judgement. Configure routes as `{name, description}` pairs; each becomes an `out_<name>` source port. Emits Context-only (same shape as deterministic router). Decision metadata (chosen route, confidence, reasoning, token usage) lands on the trace span as attributes. |
| `llm_tools` | ReAct / function-calling primitive. Declare tools as `{name, description, inputSchema}` triples; each becomes an `out_<name>` source port that fires when the model picks that tool. Emits the structured tool args + the updated message history so the caller can wire a tool handler back into another `llm_tools.request` call with an appended `tool_result`. If the model produces a final text response instead, the `response` port fires with the text and final history. Stateless — caller maintains the conversation across iterations. |

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

## Settings

| Field | Default | Notes |
|---|---|---|
| `model` | `claude-haiku-4-5` | Any Anthropic model id (Haiku/Sonnet/Opus). |
| `systemPrompt` | *(empty)* | Sent as system role on every call. |
| `cacheSystem` | `false` | Mark the system prompt as `ephemeral`. Lets identical subsequent calls hit Anthropic's prompt cache. |
| `maxTokens` | `1024` | Output token cap. |
| `temperature` | `0` | Set higher for sampling diversity. |
| `timeoutSeconds` | `60` | Per-request HTTP timeout. |

The API key flows in via the input message (`apiKey`), not via component settings — same context-passthrough pattern other Tiny Systems modules use for credentials.

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
