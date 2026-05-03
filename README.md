# Tiny Systems LLM Module

LLM completion components for Tiny Systems flows.

## Components

| Name | Purpose |
|---|---|
| `llm_complete` | Single-turn completion via the Anthropic Messages API. Supports prompt caching on the system prompt. Emits `text`, `model`, `stopReason`, and detailed `usage` (input/output/cache_read/cache_creation tokens). |

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
