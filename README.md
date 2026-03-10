# claude-model-proxy

A lightweight proxy that exposes the Anthropic Messages API (`/v1/messages`) and forwards requests to a local [Ollama](https://ollama.com) instance. This lets you use the Claude CLI (and other Anthropic SDK clients) with any Ollama-hosted model.

```
Claude CLI  →  proxy (:8082)  →  Ollama (:11434)  →  local LLM
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally

## Installation

```bash
git clone <repo-url>
cd claude-model-proxy
uv sync
```

## Usage

Start the proxy:

```bash
uv run python proxy.py
# or
./start_service.sh
# or (after uv sync installs the script entry point)
uv run proxy
```

Then point the Claude CLI at it:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8082
export ANTHROPIC_API_KEY=proxy
claude
```

## Configuration

Edit `config.yaml` to configure the proxy:

```yaml
ollama_base_url: http://localhost:11434

proxy_host: localhost
proxy_port: 8082

# Map Claude model name fragments to Ollama models (substring match, case-insensitive)
model_map:
  opus:    llama3.2:70b
  sonnet:  llama3.2
  haiku:   llama3.2:1b
  default: llama3.2

# Context window size per model class
context_size:
  opus:    32768
  sonnet:  16384
  haiku:   8192
  default: 8192
```

### Environment variable overrides

Individual config values can be overridden with environment variables (useful for containers or `.env` files):

| Variable | Description |
|---|---|
| `OLLAMA_BASE_URL` | Ollama server URL |
| `PROXY_HOST` | Host the proxy binds to |
| `PROXY_PORT` | Port the proxy listens on |
| `OLLAMA_MODEL_MAP_<KEY>` | Override a model mapping, e.g. `OLLAMA_MODEL_MAP_OPUS=llama3.2:70b` |
| `OLLAMA_CONTEXT_SIZE_<KEY>` | Override context size, e.g. `OLLAMA_CONTEXT_SIZE_SONNET=32768` |
| `CONFIG_PATH` | Path to a custom config file |

See `.env.example` for a template.

## Model routing

When a request arrives for a Claude model (e.g. `claude-sonnet-4-6`), the proxy does a **case-insensitive substring match** against the keys in `model_map`. The first matching key wins; if nothing matches, `default` is used.

| Claude model contains | `config.yaml` key | Env var override             |
|-----------------------|-------------------|------------------------------|
| `opus`                | `opus`            | `OLLAMA_MODEL_MAP_OPUS`      |
| `sonnet`              | `sonnet`          | `OLLAMA_MODEL_MAP_SONNET`    |
| `haiku`               | `haiku`           | `OLLAMA_MODEL_MAP_HAIKU`     |
| *(no match)*          | `default`         | `OLLAMA_MODEL_MAP_DEFAULT`   |

## API endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/messages` | Anthropic Messages API — supports streaming and non-streaming |
| `GET /v1/models` | Returns stub Claude model list so client checks pass |
| `GET /health` | Health check — returns current Ollama URL and model map |
