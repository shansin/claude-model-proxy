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
git clone https://github.com/shansin/claude-model-proxy.git
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

Copy `.env.example` to `.env` and edit to configure the proxy:

```bash
cp .env.example .env
```

All configuration is done via environment variables (loaded from `.env`):

```env
# Ollama server URL
OLLAMA_BASE_URL=http://localhost:11434

# Address this proxy listens on
PROXY_HOST=localhost
PROXY_PORT=8082

# Map Claude model name fragments to Ollama models
OLLAMA_MODEL_MAP_OPUS=qwen3.5:27b
OLLAMA_MODEL_MAP_SONNET=qwen3.5:27b
OLLAMA_MODEL_MAP_HAIKU=qwen3.5:27b
OLLAMA_MODEL_MAP_DEFAULT=qwen3.5:27b

# Context window size per model class
OLLAMA_CONTEXT_SIZE_OPUS=32768
OLLAMA_CONTEXT_SIZE_SONNET=32768
OLLAMA_CONTEXT_SIZE_HAIKU=8192
OLLAMA_CONTEXT_SIZE_DEFAULT=32768

# Anthropic passthrough (needed if some models fall through to Anthropic)
# ANTHROPIC_BASE_URL=https://api.anthropic.com
# ANTHROPIC_API_KEY=sk-ant-...
```

### Environment variables reference

| Variable | Description |
|---|---|
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `PROXY_HOST` | Host the proxy binds to (default: `localhost`) |
| `PROXY_PORT` | Port the proxy listens on (default: `8082`) |
| `OLLAMA_MODEL_MAP_<KEY>` | Model mapping, e.g. `OLLAMA_MODEL_MAP_OPUS=qwen3.5:27b` |
| `OLLAMA_CONTEXT_SIZE_<KEY>` | Context size, e.g. `OLLAMA_CONTEXT_SIZE_SONNET=32768` |
| `ANTHROPIC_BASE_URL` | Anthropic API URL for passthrough (default: `https://api.anthropic.com`) |
| `ANTHROPIC_API_KEY` | API key for Anthropic passthrough |

## Model routing

When a request arrives for a Claude model (e.g. `claude-sonnet-4-6`), the proxy does a **case-insensitive substring match** against the model map keys. The first matching key wins; if nothing matches, `default` is used. If no model map entries are defined, all requests are forwarded to Anthropic.

You can also set a model map entry to the special value `anthropic` to explicitly forward requests for that model directly to the real Anthropic API instead of a local Ollama model.

| Claude model contains | Env var                      |
|-----------------------|------------------------------|
| `opus`                | `OLLAMA_MODEL_MAP_OPUS`      |
| `sonnet`              | `OLLAMA_MODEL_MAP_SONNET`    |
| `haiku`               | `OLLAMA_MODEL_MAP_HAIKU`     |
| *(no match)*          | `OLLAMA_MODEL_MAP_DEFAULT`   |

## API endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/messages` | Anthropic Messages API — supports streaming and non-streaming |
| `GET /v1/models` | Returns stub Claude model list so client checks pass |
| `GET /health` | Health check — returns current Ollama URL and model map |

## Benchmarking Models

The project includes a `benchmark_model.sh` script to test and compare the performance of your installed Ollama models. It tests both code generation (Python) and layout generation (HTML) tasks.

```bash
./benchmark_model.sh
```

This will run sequentially through all models returned by `ollama list`. The script will output its progress to the terminal and generate a CSV report in the `benchmark-reports/` directory containing detailed performance metrics such as generation tokens per second, prompt processing speed, and time to first response.
