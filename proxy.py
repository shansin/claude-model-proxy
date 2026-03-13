"""
Claude CLI -> Ollama proxy (with optional Anthropic passthrough)

Exposes the Anthropic Messages API (/v1/messages) and forwards requests
to a configurable Ollama endpoint, translating formats in both directions.

If a model_map entry is set to the special value "anthropic", the request
is forwarded directly to the real Anthropic API instead of Ollama.

Configuration is loaded from environment variables (via .env file):

  OLLAMA_BASE_URL              Ollama server URL (default: http://localhost:11434)
  OLLAMA_MODEL_MAP_<KEY>       Model mapping, e.g. OLLAMA_MODEL_MAP_OPUS=llama3.2:70b
  OLLAMA_CONTEXT_SIZE_<KEY>    Context size, e.g. OLLAMA_CONTEXT_SIZE_SONNET=32768
  PROXY_HOST                   Host to listen on (default: localhost)
  PROXY_PORT                   Port to listen on (default: 8082)
  ANTHROPIC_API_KEY            API key for Anthropic passthrough
  ANTHROPIC_BASE_URL           Anthropic API URL (default: https://api.anthropic.com)

Run directly:
  uv run python proxy.py
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

load_dotenv()

class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        msg = record.getMessage()
        if record.levelno >= logging.WARNING:
            return f"{ts}  {record.levelname}  {msg}"
        return f"{ts}  {msg}"

_handler = logging.StreamHandler()
_handler.setFormatter(_Formatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler])
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config() -> dict[str, Any]:
    """Build configuration entirely from environment variables."""
    cfg: dict[str, Any] = {
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "anthropic_base_url": os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "proxy_host": os.getenv("PROXY_HOST", "localhost"),
        "proxy_port": int(os.getenv("PROXY_PORT", "8082")),
        "model_map": {},
        "context_size": {},
    }

    for key, val in os.environ.items():
        # OLLAMA_MODEL_MAP_OPUS=llama3.2:70b  ->  model_map["opus"] = "llama3.2:70b"
        if key.startswith("OLLAMA_MODEL_MAP_"):
            map_key = key[len("OLLAMA_MODEL_MAP_"):].lower()
            cfg["model_map"][map_key] = val
        # OLLAMA_CONTEXT_SIZE_SONNET=32768  ->  context_size["sonnet"] = 32768
        elif key.startswith("OLLAMA_CONTEXT_SIZE_"):
            ctx_key = key[len("OLLAMA_CONTEXT_SIZE_"):].lower()
            cfg["context_size"][ctx_key] = int(val)

    log.info("config    loaded from environment")
    return cfg


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_ANTHROPIC_PASSTHROUGH = "anthropic"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.config = cfg

    has_models = bool(cfg["model_map"])

    if has_models:
        log.info("ollama    %s", cfg["ollama_base_url"])

    key_hint = "set" if cfg.get("anthropic_api_key") else "from request"
    if has_models:
        log.info("anthropic %s  (api_key: %s)  (fallback for unmapped models)", cfg["anthropic_base_url"], key_hint)
    else:
        log.info("anthropic %s  (api_key: %s)  (all models)", cfg["anthropic_base_url"], key_hint)

    for key in cfg["model_map"]:
        label = key if key != "default" else "(default)"
        target = cfg["model_map"][key]
        if target == _ANTHROPIC_PASSTHROUGH:
            log.info("model     %-10s → %-22s (anthropic passthrough)", label, target)
        else:
            ctx = cfg["context_size"].get(key, cfg["context_size"].get("default", "—"))
            log.info("model     %-10s → %-22s ctx=%s", label, target, ctx)

    if not has_models:
        log.info("model     (all)      → anthropic  (no model_map entries)")

    proxy_url = f"http://{cfg['proxy_host']}:{cfg['proxy_port']}"
    log.info("ready     export ANTHROPIC_BASE_URL=%s ANTHROPIC_API_KEY=proxy ; claude", proxy_url)
    yield


app = FastAPI(title="Claude-to-Ollama Proxy", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_key(cfg_dict: dict, claude_model: str) -> str | None:
    """Return the first key whose substring matches claude_model, or None."""
    lower = claude_model.lower()
    for key in cfg_dict:
        if key != "default" and key in lower:
            return key
    return None


def resolve_ollama_model(cfg: dict, claude_model: str) -> str | None:
    """Return the Ollama model name for a Claude model, or None if unmapped."""
    key = _match_key(cfg["model_map"], claude_model)
    if key:
        return cfg["model_map"][key]
    return cfg["model_map"].get("default")


def resolve_context_size(cfg: dict, claude_model: str) -> int | None:
    key = _match_key(cfg["context_size"], claude_model)
    return cfg["context_size"].get(key or "default")


def estimate_tokens(messages: list[dict]) -> int:
    """Rough estimate: ~4 chars per token."""
    total = sum(len(str(m.get("content", ""))) for m in messages)
    return total // 4


def _msg_tokens(msg: dict) -> int:
    """Estimate tokens for a single message."""
    return len(str(msg.get("content", ""))) // 4


def trim_messages(messages: list[dict], ctx_size: int, msg_id: str) -> list[dict]:
    """Drop oldest non-system messages until estimated tokens fit within ctx_size.

    Reserves 25% of context for the response.  Keeps the system prompt (first
    message if role=system) and the most recent messages.
    """
    budget = int(ctx_size * 0.75)  # leave room for output
    est = estimate_tokens(messages)
    if est <= budget:
        return messages

    # Separate system prompt from conversation
    system_msgs: list[dict] = []
    conv_msgs: list[dict] = list(messages)
    if conv_msgs and conv_msgs[0].get("role") == "system":
        system_msgs = [conv_msgs.pop(0)]

    system_cost = estimate_tokens(system_msgs)
    remaining_budget = budget - system_cost

    # Keep messages from the end until budget is exhausted
    kept: list[dict] = []
    running = 0
    for msg in reversed(conv_msgs):
        cost = _msg_tokens(msg)
        if running + cost > remaining_budget:
            break
        kept.append(msg)
        running += cost
    kept.reverse()

    dropped = len(conv_msgs) - len(kept)
    if dropped > 0:
        log.warning(
            "\033[33m⚠ context trim  [%s]  dropped %d oldest messages "
            "(~%d→~%d tokens, budget=%d)\033[0m",
            msg_id[:8], dropped, est, system_cost + running, budget,
        )

    return system_msgs + kept


def _log_tool_use(name: str, inp: dict) -> None:
    """Log a tool_use block in a human-readable format."""
    def _summarize(value: Any, max_len: int = 80) -> str:
        s = str(value)
        if len(s) > max_len:
            return s[:max_len] + "…"
        return s

    if not inp:
        log.info("  \033[36m⚙ %s\033[0m()", name)
        return

    parts: list[str] = []
    for k, v in inp.items():
        if isinstance(v, str) and len(v) > 120:
            # Show length for long strings (file contents, code, etc.)
            lines = v.count("\n") + 1
            parts.append(f"{k}: ({len(v)} chars, {lines} lines)")
        elif isinstance(v, (dict, list)):
            summary = json.dumps(v, separators=(",", ":"))
            if len(summary) > 100:
                parts.append(f"{k}: {summary[:100]}…")
            else:
                parts.append(f"{k}: {summary}")
        else:
            parts.append(f"{k}: {_summarize(v)}")

    log.info("  \033[36m⚙ %s\033[0m  %s", name, "  ".join(parts))


def build_ollama_messages(body: dict) -> list[dict]:
    messages = []

    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
            if text:
                messages.append({"role": "system", "content": text})

    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            tool_calls = []
            tool_results = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    _log_tool_use(block.get("name", ""), block.get("input", {}))
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": block.get("input", {}),
                        },
                    })
                elif block.get("type") == "tool_result":
                    raw = block.get("content", "")
                    if isinstance(raw, str):
                        result_text = raw
                    else:
                        result_text = " ".join(
                            inner.get("text", "")
                            for inner in raw
                            if isinstance(inner, dict) and inner.get("type") == "text"
                        )
                    tool_results.append((block.get("tool_use_id", ""), result_text))

            if tool_results:
                # Emit any preceding text as a separate user message
                if text_parts:
                    messages.append({"role": role, "content": "\n".join(text_parts)})
                # Send each tool result as an Ollama "tool" role message
                for tool_call_id, result_text in tool_results:
                    messages.append({"role": "tool", "content": result_text, "tool_call_id": tool_call_id})
            elif tool_calls:
                # Assistant message with tool calls
                ollama_msg: dict[str, Any] = {"role": role, "content": "\n".join(text_parts), "tool_calls": tool_calls}
                messages.append(ollama_msg)
            else:
                messages.append({"role": role, "content": "\n".join(text_parts)})

    return messages


def _detect_stop_sequence(text: str, stop_sequences: list[str] | None) -> str | None:
    """Return the stop sequence that the text ends with, or None."""
    if not stop_sequences or not text:
        return None
    for seq in stop_sequences:
        if text.endswith(seq):
            return seq
    return None


def build_anthropic_response(
    ollama_resp: dict, claude_model: str, msg_id: str,
    stop_sequences: list[str] | None = None,
) -> dict:
    message = ollama_resp.get("message", {})
    text = message.get("content", "")
    tool_calls = message.get("tool_calls", [])
    input_tokens = ollama_resp.get("prompt_eval_count") or 0
    output_tokens = ollama_resp.get("eval_count") or 0
    done_reason = ollama_resp.get("done_reason") or "stop"

    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    for tc in tool_calls:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        content.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": fn.get("name", ""),
            "input": args,
        })

    matched_seq = _detect_stop_sequence(text, stop_sequences)
    if tool_calls:
        stop_reason = "tool_use"
    elif matched_seq is not None:
        stop_reason = "stop_sequence"
    else:
        stop_reason = {"stop": "end_turn", "length": "max_tokens"}.get(done_reason, "end_turn")

    if not content:
        content.append({"type": "text", "text": ""})

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": claude_model,
        "stop_reason": stop_reason,
        "stop_sequence": matched_seq,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


async def stream_anthropic_events(
    ollama_stream: AsyncIterator[str],
    claude_model: str,
    msg_id: str,
    ollama_model: str,
    t0: float,
    ctx_size: int | None = None,
    est_input_tokens: int = 0,
    stop_sequences: list[str] | None = None,
) -> AsyncIterator[str]:
    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": claude_model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": est_input_tokens,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    })
    yield sse("ping", {"type": "ping"})

    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"
    matched_stop_seq: str | None = None
    accumulated_text = ""
    text_block_started = False
    accumulated_tool_calls: list[dict] = []

    async for line in ollama_stream:
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg = data.get("message", {})

        if text_delta := msg.get("content", ""):
            accumulated_text += text_delta
            if not text_block_started:
                yield sse("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })
                text_block_started = True
            yield sse("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text_delta},
            })

        if tool_calls := msg.get("tool_calls", []):
            accumulated_tool_calls.extend(tool_calls)

        if data.get("done"):
            input_tokens = data.get("prompt_eval_count") or 0
            output_tokens = data.get("eval_count") or 0
            matched_stop_seq = _detect_stop_sequence(accumulated_text, stop_sequences)
            if accumulated_tool_calls:
                stop_reason = "tool_use"
            elif matched_stop_seq is not None:
                stop_reason = "stop_sequence"
            else:
                stop_reason = {"stop": "end_turn", "length": "max_tokens"}.get(
                    data.get("done_reason", "stop"), "end_turn"
                )
            elapsed = time.monotonic() - t0
            tps = output_tokens / elapsed if elapsed > 0 else 0
            log.info(
                "⇐ ollama  [%s]  %s  %din %dout tok  %.3fs  %.1f tok/s",
                msg_id[:8], stop_reason, input_tokens, output_tokens, elapsed, tps,
            )
            if ctx_size is not None and input_tokens > ctx_size:
                log.warning(
                    "\033[31m⚠ context overflow  [%s]  actual %d tokens > configured ctx=%d\033[0m",
                    msg_id[:8], input_tokens, ctx_size,
                )

    # Close text block if one was opened
    next_index = 0
    if text_block_started:
        yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        next_index = 1

    # Emit tool_use blocks
    for tc in accumulated_tool_calls:
        fn = tc.get("function", {})
        tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
        yield sse("content_block_start", {
            "type": "content_block_start",
            "index": next_index,
            "content_block": {"type": "tool_use", "id": tool_id, "name": fn.get("name", ""), "input": {}},
        })
        args = fn.get("arguments", {})
        partial_json = args if isinstance(args, str) else json.dumps(args)
        yield sse("content_block_delta", {
            "type": "content_block_delta",
            "index": next_index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        })
        yield sse("content_block_stop", {"type": "content_block_stop", "index": next_index})
        next_index += 1

    # Fallback: if no content was emitted at all, send an empty text block
    if next_index == 0:
        yield sse("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        yield sse("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": ""},
        })
        yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": matched_stop_seq},
        "usage": {"output_tokens": output_tokens},
    })
    yield sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Anthropic passthrough
# ---------------------------------------------------------------------------

def _resolve_anthropic_api_key(cfg: dict, request: Request) -> str:
    """Return the API key to use for Anthropic passthrough.

    Prefers a configured key; falls back to the key from the incoming request
    (unless it is the dummy value 'proxy').
    """
    if configured := cfg.get("anthropic_api_key"):
        return configured
    from_header = request.headers.get("x-api-key", "")
    if from_header and from_header.lower() != "proxy":
        return from_header
    return ""


async def _forward_to_anthropic(
    cfg: dict,
    request: Request,
    body: dict,
    claude_model: str,
    msg_id: str,
) -> Response:
    """Forward the request directly to the Anthropic API."""
    api_key = _resolve_anthropic_api_key(cfg, request)
    if not api_key:
        log.error("anthropic passthrough: no API key available")
        raise HTTPException(status_code=500, detail="No Anthropic API key configured for passthrough")

    anthropic_url = f"{cfg['anthropic_base_url'].rstrip('/')}/v1/messages"
    stream = body.get("stream", False)
    turn = len(body.get("messages", []))
    client_ip = request.client.host if request.client else "unknown"
    log.info(
        "→ proxy   [%s]  %s  %s → anthropic  turns=%d  stream=%s",
        msg_id[:8], client_ip, claude_model, turn, stream,
    )

    headers = {
        "x-api-key": api_key,
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
        "content-type": "application/json",
    }
    # Forward optional beta headers
    if beta := request.headers.get("anthropic-beta"):
        headers["anthropic-beta"] = beta

    if stream:
        async def generate():
            def _sse(event: str, data: dict) -> str:
                return f"event: {event}\ndata: {json.dumps(data)}\n\n"

            def _error_stream(error_text: str):
                """Yield a complete SSE error as a valid Anthropic message."""
                yield _sse("message_start", {
                    "type": "message_start",
                    "message": {"id": msg_id, "type": "message", "role": "assistant",
                                "content": [], "model": claude_model,
                                "stop_reason": None, "stop_sequence": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0,
                                          "cache_creation_input_tokens": 0,
                                          "cache_read_input_tokens": 0}},
                })
                yield _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                                    "content_block": {"type": "text", "text": ""}})
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                   "delta": {"type": "text_delta", "text": error_text}})
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
                yield _sse("message_delta", {"type": "message_delta",
                                             "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                             "usage": {"output_tokens": 0}})
                yield _sse("message_stop", {"type": "message_stop"})

            log.info("⇒ anthropic [%s]  %s  stream", msg_id[:8], claude_model)
            t0 = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None)) as client:
                    async with client.stream("POST", anthropic_url, json=body, headers=headers) as resp:
                        if resp.status_code != 200:
                            error_body = await resp.aread()
                            error_text = error_body.decode()[:500]
                            log.error("anthropic %d: %s", resp.status_code, error_text)
                            # Parse error message if possible
                            try:
                                err_json = json.loads(error_body)
                                err_msg = err_json.get("error", {}).get("message", error_text)
                            except (json.JSONDecodeError, AttributeError):
                                err_msg = error_text
                            for event in _error_stream(f"[Anthropic error {resp.status_code}] {err_msg}"):
                                yield event
                            return
                        async for line in resp.aiter_lines():
                            if line:
                                yield line + "\n"
            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as exc:
                log.error("anthropic connection error: %s", exc)
                for event in _error_stream(f"[Anthropic connection error: {exc}]"):
                    yield event
                return
            elapsed = time.monotonic() - t0
            log.info("← proxy   [%s]  anthropic ok  %.3fs", msg_id[:8], elapsed)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        log.info("⇒ anthropic [%s]  %s  non-stream", msg_id[:8], claude_model)
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None)) as client:
            resp = await client.post(anthropic_url, json=body, headers=headers)
        elapsed = time.monotonic() - t0
        if resp.status_code != 200:
            log.error("anthropic %d: %s", resp.status_code, resp.text[:500])
        else:
            log.info("← proxy   [%s]  anthropic ok  %.3fs", msg_id[:8], elapsed)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type="application/json",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    cfg = request.app.state.config
    body = await request.json()

    claude_model = body.get("model", "claude-sonnet-4-6")
    msg_id = f"msg_{uuid.uuid4().hex}"

    # --- Anthropic passthrough (no local model defined, or explicit "anthropic" value) ---
    ollama_model = resolve_ollama_model(cfg, claude_model)
    if ollama_model is None or ollama_model == _ANTHROPIC_PASSTHROUGH:
        return await _forward_to_anthropic(cfg, request, body, claude_model, msg_id)

    # --- Ollama path ---
    ctx_size = resolve_context_size(cfg, claude_model)
    ollama_messages = build_ollama_messages(body)
    stream = body.get("stream", False)

    if ctx_size is not None:
        ollama_messages = trim_messages(ollama_messages, ctx_size, msg_id)

    turn = len(body.get("messages", []))
    client_ip = request.client.host if request.client else "unknown"
    log.info(
        "→ proxy   [%s]  %s  %s → %s  turns=%d  stream=%s",
        msg_id[:8], client_ip, claude_model, ollama_model, turn, stream,
    )

    ollama_payload: dict[str, Any] = {
        "model": ollama_model,
        "messages": ollama_messages,
        "stream": stream,
    }

    if claude_tools := body.get("tools"):
        ollama_payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in claude_tools
        ]

    options: dict[str, Any] = {}
    if ctx_size is not None:
        options["num_ctx"] = ctx_size
    if "max_tokens" in body:
        options["num_predict"] = body["max_tokens"]
    if "temperature" in body:
        options["temperature"] = body["temperature"]
    if "top_p" in body:
        options["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        options["stop"] = body["stop_sequences"]
    if options:
        ollama_payload["options"] = options

    ollama_url = f"{cfg['ollama_base_url'].rstrip('/')}/api/chat"

    if stream:
        async def generate():
            def _sse(event: str, data: dict) -> str:
                return f"event: {event}\ndata: {json.dumps(data)}\n\n"

            def _error_stream(error_text: str):
                """Yield a complete SSE sequence containing an error message."""
                yield _sse("message_start", {
                    "type": "message_start",
                    "message": {"id": msg_id, "type": "message", "role": "assistant",
                                "content": [], "model": claude_model,
                                "stop_reason": None, "stop_sequence": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0,
                                          "cache_creation_input_tokens": 0,
                                          "cache_read_input_tokens": 0}},
                })
                yield _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                                    "content_block": {"type": "text", "text": ""}})
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                   "delta": {"type": "text_delta", "text": error_text}})
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
                yield _sse("message_delta", {"type": "message_delta",
                                             "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                             "usage": {"output_tokens": 0}})
                yield _sse("message_stop", {"type": "message_stop"})

            log.info("⇒ ollama  [%s]  %s  ctx=%s", msg_id[:8], ollama_model, ctx_size)
            t0 = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None)) as client:
                    async with client.stream("POST", ollama_url, json=ollama_payload) as resp:
                        if resp.status_code != 200:
                            error_body = await resp.aread()
                            log.error("ollama %d: %s", resp.status_code, error_body.decode())
                            for event in _error_stream(f"[Ollama error {resp.status_code}]"):
                                yield event
                            return
                        est_tok = estimate_tokens(ollama_messages)
                        async for event in stream_anthropic_events(resp.aiter_lines(), claude_model, msg_id, ollama_model, t0, ctx_size, est_input_tokens=est_tok, stop_sequences=body.get("stop_sequences")):
                            yield event
            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as exc:
                log.error("ollama connection error: %s", exc)
                for event in _error_stream(f"[Ollama connection error: {exc}]"):
                    yield event
                return
            log.info("← proxy   [%s]  ok", msg_id[:8])

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        log.info("⇒ ollama  [%s]  %s  ctx=%s", msg_id[:8], ollama_model, ctx_size)
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None)) as client:
            resp = await client.post(ollama_url, json=ollama_payload)
            if resp.status_code != 200:
                log.error("ollama %d: %s", resp.status_code, resp.text)
                error_body = {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"Ollama returned {resp.status_code}: {resp.text}",
                    },
                }
                return Response(
                    content=json.dumps(error_body),
                    status_code=resp.status_code,
                    media_type="application/json",
                )
            ollama_data = resp.json()
        elapsed = time.monotonic() - t0
        in_tok = ollama_data.get("prompt_eval_count") or 0
        out_tok = ollama_data.get("eval_count") or 0
        tps = out_tok / elapsed if elapsed > 0 else 0
        stop_reason = "tool_use" if ollama_data.get("message", {}).get("tool_calls") else "end_turn"
        log.info(
            "⇐ ollama  [%s]  %s  %din %dout tok  %.3fs  %.1f tok/s",
            msg_id[:8], stop_reason, in_tok, out_tok, elapsed, tps,
        )
        if ctx_size is not None and in_tok > ctx_size:
            log.warning(
                "\033[31m⚠ context overflow  [%s]  actual %d tokens > configured ctx=%d\033[0m",
                msg_id[:8], in_tok, ctx_size,
            )
        log.info("← proxy   [%s]  ok", msg_id[:8])
        return Response(
            content=json.dumps(build_anthropic_response(ollama_data, claude_model, msg_id, stop_sequences=body.get("stop_sequences"))),
            media_type="application/json",
        )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "claude-opus-4-6", "object": "model"},
            {"id": "claude-sonnet-4-6", "object": "model"},
            {"id": "claude-haiku-4-5-20251001", "object": "model"},
        ],
    }


@app.get("/health")
async def health(request: Request):
    cfg = request.app.state.config
    return {"status": "ok", "ollama_base_url": cfg["ollama_base_url"], "model_map": cfg["model_map"]}


def main():
    import uvicorn
    cfg = load_config()
    uvicorn.run("proxy:app", host=cfg["proxy_host"], port=cfg["proxy_port"], reload=False)


if __name__ == "__main__":
    main()
