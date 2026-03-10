"""
Claude CLI -> Ollama proxy

Exposes the Anthropic Messages API (/v1/messages) and forwards requests
to a configurable Ollama endpoint, translating formats in both directions.

Configuration is loaded at startup from config.yaml (or a path set via
CONFIG_PATH env var). Individual values can be overridden with env vars:

  OLLAMA_BASE_URL              overrides ollama_base_url
  OLLAMA_MODEL_MAP_<KEY>       overrides a model_map entry, e.g.
                               OLLAMA_MODEL_MAP_OPUS=llama3.2:70b
  OLLAMA_CONTEXT_SIZE_<KEY>    overrides a context_size entry, e.g.
                               OLLAMA_CONTEXT_SIZE_SONNET=32768
  PROXY_HOST                   overrides proxy_host (default: localhost)
  PROXY_PORT                   overrides proxy_port (default: 8082)

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
import yaml
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

_CONFIG_PATH = Path(os.getenv("CONFIG_PATH", Path(__file__).parent / "config.yaml"))

_DEFAULTS: dict[str, Any] = {
    "ollama_base_url": "http://localhost:11434",
    "proxy_host": "localhost",
    "proxy_port": 8082,
    "model_map": {
        "opus":    "llama3.2:70b",
        "sonnet":  "llama3.2",
        "haiku":   "llama3.2:1b",
        "default": "llama3.2",
    },
    "context_size": {
        "opus":    32768,
        "sonnet":  16384,
        "haiku":   8192,
        "default": 8192,
    },
}


def load_config() -> dict[str, Any]:
    cfg: dict[str, Any] = json.loads(json.dumps(_DEFAULTS))  # deep copy

    if _CONFIG_PATH.exists():
        with _CONFIG_PATH.open() as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg["ollama_base_url"] = file_cfg.get("ollama_base_url", cfg["ollama_base_url"])
        cfg["proxy_host"] = file_cfg.get("proxy_host", cfg["proxy_host"])
        cfg["proxy_port"] = int(file_cfg.get("proxy_port", cfg["proxy_port"]))
        if "model_map" in file_cfg and isinstance(file_cfg["model_map"], dict):
            cfg["model_map"].update(file_cfg["model_map"])
        if "context_size" in file_cfg and isinstance(file_cfg["context_size"], dict):
            cfg["context_size"].update(file_cfg["context_size"])
        log.info("config    loaded from %s", _CONFIG_PATH)
    else:
        log.warning("config    not found at %s — using defaults", _CONFIG_PATH)

    # Env var overrides
    if base_url := os.getenv("OLLAMA_BASE_URL"):
        cfg["ollama_base_url"] = base_url
    if host := os.getenv("PROXY_HOST"):
        cfg["proxy_host"] = host
    if port := os.getenv("PROXY_PORT"):
        cfg["proxy_port"] = int(port)

    for key, val in os.environ.items():
        # OLLAMA_MODEL_MAP_OPUS=llama3.2:70b  ->  model_map["opus"] = "llama3.2:70b"
        if key.startswith("OLLAMA_MODEL_MAP_"):
            map_key = key[len("OLLAMA_MODEL_MAP_"):].lower()
            cfg["model_map"][map_key] = val
        # OLLAMA_CONTEXT_SIZE_SONNET=32768  ->  context_size["sonnet"] = 32768
        elif key.startswith("OLLAMA_CONTEXT_SIZE_"):
            ctx_key = key[len("OLLAMA_CONTEXT_SIZE_"):].lower()
            cfg["context_size"][ctx_key] = int(val)

    return cfg


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.config = cfg
    log.info("ollama    %s", cfg["ollama_base_url"])
    for key in cfg["model_map"]:
        label = key if key != "default" else "(default)"
        ctx = cfg["context_size"].get(key, cfg["context_size"].get("default", "—"))
        log.info("model     %-10s → %-22s ctx=%s", label, cfg["model_map"][key], ctx)

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


def resolve_ollama_model(cfg: dict, claude_model: str) -> str:
    key = _match_key(cfg["model_map"], claude_model)
    return cfg["model_map"][key] if key else cfg["model_map"].get("default", "llama3.2")


def resolve_context_size(cfg: dict, claude_model: str) -> int | None:
    key = _match_key(cfg["context_size"], claude_model)
    return cfg["context_size"].get(key or "default")


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
                    log.info("  tool_use  %s\n%s", block.get("name"), json.dumps(block.get("input", {}), indent=2))
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


def build_anthropic_response(ollama_resp: dict, claude_model: str, msg_id: str) -> dict:
    message = ollama_resp.get("message", {})
    text = message.get("content", "")
    tool_calls = message.get("tool_calls", [])
    input_tokens = ollama_resp.get("prompt_eval_count", 0)
    output_tokens = ollama_resp.get("eval_count", 0)
    done_reason = ollama_resp.get("done_reason", "stop")

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

    if tool_calls:
        stop_reason = "tool_use"
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
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


async def stream_anthropic_events(
    ollama_stream: AsyncIterator[str],
    claude_model: str,
    msg_id: str,
    ollama_model: str,
    t0: float,
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
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield sse("ping", {"type": "ping"})

    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"
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
            input_tokens = data.get("prompt_eval_count", 0)
            output_tokens = data.get("eval_count", 0)
            if accumulated_tool_calls:
                stop_reason = "tool_use"
            else:
                stop_reason = {"stop": "end_turn", "length": "max_tokens"}.get(
                    data.get("done_reason", "stop"), "end_turn"
                )
            elapsed = time.monotonic() - t0
            log.info(
                "⇐ ollama  [%s]  %s  %din %dout tok  %.3fs",
                msg_id[:8], stop_reason, input_tokens, output_tokens, elapsed,
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
        yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    cfg = request.app.state.config
    body = await request.json()

    claude_model = body.get("model", "claude-sonnet-4-6")
    ollama_model = resolve_ollama_model(cfg, claude_model)
    ctx_size = resolve_context_size(cfg, claude_model)
    ollama_messages = build_ollama_messages(body)
    stream = body.get("stream", False)
    msg_id = f"msg_{uuid.uuid4().hex}"

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
    if options:
        ollama_payload["options"] = options

    ollama_url = f"{cfg['ollama_base_url'].rstrip('/')}/api/chat"

    if stream:
        async def generate():
            log.info("⇒ ollama  [%s]  %s  ctx=%s", msg_id[:8], ollama_model, ctx_size)
            t0 = time.monotonic()
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None)) as client:
                async with client.stream("POST", ollama_url, json=ollama_payload) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        log.error("ollama %d: %s", resp.status_code, error_body.decode())
                        # Headers already sent; emit a valid terminal SSE sequence
                        # so the client doesn't hang waiting for message_stop.
                        def _sse(event: str, data: dict) -> str:
                            return f"event: {event}\ndata: {json.dumps(data)}\n\n"
                        yield _sse("message_start", {
                            "type": "message_start",
                            "message": {"id": msg_id, "type": "message", "role": "assistant",
                                        "content": [], "model": claude_model,
                                        "stop_reason": None, "stop_sequence": None,
                                        "usage": {"input_tokens": 0, "output_tokens": 0}},
                        })
                        yield _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                                            "content_block": {"type": "text", "text": ""}})
                        yield _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                           "delta": {"type": "text_delta",
                                                                     "text": f"[Ollama error {resp.status_code}]"}})
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
                        yield _sse("message_delta", {"type": "message_delta",
                                                     "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                                     "usage": {"output_tokens": 0}})
                        yield _sse("message_stop", {"type": "message_stop"})
                        return
                    async for event in stream_anthropic_events(resp.aiter_lines(), claude_model, msg_id, ollama_model, t0):
                        yield event
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
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            ollama_data = resp.json()
        elapsed = time.monotonic() - t0
        in_tok = ollama_data.get("prompt_eval_count", 0)
        out_tok = ollama_data.get("eval_count", 0)
        stop_reason = "tool_use" if ollama_data.get("message", {}).get("tool_calls") else "end_turn"
        log.info(
            "⇐ ollama  [%s]  %s  %din %dout tok  %.3fs",
            msg_id[:8], stop_reason, in_tok, out_tok, elapsed,
        )
        log.info("← proxy   [%s]  ok", msg_id[:8])
        return Response(
            content=json.dumps(build_anthropic_response(ollama_data, claude_model, msg_id)),
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
