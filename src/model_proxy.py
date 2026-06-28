"""
OpenAI-compatible HTTP proxy for Gemini and Anthropic models.

Routes /v1/chat/completions requests to the installed CLI tools,
using your existing account subscriptions — NO API keys needed.

  - gemini-*  → `npx @google/gemini-cli -p` (Google account subscription)
  - claude-*  → `claude -p`                 (Anthropic account subscription)
  - other     → 400 Unsupported model

Both streaming (SSE) and non-streaming responses are supported.

Usage:
    python src/model_proxy.py                     # starts on port 9090
    MODEL_PROXY_PORT=8080 python src/model_proxy.py  # custom port

Endpoints:
    POST /v1/chat/completions   — OpenAI-compatible chat completion
    GET  /v1/models             — list available models
    GET  /health                — liveness check

CLI commands used (must be installed & authenticated):
    claude  — Claude Code CLI (https://claude.ai/code)
    gemini  — Gemini CLI (npm i -g @google/gemini-cli)
"""

import os
import sys
import json
import time
import uuid
import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── CLI discovery ─────────────────────────────────────────────────────────────

def _find_cli(name: str, npm_package: Optional[str] = None) -> Optional[str]:
    """Find a CLI executable. Returns the full path, or None."""
    result = shutil.which(name)
    if result:
        return result
    # On Windows, also check common locations
    if sys.platform == "win32":
        for ext in (".exe", ".cmd", ".bat"):
            result = shutil.which(name + ext)
            if result:
                return result
    return None


CLAUDE_CLI = _find_cli("claude")
GEMINI_CLI = _find_cli("gemini")

# If gemini isn't on PATH directly, we'll use npx
GEMINI_NPX_PACKAGE = "@google/gemini-cli"


def _claude_available() -> bool:
    return CLAUDE_CLI is not None


def _gemini_available() -> bool:
    return GEMINI_CLI is not None or shutil.which("npx") is not None


# ── Request / Response models ─────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=65536)
    stream: bool = False
    top_p: Optional[float] = None
    stop: Optional[list[str]] = None

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str


# ── Helpers: OpenAI-compatible response formatting ────────────────────────────

def _oi_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _oi_timestamp() -> int:
    return int(time.time())


def _format_non_stream_response(model: str, content: str, prompt_tokens: int = 0,
                                 completion_tokens: int = 0) -> dict:
    """Format a non-streaming OpenAI-compatible response."""
    return {
        "id": _oi_completion_id(),
        "object": "chat.completion",
        "created": _oi_timestamp(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _format_stream_chunk(model: str, content: str, chunk_id: str,
                          finish_reason: Optional[str] = None) -> str:
    """Format a single SSE chunk for streaming response."""
    delta = {"role": "assistant", "content": content} if content else {}
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": _oi_timestamp(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ── Prompt construction from messages ─────────────────────────────────────────

def _build_prompt_from_messages(messages: list[Message]) -> str:
    """
    Convert OpenAI-style messages array into a single prompt string
    suitable for CLI stdin.

    System messages become prefix instructions.
    Multi-turn conversations are formatted with role labels.
    """
    parts = []
    system_parts = []

    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        elif msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")

    prompt = "\n\n".join(parts)

    if system_parts:
        system_text = "\n\n".join(system_parts)
        prompt = f"[System Instructions]\n{system_text}\n\n[Conversation]\n{prompt}"

    return prompt


# ── Backend: Claude CLI ───────────────────────────────────────────────────────

async def _route_claude_sync(request: ChatCompletionRequest) -> dict:
    """Non-streaming Claude CLI call → OpenAI-compatible response."""
    if not _claude_available():
        raise EnvironmentError(
            "claude CLI not found. Install it from https://claude.ai/code"
        )

    prompt = _build_prompt_from_messages(request.messages)

    # Build command: claude -p --model <model> --output-format json
    cmd = [
        CLAUDE_CLI,
        "-p",
        "--model", request.model,
        "--output-format", "json",
    ]

    # Run in a temp directory to avoid project context interference
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(Path.home()),  # avoid picking up local CLAUDE.md
        env={**os.environ, "CLAUDE_CODE_SIMPLE": "1"},  # minimal mode
    )

    stdout, stderr = await asyncio.wait_for(
        proc.communicate(prompt.encode()),
        timeout=300,
    )

    if proc.returncode != 0:
        err_msg = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"claude CLI exited with code {proc.returncode}: {err_msg}")

    # Parse JSON output from claude --output-format json
    try:
        result = json.loads(stdout.decode())
        content = result.get("result", "")
        # Extract token usage if available
        usage = result.get("usage", {})
        pt = usage.get("input_tokens", 0)
        ct = usage.get("output_tokens", 0)
    except (json.JSONDecodeError, KeyError):
        # Fallback: treat raw stdout as text
        content = stdout.decode(errors="replace").strip()
        pt, ct = 0, 0

    return _format_non_stream_response(request.model, content, pt, ct)


async def _route_claude_stream(request: ChatCompletionRequest):
    """Streaming Claude CLI call → SSE chunks."""
    if not _claude_available():
        raise EnvironmentError(
            "claude CLI not found. Install it from https://claude.ai/code"
        )

    chunk_id = _oi_completion_id()
    prompt = _build_prompt_from_messages(request.messages)

    cmd = [
        CLAUDE_CLI,
        "-p",
        "--model", request.model,
        "--output-format", "stream-json",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(Path.home()),
        env={**os.environ, "CLAUDE_CODE_SIMPLE": "1"},
    )

    # Send prompt to stdin
    proc.stdin.write(prompt.encode())
    proc.stdin.write_eof()

    # Send initial role chunk
    yield _format_stream_chunk(request.model, "", chunk_id)

    # Stream stdout line by line
    buffer = ""
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        line_str = line.decode(errors="replace").strip()
        if not line_str:
            continue

        # Try to parse as JSON (stream-json format from claude CLI)
        try:
            event = json.loads(line_str)
            # stream-json events have a "type" field
            if event.get("type") == "content_block_delta":
                text = event.get("delta", {}).get("text", "")
                if text:
                    yield _format_stream_chunk(request.model, text, chunk_id)
            elif event.get("type") == "message_stop":
                break
            elif event.get("type") == "result":
                # Final result — emit any remaining content
                text = event.get("result", "")
                if text and not buffer:
                    yield _format_stream_chunk(request.model, text, chunk_id)
            elif "result" in event:
                # Some output formats just have a result field
                text = event["result"]
                if text:
                    yield _format_stream_chunk(request.model, text, chunk_id)
        except json.JSONDecodeError:
            # Plain text line — stream it directly
            if line_str:
                yield _format_stream_chunk(request.model, line_str + "\n", chunk_id)

    # Wait for process to finish
    await proc.wait()

    # Final chunk with finish_reason
    yield _format_stream_chunk(request.model, "", chunk_id, finish_reason="stop")
    yield "data: [DONE]\n\n"


# ── Backend: Gemini CLI ───────────────────────────────────────────────────────

async def _route_gemini_sync(request: ChatCompletionRequest) -> dict:
    """Non-streaming Gemini CLI call → OpenAI-compatible response."""
    if not _gemini_available():
        raise EnvironmentError(
            "gemini CLI not found. Install with: npm i -g @google/gemini-cli"
        )

    prompt = _build_prompt_from_messages(request.messages)

    # Build command: gemini -p --model <model> --output-format json --skip-trust
    if GEMINI_CLI:
        cmd = [
            GEMINI_CLI,
            "-p", prompt,
            "--model", request.model,
            "--output-format", "json",
            "--skip-trust",
        ]
        stdin_data = None
    else:
        # Use npx as fallback
        cmd = [
            shutil.which("npx"),
            GEMINI_NPX_PACKAGE,
            "-p", prompt,
            "--model", request.model,
            "--output-format", "json",
            "--skip-trust",
        ]
        stdin_data = None

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin_data else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(Path.home()),
    )

    if stdin_data:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(stdin_data.encode()),
            timeout=300,
        )
    else:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=300,
        )

    if proc.returncode != 0:
        err_msg = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"gemini CLI exited with code {proc.returncode}: {err_msg}")

    # Parse JSON output from gemini --output-format json
    try:
        result = json.loads(stdout.decode())
        # Gemini CLI JSON output format varies — try common fields
        content = (
            result.get("response", {})
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            if isinstance(result, dict) and "response" in result
            else result.get("text", "") if isinstance(result, dict)
            else str(result)
        )
        if not content:
            # Fallback: try "result" or "content" field
            content = result.get("result", result.get("content", ""))
            if isinstance(content, dict):
                content = content.get("text", str(content))
    except (json.JSONDecodeError, KeyError, IndexError):
        # Fallback: treat raw stdout as text
        content = stdout.decode(errors="replace").strip()

    return _format_non_stream_response(request.model, str(content))


async def _route_gemini_stream(request: ChatCompletionRequest):
    """Streaming Gemini CLI call → SSE chunks."""
    if not _gemini_available():
        raise EnvironmentError(
            "gemini CLI not found. Install with: npm i -g @google/gemini-cli"
        )

    chunk_id = _oi_completion_id()
    prompt = _build_prompt_from_messages(request.messages)

    if GEMINI_CLI:
        cmd = [
            GEMINI_CLI,
            "-p", prompt,
            "--model", request.model,
            "--output-format", "stream-json",
            "--skip-trust",
        ]
    else:
        cmd = [
            shutil.which("npx"),
            GEMINI_NPX_PACKAGE,
            "-p", prompt,
            "--model", request.model,
            "--output-format", "stream-json",
            "--skip-trust",
        ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(Path.home()),
    )

    # Send initial role chunk
    yield _format_stream_chunk(request.model, "", chunk_id)

    # Stream stdout line by line
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        line_str = line.decode(errors="replace").strip()
        if not line_str:
            continue

        # Try to parse as stream-json
        try:
            event = json.loads(line_str)
            # Gemini stream-json format
            if event.get("type") == "content_block_delta":
                text = event.get("delta", {}).get("text", "")
                if text:
                    yield _format_stream_chunk(request.model, text, chunk_id)
            elif "text" in event:
                text = event["text"]
                if text:
                    yield _format_stream_chunk(request.model, text, chunk_id)
            elif "candidates" in event:
                # Gemini API-style chunk
                for cand in event["candidates"]:
                    parts = cand.get("content", {}).get("parts", [])
                    for part in parts:
                        text = part.get("text", "")
                        if text:
                            yield _format_stream_chunk(request.model, text, chunk_id)
        except json.JSONDecodeError:
            # Plain text line
            if line_str:
                yield _format_stream_chunk(request.model, line_str + "\n", chunk_id)

    await proc.wait()

    # Final chunk with finish_reason
    yield _format_stream_chunk(request.model, "", chunk_id, finish_reason="stop")
    yield "data: [DONE]\n\n"


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Model Proxy — Gemini & Claude CLIs via OpenAI-compatible API",
    description=(
        "Routes /v1/chat/completions to Gemini CLI or Claude CLI using your "
        "existing account subscriptions. No API keys needed."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_MODEL_PREFIXES = {"gemini", "claude"}


@app.get("/health")
async def health():
    """Liveness check + CLI availability."""
    return {
        "status": "healthy",
        "claude_cli": {
            "available": _claude_available(),
            "path": CLAUDE_CLI,
        },
        "gemini_cli": {
            "available": _gemini_available(),
            "path": GEMINI_CLI or f"npx {GEMINI_NPX_PACKAGE}",
        },
        "supported_prefixes": sorted(SUPPORTED_MODEL_PREFIXES),
    }


@app.get("/v1/models")
async def list_models():
    """List models available through this proxy."""
    models = [
        # Gemini models
        ModelInfo(id="gemini-2.5-pro", owned_by="google"),
        ModelInfo(id="gemini-2.5-flash", owned_by="google"),
        ModelInfo(id="gemini-2.0-flash", owned_by="google"),
        # Claude models
        ModelInfo(id="claude-sonnet-4-5-20250514", owned_by="anthropic"),
        ModelInfo(id="claude-opus-4-20250514", owned_by="anthropic"),
        ModelInfo(id="claude-3-opus-20240229", owned_by="anthropic"),
        ModelInfo(id="claude-3-5-sonnet-20241022", owned_by="anthropic"),
    ]
    return {"object": "list", "data": [m.model_dump() for m in models]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.
    Routes to Gemini CLI or Claude CLI based on model name prefix.
    Uses your existing account subscriptions — no API keys needed.
    """
    model_lower = request.model.lower()
    is_gemini = model_lower.startswith("gemini")
    is_claude = model_lower.startswith("claude")

    if not is_gemini and not is_claude:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {request.model}. "
                   f"This proxy handles models with prefixes: {sorted(SUPPORTED_MODEL_PREFIXES)}",
        )

    try:
        if request.stream:
            if is_gemini:
                return StreamingResponse(
                    _route_gemini_stream(request),
                    media_type="text/event-stream",
                )
            else:
                return StreamingResponse(
                    _route_claude_stream(request),
                    media_type="text/event-stream",
                )
        else:
            if is_gemini:
                return await _route_gemini_sync(request)
            else:
                return await _route_claude_sync(request)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="CLI request timed out (300s)")


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PROXY_PORT", "9090"))
    print(f"[ModelProxy] Starting on port {port}")
    print(f"[ModelProxy] Claude CLI:  {'✅ ' + CLAUDE_CLI if _claude_available() else '❌ not found'}")
    print(f"[ModelProxy] Gemini CLI:  {'✅ ' + (GEMINI_CLI or 'npx @google/gemini-cli') if _gemini_available() else '❌ not found'}")
    print(f"[ModelProxy] No API keys needed — uses your account subscriptions via CLI")
    uvicorn.run(
        "model_proxy:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        timeout_keep_alive=300,
    )
