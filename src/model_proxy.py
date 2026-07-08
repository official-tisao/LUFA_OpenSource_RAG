"""
OpenAI-compatible HTTP proxy for Gemini and Anthropic models.

Routes /v1/chat/completions requests to the installed CLI tools,
using your existing account subscriptions — NO API keys needed.

  - gemini-*  → `antigravity chat --mode ask` (Google Antigravity IDE — Gemini via Google account)
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
    antigravity  — Google Antigravity (VS Code fork with built-in Gemini)
    claude       — Claude Code CLI (https://claude.ai/code)

Gemini via antigravity:
    The `antigravity chat --mode ask` command opens a GUI chat panel.
    This proxy uses a temp-file bridge: it writes the prompt to a temp
    file and launches antigravity on it, then polls a response file until
    the answer appears. The GEMINI_RESPONSE_DIR env var controls where
    response files are written (default: system temp dir).

    For headless/non-GUI environments, set GEMINI_FALLBACK_CLI=1 to
    fall back to the Gemini CLI (npm i -g @google/gemini-cli) instead.
"""

import os
import re
import sys
import json
import time
import uuid
import asyncio
import shutil
import tempfile
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


CLAUDE_CLI       = _find_cli("claude")
ANTIGRAVITY_CLI  = _find_cli("antigravity")
GEMINI_CLI       = _find_cli("gemini")
GEMINI_NPX_PACKAGE = "@google/gemini-cli"

# If GEMINI_FALLBACK_CLI is set, prefer the headless gemini CLI over antigravity
_GEMINI_FALLBACK = os.environ.get("GEMINI_FALLBACK_CLI", "").lower() in ("1", "true", "yes")

# Response polling directory for the antigravity file bridge.
# Defaults to <system temp>/antigravity_responses (on Windows this resolves to
# %LOCALAPPDATA%\Temp\antigravity_responses — where the companion writes files).
# Override with the GEMINI_RESPONSE_DIR env var.
GEMINI_RESPONSE_DIR = os.environ.get(
    "GEMINI_RESPONSE_DIR",
    str(Path(tempfile.gettempdir()) / "antigravity_responses"),
)

# Antigravity file-bridge polling cadence / limits.
POLL_INTERVAL = float(os.environ.get("GEMINI_POLL_INTERVAL", "2.0"))        # seconds between reads
RESPONSE_TIMEOUT = float(os.environ.get("GEMINI_RESPONSE_TIMEOUT", "600"))  # give up after N seconds
STABILITY_CYCLES = int(os.environ.get("GEMINI_STABILITY_CYCLES", "2"))      # plain-text: unchanged N polls = done


def _claude_available() -> bool:
    return CLAUDE_CLI is not None


def _gemini_available() -> bool:
    """Gemini is available if antigravity is installed (or gemini CLI as fallback)."""
    if _GEMINI_FALLBACK:
        return GEMINI_CLI is not None or shutil.which("npx") is not None
    return ANTIGRAVITY_CLI is not None


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


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for usage reporting when the
    backend does not return real token counts."""
    if not text:
        return 0
    return max(1, len(text) // 4)


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


# ── Backend: Gemini via Antigravity ────────────────────────────────────────────

def _build_gemini_cmd(prompt: str, model: str, output_format: str = "text") -> list:
    """
    Build the command to invoke Gemini. Uses antigravity chat by default,
    falls back to gemini CLI if GEMINI_FALLBACK_CLI=1.
    """
    if _GEMINI_FALLBACK and (GEMINI_CLI or shutil.which("npx")):
        # Headless gemini CLI mode (for non-GUI environments)
        if GEMINI_CLI:
            base = [GEMINI_CLI]
        else:
            base = [shutil.which("npx"), GEMINI_NPX_PACKAGE]
        return base + [
            "-p", prompt,
            "--model", model,
            "--output-format", output_format,
            "--skip-trust",
        ]
    else:
        # Antigravity GUI mode — launches chat in 'ask' mode (no tool use)
        # Antigravity chat reads from stdin with '-' or takes prompt as arg
        return [
            ANTIGRAVITY_CLI,
            "chat",
            "--mode", "ask",
            prompt,
        ]


# Matches a Windows path inside an antigravity_responses folder ending in a
# response-file extension — used to discover where antigravity wrote the answer.
_RESPONSE_PATH_RE = re.compile(
    r'([A-Za-z]:\\[^\s"\'<>|]*antigravity_responses[^\s"\'<>|]*\.(?:json|txt|md|ndjson))',
    re.IGNORECASE,
)


def _request_files(request_id: str):
    """Return (response, prompt, request) file paths for a request id."""
    resp_dir = Path(GEMINI_RESPONSE_DIR)
    resp_dir.mkdir(parents=True, exist_ok=True)
    return (
        resp_dir / f"{request_id}.json",
        resp_dir / f"{request_id}_prompt.txt",
        resp_dir / f"{request_id}_request.json",
    )


def _write_request_files(request_id: str, prompt: str, model: str) -> Path:
    """
    Write the prompt + a request descriptor so an antigravity companion knows
    what to answer and where to write the response, and seed a pending response
    file the proxy will poll. Returns the response file path.

    Companion contract — write the answer to <request_id>.json as either:
        {"status": "complete", "content": "<full answer>"}
      (or stream partials with {"status": "pending", "content": "<so far>"}),
    OR write plain text / markdown (optionally ending with a <<END>> sentinel).
    """
    resp_file, prompt_file, request_file = _request_files(request_id)
    prompt_file.write_text(prompt, encoding="utf-8")
    request_file.write_text(json.dumps({
        "id": request_id,
        "model": model,
        "prompt": prompt,
        "response_path": str(resp_file),
        "status": "pending",
    }, ensure_ascii=False), encoding="utf-8")
    resp_file.write_text(json.dumps({"status": "pending", "content": ""}), encoding="utf-8")
    return resp_file


def _read_response_content(path: Path):
    """
    Read a response file in JSON or plain-text form → (content, is_complete).
      JSON: {"status": "...", "content"/"text"/"response"/"result": "..."}
      Text: raw content; a trailing <<END>> sentinel marks completion.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return "", False
    if not raw.strip():
        return "", False

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            status = str(data.get("status", "")).strip().lower()
            content = data.get("content")
            if content in (None, ""):
                content = data.get("text") or data.get("response") or data.get("result") or ""
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False)
            complete = status in ("complete", "done", "finished", "ok", "success")
            if "status" not in data and content:   # JSON w/ content but no status = final
                complete = True
            return str(content), complete
    except json.JSONDecodeError:
        pass

    if "<<END>>" in raw:
        return raw.split("<<END>>")[0].strip(), True
    return raw.rstrip(), False


def _discover_response(request_id: str, since_ts: float, stdout_text: str = ""):
    """
    Find the file the answer is (being) written to and read it.
    Returns (path, content, complete) — or (None, "", False) when nothing
    usable exists yet.

    IMPORTANT: a file is only accepted if it actually carries content or a
    complete status. The proxy seeds <request_id>.json itself with a pending
    stub, so bare existence must NOT count as discovery (that previously made
    the poller lock onto its own empty stub and wait forever).

    Priority:
      1. deterministic  <dir>/<request_id>.json  (companion contract)
      2. a path regex-parsed from antigravity stdout
      3. response-like files in the dir modified since launch (newest first)
    """
    resp_dir = Path(GEMINI_RESPONSE_DIR)
    candidates = []

    deterministic = resp_dir / f"{request_id}.json"
    if deterministic.exists():
        candidates.append(deterministic)

    if stdout_text:
        m = _RESPONSE_PATH_RE.search(stdout_text)
        if m:
            p = Path(m.group(1))
            if p.exists() and p not in candidates:
                candidates.append(p)

    if resp_dir.exists():
        recent = [
            p for p in resp_dir.glob("*")
            if p.suffix.lower() in (".json", ".txt", ".md", ".ndjson")
            and not p.name.endswith("_prompt.txt")
            and not p.name.endswith("_request.json")
            and p.stat().st_mtime >= since_ts - 1
        ]
        recent.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for p in recent:
            if p not in candidates:
                candidates.append(p)

    for p in candidates:
        content, complete = _read_response_content(p)
        if content or complete:
            return p, content, complete

    return None, "", False


async def _launch_antigravity(prompt: str, model: str):
    """
    Launch `antigravity chat` (GUI) fire-and-forget and return (proc, stdout_acc).
    stdout is drained in the background so a printed response path can be regexed;
    we never block on the GUI process exiting.
    """
    cmd = _build_gemini_cmd(prompt, model)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(Path.home()),
    )
    stdout_acc = {"text": ""}

    async def _drain():
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                stdout_acc["text"] += line.decode(errors="replace")
        except Exception:
            pass

    asyncio.ensure_future(_drain())
    return proc, stdout_acc


async def _poll_antigravity_file(request_id: str, since_ts: float, stdout_acc: dict) -> str:
    """
    Poll the response dir every POLL_INTERVAL until a response with content is
    complete (JSON status) or stops growing (plain text), or RESPONSE_TIMEOUT.
    Re-discovers every cycle — never locks onto the proxy's own pending stub.
    """
    start = time.time()
    last_content = ""
    stable = 0
    cycles = 0
    while (time.time() - start) < RESPONSE_TIMEOUT:
        await asyncio.sleep(POLL_INTERVAL)
        cycles += 1
        resp_path, content, complete = _discover_response(
            request_id, since_ts, stdout_acc.get("text", ""))

        if resp_path is None:
            if cycles % 5 == 0:  # ~every 10s at the default 2s interval
                print(f"[ModelProxy] Waiting for antigravity response "
                      f"({time.time() - start:.0f}s elapsed) — expecting "
                      f"{Path(GEMINI_RESPONSE_DIR) / (request_id + '.json')}")
            continue

        if complete and content:
            print(f"[ModelProxy] Response complete from {resp_path.name} "
                  f"({len(content)} chars, {time.time() - start:.0f}s)")
            return content
        if content and content == last_content:
            stable += 1
            if stable >= STABILITY_CYCLES:
                print(f"[ModelProxy] Response stable from {resp_path.name} "
                      f"({len(content)} chars, {time.time() - start:.0f}s)")
                return content
        else:
            stable = 0
            last_content = content

    if last_content:
        return last_content
    raise TimeoutError(
        f"Antigravity response not received within {RESPONSE_TIMEOUT:.0f}s. "
        f"Nothing wrote content to {GEMINI_RESPONSE_DIR}. Either install a companion "
        f"that writes answers to <request_id>.json there, or set GEMINI_FALLBACK_CLI=1 "
        f"to use the headless gemini CLI."
    )


async def _stream_antigravity_file(request_id: str, since_ts: float, stdout_acc: dict,
                                   model: str, chunk_id: str):
    """Yield SSE deltas as the response file grows; finish on complete/stable/timeout."""
    start = time.time()
    emitted = ""
    last_content = ""
    stable = 0

    yield _format_stream_chunk(model, "", chunk_id)  # initial role chunk

    while (time.time() - start) < RESPONSE_TIMEOUT:
        await asyncio.sleep(POLL_INTERVAL)
        resp_path, content, complete = _discover_response(
            request_id, since_ts, stdout_acc.get("text", ""))
        if resp_path is None:
            continue

        # Emit only the newly-appended portion (common append-only case);
        # on a rewrite, emit whatever is beyond what we've already sent.
        if content and content.startswith(emitted):
            delta = content[len(emitted):]
        elif content and len(content) > len(emitted):
            delta = content[len(emitted):]
        else:
            delta = ""
        if delta:
            yield _format_stream_chunk(model, delta, chunk_id)
            emitted += delta

        if complete and content:
            break
        if content and content == last_content:
            stable += 1
            if stable >= STABILITY_CYCLES:
                break
        else:
            stable = 0
            last_content = content

    yield _format_stream_chunk(model, "", chunk_id, finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _route_gemini_sync(request: ChatCompletionRequest) -> dict:
    """Non-streaming Gemini call (via antigravity or gemini CLI) → OpenAI-compatible response."""
    if not _gemini_available():
        if _GEMINI_FALLBACK:
            raise EnvironmentError(
                "gemini CLI not found. Install with: npm i -g @google/gemini-cli"
            )
        raise EnvironmentError(
            "antigravity not found. Install Google Antigravity IDE, "
            "or set GEMINI_FALLBACK_CLI=1 to use the headless gemini CLI instead."
        )

    prompt = _build_prompt_from_messages(request.messages)
    request_id = _oi_completion_id()

    if _GEMINI_FALLBACK:
        # ── Headless mode: gemini CLI with stdout ──
        cmd = _build_gemini_cmd(prompt, request.model, output_format="json")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path.home()),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"gemini CLI exited with code {proc.returncode}: {err_msg}")

        # Parse JSON output from gemini --output-format json
        try:
            result = json.loads(stdout.decode())
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
                content = result.get("result", result.get("content", ""))
                if isinstance(content, dict):
                    content = content.get("text", str(content))
        except (json.JSONDecodeError, KeyError, IndexError):
            content = stdout.decode(errors="replace").strip()

        return _format_non_stream_response(
            request.model, str(content),
            _estimate_tokens(prompt), _estimate_tokens(str(content)),
        )

    else:
        # ── Antigravity GUI mode (file bridge) ──
        # Launch antigravity chat fire-and-forget, then poll the response file
        # every POLL_INTERVAL seconds until the answer is complete. The companion
        # writes the answer to <request_id>.json (see _write_request_files).
        since_ts = time.time()
        _write_request_files(request_id, prompt, request.model)
        _proc, stdout_acc = await _launch_antigravity(prompt, request.model)

        try:
            content = await _poll_antigravity_file(request_id, since_ts, stdout_acc)
        finally:
            # Leave the antigravity GUI running; just clean up our bridge files.
            for f in _request_files(request_id):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass

        return _format_non_stream_response(
            request.model, str(content),
            _estimate_tokens(prompt), _estimate_tokens(str(content)),
        )


async def _route_gemini_stream(request: ChatCompletionRequest):
    """Streaming Gemini call (via antigravity or gemini CLI) → SSE chunks."""
    if not _gemini_available():
        if _GEMINI_FALLBACK:
            raise EnvironmentError(
                "gemini CLI not found. Install with: npm i -g @google/gemini-cli"
            )
        raise EnvironmentError(
            "antigravity not found. Install Google Antigravity IDE, "
            "or set GEMINI_FALLBACK_CLI=1 to use the headless gemini CLI instead."
        )

    chunk_id = _oi_completion_id()
    prompt = _build_prompt_from_messages(request.messages)

    if not _GEMINI_FALLBACK:
        # ── Antigravity GUI mode (file bridge): stream the response file as it grows ──
        request_id = chunk_id
        since_ts = time.time()
        _write_request_files(request_id, prompt, request.model)
        _proc, stdout_acc = await _launch_antigravity(prompt, request.model)
        try:
            async for chunk in _stream_antigravity_file(request_id, since_ts, stdout_acc,
                                                        request.model, chunk_id):
                yield chunk
        finally:
            for f in _request_files(request_id):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
        return

    # ── Headless fallback: gemini CLI with stream-json over stdout ──
    cmd = _build_gemini_cmd(prompt, request.model, output_format="stream-json")

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
    gemini_info = {
        "available": _gemini_available(),
        "backend": "antigravity" if not _GEMINI_FALLBACK else "gemini-cli-fallback",
        "path": ANTIGRAVITY_CLI if not _GEMINI_FALLBACK else (GEMINI_CLI or f"npx {GEMINI_NPX_PACKAGE}"),
    }
    return {
        "status": "healthy",
        "claude_cli": {
            "available": _claude_available(),
            "path": CLAUDE_CLI,
        },
        "gemini": gemini_info,
        "supported_prefixes": sorted(SUPPORTED_MODEL_PREFIXES),
    }


@app.get("/v1/models")
async def list_models():
    """List models available through this proxy."""
    models = [
        # Gemini models
        ModelInfo(id="gemini-2.5-pro", owned_by="google"),
        ModelInfo(id="gemini-2.5-flash", owned_by="google"),
        ModelInfo(id="gemini-3.5-flash", owned_by="google"),
        ModelInfo(id="gemini-3.1-pro", owned_by="google"),
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
    is_gemini = model_lower.startswith("gemini") | model_lower.startswith("MODEL_PLACEHOLDER_M20") 
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
    gemini_backend = "antigravity" if not _GEMINI_FALLBACK else "gemini CLI (fallback)"
    print(f"[ModelProxy] Starting on port {port}")
    print(f"[ModelProxy] Claude CLI:        {'OK ' + CLAUDE_CLI if _claude_available() else 'NOT FOUND'}")
    print(f"[ModelProxy] Gemini backend:     {'OK ' + (ANTIGRAVITY_CLI if not _GEMINI_FALLBACK else (GEMINI_CLI or 'npx @google/gemini-cli')) if _gemini_available() else 'NOT FOUND'}")
    print(f"[ModelProxy] Gemini mode:        {gemini_backend}")
    print(f"[ModelProxy] Response dir:       {GEMINI_RESPONSE_DIR}")
    print(f"[ModelProxy] Poll interval:      {POLL_INTERVAL}s  (timeout {RESPONSE_TIMEOUT:.0f}s)")
    print(f"[ModelProxy] No API keys needed — uses your account subscriptions via CLI")
    uvicorn.run(
        "model_proxy:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        timeout_keep_alive=600,
    )
