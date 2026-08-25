#!/usr/bin/env python3
"""
Shared LLM streaming helpers for the LUFA RAG system.

Long single-shot ("compact") completions on Ollama / cloud models keep the HTTP
connection idle until the WHOLE answer is ready, which trips read-timeouts on big
generations. Streaming reads tokens as they arrive, resetting the read clock on
every chunk, so long answers no longer time out. These helpers accumulate the
streamed deltas and return the full text — a drop-in replacement for
`str(llm.complete(prompt))` and `client.chat.completions.create(...)`.

Set LUFA_DISABLE_STREAMING=1 to force the old blocking path everywhere.
"""

import os
import time

__all__ = ["stream_complete", "stream_complete_timed", "stream_openai_chat",
           "streaming_enabled", "estimate_tokens", "compute_context_window",
           "apply_context_window"]


def streaming_enabled() -> bool:
    """Streaming is on unless explicitly disabled via env var."""
    return os.environ.get("LUFA_DISABLE_STREAMING", "").strip().lower() not in ("1", "true", "yes")


# ── Automatic context-window sizing ──────────────────────────────────────────
# Sizing happens HERE, at the boundary where a request leaves Python, because only
# here is the fully-assembled prompt known (all retrieved chunks + the rewritten
# query + the system prompt). A fixed window either wastes VRAM or silently truncates
# the longest prompts, since clause chunks vary widely in length even at a fixed top_k of 5.
#
# The window must cover the prompt AND the tokens the model still has to emit, so a
# generation reserve is added before rounding up to the next power-of-two bucket.
# Buckets keep Ollama from reloading the model for every slightly different size.
_CTX_BUCKETS = (4096, 8192, 12288, 16384, 24576, 32768)


def _cfg(key, default):
    try:
        from config_loader import cfg
        v = cfg(key, default)
        return type(default)(v) if v is not None else default
    except Exception:
        return default


def estimate_tokens(text: str) -> int:
    """
    Conservative token estimate without loading a tokenizer.

    ~3.2 characters per token deliberately OVER-estimates: English averages ~4, but
    French and German (this corpus is bilingual, plus a German query set) tokenise
    less efficiently, and under-estimating would truncate the prompt.
    """
    if not text:
        return 0
    return int(len(str(text)) / 3.2) + 1


def predict_output_tokens(prompt_tokens: int) -> int:
    """
    Predict how many tokens the model still has to emit for a prompt of this size.

    Scales with the input rather than using a flat reserve: a 7-chunk agentic retry
    warrants more room than a short first-pass query. The ratio follows the usual
    range for non-reasoning instruct models on extractive RAG QA, where the answer is
    a fraction of the retrieved context (~0.15-0.35 of input); 0.35 is chosen as the
    conservative end of that band. Clamped so a tiny prompt still gets usable room and
    a huge one cannot balloon the KV cache.
        ratio  models.llm.output_reserve_ratio   (default 0.35)
        floor  models.llm.output_reserve_min     (default 768)
        cap    models.llm.output_reserve_max     (default 4096)
    """
    ratio = _cfg("models.llm.output_reserve_ratio", 0.35)
    lo = _cfg("models.llm.output_reserve_min", 768)
    hi = _cfg("models.llm.output_reserve_max", 4096)
    return int(max(lo, min(hi, round(prompt_tokens * ratio))))


def compute_context_window(prompt: str, reserve_output: int = None,
                           base: int = None, maximum: int = None) -> tuple:
    """
    Choose num_ctx for one request.
    Returns (context_window, estimated_prompt_tokens, predicted_output_tokens).

    base     floor for ordinary requests (config: models.llm.context_window)
    reserve  predicted generation headroom; defaults to predict_output_tokens()
    maximum  ceiling (config: models.llm.max_context_window). Exceeding the 6 GB card
             is allowed — Windows spills into the 24 GB shared pool — but the spill is
             recorded via gpu_vram_shared_mb so its cost stays visible in the results.
    """
    base = base if base is not None else _cfg("models.llm.context_window", 12288)
    maximum = maximum if maximum is not None else _cfg("models.llm.max_context_window", 24576)

    prompt_tokens = estimate_tokens(prompt)
    reserve_output = (reserve_output if reserve_output is not None
                      else predict_output_tokens(prompt_tokens))

    needed = prompt_tokens + reserve_output
    if needed <= base:
        return base, prompt_tokens, reserve_output
    for b in _CTX_BUCKETS:
        if needed <= b <= maximum:
            return b, prompt_tokens, reserve_output
    return maximum, prompt_tokens, reserve_output


def apply_context_window(llm, prompt: str):
    """
    Size and pin the context window on a LlamaIndex Ollama client for THIS request,
    and record what was used on the client so callers can log it per query.

    Returns (context_window, prompt_tokens); both "" for non-Ollama clients, whose
    context is managed server-side.
    """
    if not hasattr(llm, "context_window"):
        return "", ""
    try:
        ctx, ptok, otok = compute_context_window(prompt)
        llm.context_window = ctx
        llm._last_context_window = ctx
        llm._last_prompt_tokens = ptok
        llm._last_predicted_output_tokens = otok
        return ctx, ptok
    except Exception:
        return "", ""


# ── On per-query KV-cache isolation ──────────────────────────────────────────
# No cache-clearing call is issued between queries, and none is needed: each request
# is already fully independent. Verified against the installed client — its
# stream_complete payload carries no `context`, no `history` and no message list, only
# the single assembled prompt, so one query's tokens can never enter another query's
# attention. The KV cache is rebuilt from that prompt every time.
#
# The one thing Ollama does reuse across calls is the cached *prefix* two consecutive
# prompts happen to share (here only the ~150-token system prompt). That changes
# prefill speed very slightly, never the output. Eliminating even that would require
# evicting the model (keep_alive: 0), which would cold-start a 3-6 GB reload on every
# single query — no benefit, hours of added runtime, and it would defeat the warm-up
# protocol. So the model stays resident and each query still gets its own clean cache.


def stream_complete_timed(llm, prompt: str, verbose: bool = False):
    """
    Like `stream_complete` but also returns the time-to-first-token (TTFT).

    Returns (text, ttft_seconds). `ttft_seconds` is the wall-clock (via
    time.perf_counter) from the call start to the first non-empty streamed token,
    rounded to 4 dp — or "" when it cannot be measured (streaming disabled, no
    `stream_complete`, or a blocking fallback was taken). Never raises: any stream
    error falls back to a single blocking completion with a blank TTFT.
    """
    # Size the context window for THIS prompt before the request leaves Python.
    apply_context_window(llm, prompt)

    start = time.perf_counter()

    if not streaming_enabled() or not hasattr(llm, "stream_complete"):
        return str(llm.complete(prompt)).strip(), ""

    parts = []
    last_text = ""
    ttft = ""
    try:
        for chunk in llm.stream_complete(prompt):
            delta = getattr(chunk, "delta", None)
            if delta:
                if ttft == "":
                    ttft = round(time.perf_counter() - start, 4)
                parts.append(delta)
                if verbose:
                    print(delta, end="", flush=True)
            else:
                t = getattr(chunk, "text", None)
                if t is not None:
                    if ttft == "":
                        ttft = round(time.perf_counter() - start, 4)
                    last_text = t
        if verbose:
            print()
    except Exception as e:
        # Stream broke — fall back to a single blocking call (no measurable TTFT).
        print(f"      [llm_utils] stream_complete fell back to blocking: {e}")
        return str(llm.complete(prompt)).strip(), ""

    if parts:
        return "".join(parts).strip(), ttft
    if last_text:
        return last_text.strip(), ttft
    # Stream produced nothing usable — last resort blocking call.
    return str(llm.complete(prompt)).strip(), ""


def stream_complete(llm, prompt: str, verbose: bool = False) -> str:
    """
    Stream a LlamaIndex LLM completion and return the full accumulated text.

    Thin wrapper over `stream_complete_timed` (discards the TTFT). Falls back to
    blocking `llm.complete()` when streaming is disabled, the LLM has no
    `stream_complete`, or the stream yields nothing / errors mid-way.
    """
    text, _ = stream_complete_timed(llm, prompt, verbose=verbose)
    return text


def stream_openai_chat(client, model: str, messages, verbose: bool = False, **kwargs) -> str:
    """
    Stream an OpenAI-compatible chat completion and return the full text.

    Falls back to a non-streaming request when streaming is disabled or errors.
    Extra kwargs (temperature, max_tokens, timeout, ...) pass through unchanged.
    """
    kwargs.pop("stream", None)

    if not streaming_enabled():
        resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
        return (resp.choices[0].message.content or "").strip()

    parts = []
    try:
        stream = client.chat.completions.create(
            model=model, messages=messages, stream=True, **kwargs
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
            except (AttributeError, IndexError, TypeError):
                delta = None
            if delta:
                parts.append(delta)
                if verbose:
                    print(delta, end="", flush=True)
        if verbose:
            print()
    except Exception as e:
        print(f"      [llm_utils] stream_openai_chat fell back to blocking: {e}")
        resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
        return (resp.choices[0].message.content or "").strip()

    if parts:
        return "".join(parts).strip()
    # Nothing streamed — fall back once.
    resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return (resp.choices[0].message.content or "").strip()
