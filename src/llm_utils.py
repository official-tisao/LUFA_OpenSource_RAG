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

__all__ = ["stream_complete", "stream_openai_chat", "streaming_enabled"]


def streaming_enabled() -> bool:
    """Streaming is on unless explicitly disabled via env var."""
    return os.environ.get("LUFA_DISABLE_STREAMING", "").strip().lower() not in ("1", "true", "yes")


def stream_complete(llm, prompt: str, verbose: bool = False) -> str:
    """
    Stream a LlamaIndex LLM completion and return the full accumulated text.

    Falls back to blocking `llm.complete()` when streaming is disabled, the LLM
    has no `stream_complete`, or the stream yields nothing / errors mid-way.
    """
    if not streaming_enabled() or not hasattr(llm, "stream_complete"):
        return str(llm.complete(prompt)).strip()

    parts = []
    last_text = ""
    try:
        for chunk in llm.stream_complete(prompt):
            delta = getattr(chunk, "delta", None)
            if delta:
                parts.append(delta)
                if verbose:
                    print(delta, end="", flush=True)
            else:
                t = getattr(chunk, "text", None)
                if t is not None:
                    last_text = t
        if verbose:
            print()
    except Exception as e:
        # Stream broke — fall back to a single blocking call.
        print(f"      [llm_utils] stream_complete fell back to blocking: {e}")
        return str(llm.complete(prompt)).strip()

    if parts:
        return "".join(parts).strip()
    if last_text:
        return last_text.strip()
    # Stream produced nothing usable — last resort blocking call.
    return str(llm.complete(prompt)).strip()


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
