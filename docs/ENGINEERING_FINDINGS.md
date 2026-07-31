# Engineering Findings & Measurement Notes

Hard-won discoveries made while instrumenting the LUFA RAG pipeline for the Chapter-4
evaluation metrics. Several of these silently corrupt results if you do not know about
them, so they are recorded here in full rather than left in commit messages.

Hardware referenced throughout: **NVIDIA GTX 1660 Ti, 6 GB VRAM** (plus 24 GB of system
RAM configured as shared/secondary VRAM), Windows 11, Ollama, ChromaDB + Nomic Embed v2.

---

## 1. Ollama silently ran the models on the CPU (~8× slowdown)

**Symptom.** CPU sat at 71-75% during generation; GPU looked idle.

**Diagnosis.** `ollama ps` is the tool that reveals this, it prints the device split:

```
NAME                          SIZE     PROCESSOR          CONTEXT
llama3.2:3b-instruct-q4_K_M   18 GB    77%/23% CPU/GPU    131072
```

The model weights are only ~2 GB, but **SIZE showed 18 GB**. The cause is the
**context window**, not the layer count:

- Llama 3.x advertises a **131 072-token** context.
- LlamaIndex's `Ollama` client defaults `context_window = -1`, which makes it send
  `num_ctx = <model maximum>`.
- A 131 072-token KV cache is ~18 GB. It cannot fit in 6 GB, so Ollama offloads most
  layers to the CPU, while still reporting the model as "loaded".

**Fix (two parts, both are required).**

1. Pin the context client-side. In `config/config.yaml`:
   ```yaml
   models:
     llm:
       context_window: 12288
   ```
   applied in `src/model_api_auth.py::get_ollama_client`.
2. Force full GPU offload in a Modelfile (`modelfiles/`):
   ```
   FROM llama3.2:3b-instruct-q4_K_M
   PARAMETER num_gpu 33     # above the layer count; Ollama caps it at "all"
   PARAMETER num_ctx 8192
   ```
   ```sh
   ollama create llama3.2:3b-gpu -f modelfiles/Modelfile.llama3.2-3b-gpu
   ```

**Layer counts** (for `num_gpu`; anything ≥ the count means "all layers"):
Llama 3.2 3B = **28**, Llama 3.1 8B = **32**, Mistral 7B = **32**.

**Measured effect** (identical questions, before → after):

| Metric | CPU-bound | 100% GPU | Change |
|---|---|---|---|
| End-to-end latency | 82.3 s mean | **~10 s** | **~8× faster** |
| TTFT | 17.9 s | 2.7-4.9 s | ~4× faster |
| Ollama CPU | (system 71-75%) | **6.6%** |, |
| GPU utilisation | ~0-20% | **63-79%** |, |

**Thesis consequence.** On the CPU-bound configuration the system missed Chapter 4's
"end-to-end < 60 s in 75% of queries" target. Correctly configured it passes with a wide
margin. Publishing the misconfigured numbers would have badly understated the system.

### `num_gpu` alone is not enough
`mistral:7b` with no `num_gpu` ran at **17%/83% CPU/GPU** even at a 4096 context -
Ollama under-commits when it is unsure. An explicit `num_gpu` fixed it.

### VRAM fit at various context sizes (all 100% GPU)

| Model | 8192 ctx | 12288 ctx |
|---|---|---|
| llama3.2:3b-gpu | 3.1 GB | 3.5 GB |
| mistral:7b-gpu | 5.5 GB | 6.0 GB |
| llama3.1:8b-gpu | 5.8 GB | 6.4 GB* |

\* Ollama's reported SIZE is its own estimate and **overstates** actual allocation -
`nvidia-smi` showed **5 827 MiB of 6 144** for the 8B at 12 288. Trust `nvidia-smi`, and
cross-check with the Windows perf counter (below).

---

## 2. Per-process GPU metrics are impossible on this hardware

Attempting to scope GPU usage to the Ollama process fails on a consumer GTX under WDDM:

```
$ nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
1852, [Insufficient Permissions], [N/A]
22884, ...\ollama\llama-server.exe, [N/A]

$ nvidia-smi pmon -c 1
The feature is not supported in this configuration
```

**Consequence.** GPU figures are necessarily **card-wide**; CPU and RAM *can* be scoped to
Ollama. Report them as such, do not claim per-process GPU attribution. Ollama is the only
significant compute client during a run (idle desktop baseline ≈ 150-320 MB), so card-wide
is a close proxy.

### The inference process is not called "ollama"
The model runs in **`llama-server.exe`**, a child of the Ollama service, matching only on
the name `ollama` misses the process doing all the work. Match the executable **path**
containing `ollama`, or the name `llama-server`. See `src/system_metrics.py`.

### psutil CPU sampling gotchas
- `Process.cpu_percent()` returns 0.0 on its **first** call, it needs a priming call to
  establish a delta baseline.
- It reports a percentage of **one core**; divide by `psutil.cpu_count()` for a 0-100
  figure comparable to Task Manager.
- `llama-server.exe` appears only once a model loads, so the sampler must rediscover
  processes periodically, while **keeping existing `Process` objects**, since recreating
  them resets the CPU baseline to 0.

---

## 3. Dedicated vs shared VRAM (the 24 GB secondary pool)

`nvidia-smi` reports **dedicated** VRAM only. When a model exceeds the 6 GB card, Windows
WDDM spills into **shared** GPU memory (system RAM addressed as VRAM), invisible to
`nvidia-smi`, but the main reason a run slows down. Read it via performance counters:

```powershell
Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage'
```

**Cost: ~1.0-1.8 s per call** (≈4.2 s for two separate calls). Far too slow to sample in a
loop, so it is measured **once per query, outside the timed region**: sampling it inside
would distort the very latency it describes.

Recorded as `gpu_vram_dedicated_mb` / `gpu_vram_shared_mb`. The perf counter and
`nvidia-smi` agree closely (4291 MB vs 4286 MB), which makes a useful cross-check.

---

## 4. Automatic per-request context sizing

A fixed context window either wastes VRAM or truncates the longest agentic retries
(`top_k` grows to 7 chunks on the third attempt). `src/llm_utils.py` sizes it per request,
at the boundary where the call leaves Python, the only place the fully assembled prompt
(all chunks + rewritten query + system prompt) is known.

- **Base** 12288, widened by bucket (4k/8k/12k/16k/24k/32k) when
  `prompt + predicted_output` exceeds it, capped at 24576.
- **Output prediction scales with input**: `ratio 0.35`, clamped to 768-4096. For
  non-reasoning instruct models on extractive RAG QA, output is typically 0.15-0.35 of
  input; 0.35 is the conservative end.
- **Token estimate** uses ~3.2 chars/token, deliberately pessimistic, because French and
  German tokenise less efficiently than English and under-estimating truncates.

Recorded per query: `context_window_used`, `prompt_tokens_est`, `predicted_output_tokens`.
Observed in practice: real prompts are ~900 tokens (chunks are well under the 512-token
cap), so requests comfortably stay in the 12288 base bucket.

---

## 5. KV cache is already isolated per query: do not unload the model

Concern: does one query's cache contaminate the next? **No.** Verified by inspecting the
installed client: `stream_complete`'s payload contains no `context`, no `history` and no
message list, only the single assembled prompt. Ollama's `/api/generate` is stateless, so
the KV cache is rebuilt per request and one query's tokens can never enter another's
attention.

The only cross-request reuse is the **shared prompt prefix** (here just the ~150-token
system prompt), which affects prefill speed slightly and never the output.

**Do not "clear" it by unloading the model** (`keep_alive: 0`): that evicts 3-6 GB and
forces a cold reload on every query, hours of added runtime, no benefit, and it destroys
the warm-up protocol.

---

## 6. Cold start is real: warm up before timing

| | Time |
|---|---|
| Retrieval, cold (first call) | **7.92 s** |
| Retrieval, warm | **2.74 s** |

A **2.9×** difference from index load, BM25 corpus construction and the first embedding
call. The pipeline therefore runs the first question's retrieval twice and records only
the warm pass (`warmup_applied = 1` marks that row). Without this the first query of every
batch is a significant outlier.

---

## 7. Data-integrity bugs found while instrumenting

Each of these silently destroys or corrupts data:

1. **`retrieval.py --force` reused the old-schema migration path**, which `unlink()`s the
   CSV and carries only 5 columns forward, it would have wiped translation and telemetry
   columns. Separated "re-retrieve" (`force_all`) from "migrate old schema"
   (`migrate_mode`).
2. **`answer_generator.py` blanked source columns on error rows.** The fallback read
   `row.get("source{i}_...")` from the *input questions* CSV, which has no source columns,
   so `upsert_row` overwrote good retrieval with empty strings. Now falls back to the
   cached lufa row.
3. **Batch 2 overwrote batch 1's telemetry with blanks.** The first pass reuses cached
   retrieval, so no retrieval happens and `retrieval_latency_s` came back empty, clobbering
   the measured value. Blank telemetry keys are now dropped before `upsert_row`, so
   previously measured values survive.
4. **`attempts` was 0 when the agentic loop exhausted its retries.** `real_attempt` was
   only assigned on a grounded break, and `metrics.py` then "repaired" 0 → 1, understating
   the loop's true cost in the mean-attempts statistic. Now assigned every iteration.
5. **`metrics.py` read `grounded`/`attempts` from the previous eval row**, not the freshly
   regenerated lufa row, carrying stale values forward after a regeneration.
6. **Stale judge scores.** After regenerating answers, stored judge scores describe the
   *old* answers. `--force_judge` re-judges instead of preserving them.

**Schema safety.** `csv_utils.migrate_csv_schema` backs up to `.bak` and carries columns
**by name**, so new columns can be added anywhere without shifting existing values.
Verified on a real 426-row file: 55 → 73 columns, every value intact, new columns blank.

---

## 8. Retrieval is deterministic and shared: measure it once

`_retrieve_nodes` is deterministic given the same question (same ChromaDB, same embedding
model, same BM25 corpus, same RRF fusion), and the three monolingual directories ask the
**same 426 questions**. Re-running retrieval per directory produces byte-identical source
columns and merely re-measures the same retriever.

`src/reuse_retrieval.py` copies the retrieval columns from one measured run into the
others. This is faster *and* more internally consistent, and it matches the thesis argument
that all systems inherit one shared retrieval ceiling. The German set is excluded, it asks
different questions in a different language.

---

## 9. Cross-lingual evaluation without a translation bridge

The original German pipeline translated the query to English, retrieved in English, and
translated the answer back, which tests the *translator*, not cross-lingual retrieval.

The no-translation mode (`--no_translate`) retrieves with the **raw German query** against
the multilingual index and answers **in German**, which is the real test of H2.

Two traps:

- **`_generate_from_nodes` told every non-French language "Respond in English."** German
  answers would silently have come back in English. The instruction is now language-aware
  and additionally tells the model to keep article/clause numbers verbatim.
- **The benchmark's `expected_answer` and `ground_source_truth` are English**, so lexical
  metrics (Token-F1/BLEU/ROUGE/METEOR) against a German answer would score near zero -
  measuring translation, not quality. A **post-hoc** English rendering is stored
  (`answer_metrics_translation`) and used *only* for lexical metrics; the judge always sees
  the native-language answer. The rendering never re-enters the pipeline.

The citation regex also had to learn `Artikel`/`Absatz`/`Ziffer`, verified
`"Gemäß Artikel 7, Absatz 7.20.1"` → 1.0, article-only → 0.5.

---

## 10. Judge model performance

`tensortemplar/prometheus2:8x7b-Q4_K_S` is a **26 GB** model on a 6 GB card, so it is
heavily CPU-offloaded, but its cost is dominated by prompt length, not model size:

| Judge prompt | Prompt chars | Time |
|---|---|---|
| answer_relevance | 1 035 | 8.6 s |
| faithfulness | 3 521 | 21.7 s |
| context_precision | 2 784 | 21.0 s |
| **Total (3 separate prompts)** | | **51.3 s/row** |

First call pays a **40.4 s** model load; subsequent short calls are **0.7 s**. Budget the
load once per batch, not per row.

Separate prompts cost ~3× a single combined prompt but avoid asking one prompt to reason
about three criteria at once.

---

## Quick diagnostic commands

```sh
# THE command for CPU/GPU split: check this whenever performance looks wrong
ollama ps

# Real VRAM (trust this over Ollama's reported SIZE)
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

# Shared (secondary) VRAM, invisible to nvidia-smi
powershell -Command "Get-Counter '\GPU Adapter Memory(*)\Shared Usage'"

# Confirm a model's baked-in parameters
ollama show --modelfile llama3.1:8b-gpu | grep PARAMETER
```

---

## Data provenance anomaly: 1 row received extra generation attempts

**Row:** `test_en_019` in `tests/llama-3.1-8b/Judge-Prometheus-8x7b-v2.0/lufa_out_data.csv`

During the repair pass for the 7 timeout (`ERROR`) rows, `answer_generator.py` was invoked
against the FULL question set. Its resume gate treats a row as incomplete when

```python
answer not in ("", "ERROR") AND (mode != "local" OR max_retries < 2 OR grounded == "true")
```

so with `--mode local --max_retries 3` every **ungrounded** row is also eligible, not just the
`ERROR` ones, 42 rows (7 ERROR + 35 ungrounded), rather than the intended 7. The run was
stopped after one row had been rewritten.

**Effect:** `test_en_019` was ungrounded-but-valid after the main run and received a second
round of up to 3 attempts, which produced a grounded answer. Its `attempts` column records
only the second run, so the row now looks like an ordinary first-attempt success. Every other
row reflects at most 3 attempts. Grounded count for this directory went 384 → 385 of 426
(90.1% → 90.4%), a 0.23 pp shift. Its latency/telemetry are from the second run and are valid
measurements in themselves.

**Not recoverable:** the previous answer text was never persisted (the log records only the
character count), so the pre-repair state of this row cannot be restored.

**DECISION (user, 2026-07-29): do NOT re-run the other ungrounded rows.** Giving every
ungrounded row a second corrective pass would amount to running the loop at max_retries=6 for
failures only, changing the protocol Chapter 4 describes. The stated protocol therefore remains
**max_retries=3 for all 426 rows**, and this single row is reported as a footnote instead.

Footnote wording for the thesis: "One query (test_en_019) in the Llama 3.1 8B condition was
inadvertently regenerated during a repair pass and so received up to three additional
corrective attempts; it reached a grounded verdict on that second pass. The affected figure is
the grounded rate for this system (385 rather than 384 of 426, a 0.23 pp difference). All other
queries in all conditions were run with at most three attempts."

Note the 7 timeout rows are NOT part of this anomaly: they produced no answer at all in the
main run (the request aborted), so repairing them supplies their first completed generation
rather than extra attempts.

**Correct procedure for targeted repairs** (used for the 7 real failures): build an input CSV
containing only the intended `question_id`s and pass it as `--input`, so the loop cannot reach
any other row.

## Config: documented `LUFA_` env override never worked

`config_loader._env_override` derived the variable name as `dotted_key.upper().replace(".","_")`
with **no prefix**, while its own docstring and CLAUDE.md both document
`LUFA_MODELS_LLM_NAME`-style names. Every documented override silently did nothing and fell
back to the YAML value. Fixed to try `LUFA_<KEY>` first and then the bare `<KEY>` for backward
compatibility. Also made `models.llm.request_timeout` actually reach the client, `rag_engine`
had it hardcoded to `240.0`, ignoring config entirely.

## `grounded` was silently redefined by the judge threshold (REMOVED)

`metrics.py` contained:

```python
if str(new_grounded).lower() == "false":
    new_grounded = "true" if float(cp) > 0.4 else "false"
```

so any row the reflector marked ungrounded was flipped to grounded whenever the judge's
`context_precision` exceeded 0.4. The column therefore meant "reflector verdict" for some rows
and "context_precision > 0.4" for others, and could only ever move in one direction, upward.

Measured inflation in `evaluation_results.csv` versus the reflector verdict in
`lufa_out_data.csv`:

| directory | reflector | eval CSV | inflated |
|---|---|---|---|
| llama-3.2-3b | 347 (81.5%) | 422 (99.1%) | +75 |
| llama-3.1-8b | 391 | 408 | +17 |
| cross-lingual-german | 19 | 19 | 0 |
| mistral-7b | 425 | 425 | 0 |

Removed at the user's request; `grounded` is now carried through from the lufa row unchanged.
All four directories were rebuilt with `metrics.py --no_judge --force_det` so the stored values
are the measured reflector verdicts. Any groundedness figure quoted before this fix (notably the
llama-3.2-3b "99.1%") is wrong and must not be reused.
