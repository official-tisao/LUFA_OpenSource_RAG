import time
import httpx
import json

URL = "http://localhost:11434/api/generate"

def benchmark_local_model(model_name: str):
    payload = {
        "model": model_name,
        "prompt": "Write a 300-word fictional story about an autonomous AI script.",
        "stream": True  # Must stream to isolate TTFT
    }

    first_token_time = None
    start_time = time.time()
    token_count = 0
    final_data = {}

    try:
        with httpx.stream("POST", URL, json=payload, timeout=460.0) as response:
            for line in response.iter_lines():
                if not line:
                    continue

                chunk = json.loads(line)

                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = first_token_time - start_time

                if chunk.get("response"):  # Non-empty token
                    token_count += 1

                if chunk.get("done"):
                    final_data = chunk
                    break

        end_time = time.time()

        # Prefer Ollama's own eval_count if available (more accurate than our manual count)
        eval_count = final_data.get("eval_count", token_count)

        total_duration = end_time - start_time
        generation_duration = end_time - first_token_time  # Time AFTER first token

        # TPS measured only over the generation window (excludes prompt eval + TTFT)
        tps_after_ttft = (eval_count - 1) / generation_duration if generation_duration > 0 else 0

        print(f"\n📊 Model : {model_name}")
        print(f"   TTFT  : {ttft:.3f}s")
        print(f"   TPS   : {tps_after_ttft:.2f} tokens/sec  (after TTFT)")
        print(f"   Tokens: {eval_count}")
        print(f"   Total : {total_duration:.2f}s")

    except Exception as e:
        print(f"❌ Failed to benchmark {model_name}: {e}")


print("Running local hardware speed benchmarks...\n")
benchmark_local_model("llama3.2:3b-instruct-q4_K_M")
benchmark_local_model("tensortemplar/prometheus2:8x7b-Q4_K_S")