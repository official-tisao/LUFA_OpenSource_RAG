#!/usr/bin/env python3
"""
system_metrics.py — runtime hardware probes for the LUFA RAG simulation
(Ch4 §4.6.4 hardware utilisation).

Provides point probes (GPU VRAM, GPU utilisation, process RAM, CPU load) and a
`ResourceSampler` that polls them on a background thread for the duration of a
query, so the recorded figures describe the whole generation rather than a single
instant after it finished.

Every probe is BEST-EFFORT: it returns "" (blank) on any failure or when the tool
is unavailable, so hosts without an NVIDIA GPU or without psutil simply record
empty cells instead of crashing. Per Ch4, hardware metrics are meaningful only for
the local systems A/B/C — `sample_system_metrics` blanks them for cloud/API modes.

Why both CPU and GPU are recorded: Ollama silently splits a model across CPU and
GPU when it does not fit in VRAM. A model that appears "local GPU" can in fact run
mostly on the CPU (observed: 77%/23% CPU/GPU for Llama 3.2 3B once its 131k-token
context inflated the KV cache to ~18 GB). Logging both utilisations makes that
visible in the data instead of silently distorting the latency results.
"""

import os
import shutil
import subprocess
import threading
import time

__all__ = [
    "get_gpu_vram_mb", "get_gpu_util_percent", "get_gpu_stats",
    "get_ram_mb", "get_cpu_percent",
    "is_local_mode", "sample_system_metrics", "ResourceSampler",
]

# Simulation modes that run the model on local hardware (systems A/B/C).
LOCAL_MODES = {"local", "local-naive"}

_NVIDIA_SMI = shutil.which("nvidia-smi")


def is_local_mode(mode) -> bool:
    """True for locally-hosted generation modes (VRAM/CPU/GPU load matter only here)."""
    return str(mode).strip().lower() in LOCAL_MODES


def get_gpu_stats():
    """
    One nvidia-smi call returning (used_vram_mb, gpu_util_percent).
    Either element is "" when unavailable. Multi-GPU hosts are summed (VRAM) and
    averaged (utilisation).
    """
    if not _NVIDIA_SMI:
        return "", ""
    try:
        out = subprocess.run(
            [_NVIDIA_SMI, "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return "", ""
        mems, utils = [], []
        for line in out.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    mems.append(float(parts[0]))
                    utils.append(float(parts[1]))
                except ValueError:
                    continue
        if not mems:
            return "", ""
        return round(sum(mems), 1), round(sum(utils) / len(utils), 1)
    except Exception:
        return "", ""


def get_gpu_memory_split(timeout=15):
    """
    (dedicated_mb, shared_mb) for the GPU adapter, via Windows performance counters.

    nvidia-smi reports only DEDICATED VRAM. When a model exceeds the 6 GB card,
    Windows WDDM spills into "shared" GPU memory — system RAM addressed as VRAM
    (this machine dedicates 24 GB to that). That spill is invisible to nvidia-smi
    while being the main reason a model slows down, so it is captured separately.

    Costs ~1-1.8 s per call, so it is measured ONCE per query OUTSIDE the timed
    region — never inside the sampling loop, where it would distort the latency it
    is meant to describe. Returns ("", "") on non-Windows or on any failure.
    """
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        return "", ""
    script = (
        "$c=Get-Counter -Counter '\\GPU Adapter Memory(*)\\Dedicated Usage',"
        "'\\GPU Adapter Memory(*)\\Shared Usage' -ErrorAction SilentlyContinue;"
        "$d=($c.CounterSamples|Where-Object {$_.Path -like '*dedicated*'}|"
        "Measure-Object CookedValue -Sum).Sum;"
        "$s=($c.CounterSamples|Where-Object {$_.Path -like '*shared*'}|"
        "Measure-Object CookedValue -Sum).Sum;"
        "Write-Output ([math]::Round($d/1MB,1).ToString()+','+[math]::Round($s/1MB,1).ToString())"
    )
    try:
        out = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", script],
                             capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return "", ""
        parts = out.stdout.strip().splitlines()[-1].split(",")
        if len(parts) != 2:
            return "", ""
        return float(parts[0]), float(parts[1])
    except Exception:
        return "", ""


def get_gpu_vram_mb():
    """Used GPU memory in MB, or "" when unavailable."""
    return get_gpu_stats()[0]


def get_gpu_util_percent():
    """GPU utilisation percentage, or "" when unavailable."""
    return get_gpu_stats()[1]


def get_ram_mb():
    """Resident set size of the current (Python) process in MB. "" if psutil missing."""
    try:
        import psutil
    except Exception:
        return ""
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return ""


def get_cpu_percent(interval=None):
    """
    System-wide CPU utilisation percentage. With interval=None this returns the load
    since the previous call, so the sampler primes it once before collecting.
    """
    try:
        import psutil
    except Exception:
        return ""
    try:
        return round(psutil.cpu_percent(interval=interval), 1)
    except Exception:
        return ""


# ── Ollama-scoped probes ─────────────────────────────────────────────────────
# The model actually executes in `llama-server.exe`, a CHILD of the Ollama service —
# not in `ollama.exe`. Matching only on the name "ollama" misses the process doing all
# the work, so match the executable path as well.
_OLLAMA_PROC_NAMES = ("ollama.exe", "ollama", "ollama app.exe", "llama-server.exe", "llama-server")


def _ollama_processes():
    try:
        import psutil
    except Exception:
        return []
    procs = []
    for p in psutil.process_iter(["pid", "name", "exe"]):
        try:
            name = (p.info.get("name") or "").lower()
            exe = (p.info.get("exe") or "").lower()
            if name in _OLLAMA_PROC_NAMES or "ollama" in exe or "llama-server" in name:
                procs.append(p)
        except Exception:
            continue
    return procs


def get_ollama_cpu_percent(procs=None, normalize=True):
    """
    CPU utilisation of the Ollama inference processes ONLY (not the whole machine).

    psutil reports a process's CPU as a percentage of ONE core, so the raw sum can
    exceed 100 on a multi-core box. With normalize=True the value is divided by the
    core count, giving a 0-100 figure directly comparable to Task Manager.
    Returns "" when psutil is unavailable or no Ollama process is running.
    """
    try:
        import psutil
    except Exception:
        return ""
    procs = procs if procs is not None else _ollama_processes()
    if not procs:
        return ""
    total = 0.0
    seen = False
    for p in procs:
        try:
            total += p.cpu_percent(interval=None)
            seen = True
        except Exception:
            continue
    if not seen:
        return ""
    if normalize:
        cores = psutil.cpu_count(logical=True) or 1
        total /= cores
    return round(total, 1)


def get_ollama_ram_mb(procs=None):
    """Total resident memory of the Ollama inference processes in MB."""
    procs = procs if procs is not None else _ollama_processes()
    if not procs:
        return ""
    total = 0.0
    seen = False
    for p in procs:
        try:
            total += p.memory_info().rss
            seen = True
        except Exception:
            continue
    return round(total / (1024 * 1024), 1) if seen else ""


class ResourceSampler:
    """
    Poll CPU / GPU utilisation and memory on a background thread while a query runs.

    Usage:
        with ResourceSampler(enabled=is_local_mode(mode)) as rs:
            ...run the query...
        stats = rs.results        # dict of the 4 recorded columns

    Reported values (scoped to the Ollama inference processes where the hardware
    allows it):
        cpu_percent      mean CPU utilisation of the OLLAMA processes only
                         (llama-server.exe + ollama.exe), normalised to 0-100 across
                         all cores so it matches Task Manager
        system_ram_mb    peak resident memory of those same Ollama processes
        gpu_util_percent mean GPU utilisation, CARD-WIDE. Per-process GPU metrics are
                         not obtainable on this hardware: nvidia-smi returns
                         "[Insufficient Permissions]" for per-process VRAM on a
                         consumer GTX under WDDM and `pmon` reports "Not supported on
                         the device". Ollama is the only significant compute client
                         during a run, so the card-wide figure is a close proxy — but
                         it is card-wide, and should be described as such.
        gpu_vram_mb      PEAK GPU memory observed, also card-wide (the meaningful
                         figure against the 6 GB budget; a post-hoc reading misses
                         the peak). Idle desktop baseline is ~150-320 MB.

    Disabled (cloud/API modes) or on any failure, every field is "".
    """

    def __init__(self, interval: float = 2.0, enabled: bool = True,
                 measure_split: bool = True):
        self.interval = max(0.25, float(interval))
        self.enabled = bool(enabled)
        # The dedicated/shared split costs ~1-1.8s, so it is taken once on stop(),
        # after the timed region — never inside the loop.
        self.measure_split = bool(measure_split)
        self._stop = threading.Event()
        self._thread = None
        self._cpu, self._gpu, self._vram, self._ram = [], [], [], []
        self._split = ("", "")

    def _loop(self):
        # `llama-server.exe` (the process that actually runs the model) is spawned only
        # when a model loads, so it can appear AFTER sampling starts. Rediscover every
        # tick, but keep the existing psutil.Process objects in a pid-keyed map:
        # constructing a new Process resets its CPU delta baseline and would make
        # cpu_percent() return 0.0 forever.
        tracked = {}

        def _refresh():
            for p in _ollama_processes():
                if p.pid not in tracked:
                    tracked[p.pid] = p
                    try:
                        p.cpu_percent(interval=None)   # prime the new process
                    except Exception:
                        pass
            for pid in [pid for pid, p in tracked.items() if not p.is_running()]:
                tracked.pop(pid, None)
            return list(tracked.values())

        _refresh()
        while not self._stop.is_set():
            procs = _refresh()
            c = get_ollama_cpu_percent(procs)
            r = get_ollama_ram_mb(procs)
            v, g = get_gpu_stats()
            if c != "":
                self._cpu.append(c)
            if g != "":
                self._gpu.append(g)
            if v != "":
                self._vram.append(v)
            if r != "":
                self._ram.append(r)
            self._stop.wait(self.interval)

    def start(self):
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None
        if self.enabled and self.measure_split:
            self._split = get_gpu_memory_split()
        return self.results

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    @property
    def results(self) -> dict:
        def _mean(xs):
            return round(sum(xs) / len(xs), 1) if xs else ""

        def _peak(xs):
            return round(max(xs), 1) if xs else ""

        if not self.enabled:
            return {"cpu_percent": "", "gpu_util_percent": "",
                    "gpu_vram_mb": "", "system_ram_mb": "",
                    "gpu_vram_dedicated_mb": "", "gpu_vram_shared_mb": ""}
        out = {
            "cpu_percent": _mean(self._cpu),
            "gpu_util_percent": _mean(self._gpu),
            "gpu_vram_mb": _peak(self._vram),
            "system_ram_mb": _peak(self._ram),
            "gpu_vram_dedicated_mb": self._split[0],
            "gpu_vram_shared_mb": self._split[1],
        }
        # Very short queries can finish before the first tick — fall back to one
        # immediate reading so the row is not left blank.
        if out["gpu_vram_mb"] == "" and out["cpu_percent"] == "":
            procs = _ollama_processes()
            get_ollama_cpu_percent(procs)
            time.sleep(0.15)
            v, g = get_gpu_stats()
            out.update({"cpu_percent": get_ollama_cpu_percent(procs),
                        "gpu_util_percent": g, "gpu_vram_mb": v,
                        "system_ram_mb": get_ollama_ram_mb(procs)})
        return out


def sample_system_metrics(mode) -> dict:
    """
    Point-in-time reading of all four hardware columns for local modes; blanks for
    cloud/API (Ch4 §4.6.4 — hardware metrics apply only to local systems A/B/C).
    Prefer ResourceSampler when a duration is available.
    """
    if not is_local_mode(mode):
        return {"gpu_vram_mb": "", "system_ram_mb": "",
                "cpu_percent": "", "gpu_util_percent": ""}
    procs = _ollama_processes()
    get_ollama_cpu_percent(procs)      # prime
    time.sleep(0.15)
    vram, util = get_gpu_stats()
    return {"gpu_vram_mb": vram, "system_ram_mb": get_ollama_ram_mb(procs),
            "cpu_percent": get_ollama_cpu_percent(procs), "gpu_util_percent": util}


if __name__ == "__main__":
    print("nvidia-smi:", bool(_NVIDIA_SMI))
    procs = _ollama_processes()
    print("ollama processes:", [(p.pid, p.info.get("name")) for p in procs])
    print("point sample (local)   :", sample_system_metrics("local"))
    print("point sample (frontier):", sample_system_metrics("frontier"))
    with ResourceSampler(interval=0.5) as rs:
        time.sleep(2.2)
    print("sampled over 2.2s      :", rs.results)
