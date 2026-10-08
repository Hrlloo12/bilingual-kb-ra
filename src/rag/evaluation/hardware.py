from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


def _nvidia_smi(fields: str) -> list[str]:
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def gpu_snapshot() -> dict | None:
    lines = _nvidia_smi("utilization.gpu,memory.used")
    if not lines:
        return None
    try:
        utilization, memory = (float(value) for value in lines[0].split(","))
    except ValueError:
        return None
    return {"utilization_percent": utilization, "memory_used_mib": memory}


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def host_info() -> dict:
    import psutil

    gpus = [dict(zip(("name", "memory_total_mib", "driver"), (part.strip() for part in line.split(",")))) for line in _nvidia_smi("name,memory.total,driver_version")]
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    return {
        "cpu_model": cpu_model(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "usable_cpus": len(affinity) if affinity else psutil.cpu_count(logical=True),
        "ram_total_mib": round(psutil.virtual_memory().total / 2**20),
        "os": platform.platform(),
        "python": platform.python_version(),
        "gpus": gpus,
    }
