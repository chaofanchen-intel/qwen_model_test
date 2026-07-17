"""Background hardware utilization monitors.

CpuMonitor reads /proc/stat. GpuMonitor shells out to xpu-smi (sudo required for
the engine_util JSON layout). NpuMonitor reads Intel VPU sysfs via /sys/bus/pci
or sudo cat (read access on PMT GUID requires root).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import threading
import time
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover - fallback is for lean envs
    psutil = None


# =============================================================================
# CPU
# =============================================================================

class CpuMonitor:
    """Per-core CPU utilization monitor via /proc/stat."""

    def __init__(self, interval_ms: int = 100):
        self._interval = interval_ms / 1000.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._prev_times: dict[str, list[int]] = {}
        self._samples: list[dict[str, float]] = []
        self._aggregate_sum = 0.0
        self._aggregate_count = 0

    def start(self):
        self._running = True
        self._prev_times = self._read_proc_stat()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def reset(self):
        with self._lock:
            self._samples.clear()
            self._aggregate_sum = 0.0
            self._aggregate_count = 0

    def get_summary(self) -> dict:
        with self._lock:
            if not self._samples:
                return {"avg_total": 0.0, "per_core": {}}
            avg_total = self._aggregate_sum / self._aggregate_count if self._aggregate_count else 0.0
            per_core_acc: dict[str, list[float]] = {}
            for sample in self._samples:
                for core, util in sample.items():
                    per_core_acc.setdefault(core, []).append(util)
            per_core_result = {k: statistics.fmean(v) for k, v in per_core_acc.items()}
            return {"avg_total": avg_total, "per_core": per_core_result}

    def _loop(self):
        while self._running:
            time.sleep(self._interval)
            cur = self._read_proc_stat()
            sample = {}
            total_util = 0.0
            core_count = 0
            for key, cur_vals in cur.items():
                if key not in self._prev_times:
                    continue
                prev_vals = self._prev_times[key]
                prev_idle = prev_vals[3] + prev_vals[4]
                cur_idle = cur_vals[3] + cur_vals[4]
                prev_total = sum(prev_vals)
                cur_total = sum(cur_vals)
                delta_total = cur_total - prev_total
                delta_idle = cur_idle - prev_idle
                if delta_total > 0:
                    util = max(0.0, min(100.0, 100.0 * (1.0 - delta_idle / delta_total)))
                else:
                    util = 0.0
                if key != "cpu":
                    sample[key] = util
                    total_util += util
                    core_count += 1
            self._prev_times = cur
            if core_count > 0:
                with self._lock:
                    self._samples.append(sample)
                    self._aggregate_sum += total_util / core_count
                    self._aggregate_count += 1

    @staticmethod
    def _read_proc_stat() -> dict[str, list[int]]:
        result = {}
        try:
            with open("/proc/stat", "r") as f:
                for line in f:
                    if not line.startswith("cpu"):
                        break
                    parts = line.split()
                    label = parts[0]
                    values = [int(x) for x in parts[1:8]]
                    result[label] = values
        except (OSError, ValueError):
            pass
        return result


# =============================================================================
# GPU
# =============================================================================

class GpuMonitor:
    """Intel GPU utilization monitor via xpu-smi."""

    def __init__(self, interval_ms: int = 500, command: Optional[str] = None):
        self._interval = interval_ms / 1000.0
        self._command = command or os.environ.get(
            "XPU_SMI_CMD", "sudo xpu-smi stats -d 1 -j"
        )
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._sum = 0.0
        self._count = 0
        self._available = False

    def start(self):
        if not self._check_available():
            return
        self._available = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def reset(self):
        with self._lock:
            self._sum = 0.0
            self._count = 0

    def get_summary(self) -> dict:
        with self._lock:
            avg = self._sum / self._count if self._count else 0.0
            return {"available": self._available, "avg_util": avg, "samples": self._count}

    def _check_available(self) -> bool:
        try:
            result = subprocess.run(["which", "xpu-smi"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _loop(self):
        while self._running:
            time.sleep(self._interval)
            util = self._read_gpu_util()
            if util is not None:
                with self._lock:
                    self._sum += util
                    self._count += 1

    def _read_gpu_util(self) -> Optional[float]:
        try:
            result = subprocess.run(self._command.split(), capture_output=True,
                                    text=True, timeout=5)
            if result.returncode != 0:
                return None
            data = json.loads(result.stdout)
            if "engine_util" in data and "compute" in data["engine_util"]:
                vals = [it.get("value", 0) for it in data["engine_util"]["compute"]
                        if isinstance(it, dict)]
                return max(vals) if vals else None
            if "gpu_util" in data:
                return float(data["gpu_util"])
        except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, OSError, KeyError):
            pass
        return None


# =============================================================================
# NPU
# =============================================================================

class NpuMonitor:
    """Intel NPU utilization monitor via VPU sysfs and intel_pmt telemetry."""

    PMT_GUIDS = {
        "0x130670b2": "MTL", "0x1306a0b3": "ARL", "0x1306a0b2": "ARL_H",
        "0x1306a0b4": "ARL_S", "0x3072005": "LNL", "0x3086000": "PTL",
        "0x308d000": "WCL",
    }

    def __init__(self, interval_ms: int = 500):
        self._interval = interval_ms / 1000.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._sum = 0.0
        self._count = 0
        self._available = False
        self._npu_busy_path: Optional[str] = None
        self._cpu_gen: Optional[str] = None
        self._prev_busy_us: Optional[int] = None
        self._prev_time: Optional[float] = None

    def start(self):
        self._detect_npu()
        if not self._available:
            return
        self._running = True
        self._prev_busy_us = self._read_busy_time()
        self._prev_time = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def reset(self):
        with self._lock:
            self._sum = 0.0
            self._count = 0
        self._prev_busy_us = self._read_busy_time()
        self._prev_time = time.monotonic()

    def get_summary(self) -> dict:
        with self._lock:
            avg = self._sum / self._count if self._count else 0.0
            return {"available": self._available, "cpu_gen": self._cpu_gen,
                    "avg_util": avg, "samples": self._count}

    def _detect_npu(self):
        drv_path = "/sys/bus/pci/drivers/intel_vpu/"
        if os.path.exists(drv_path):
            for entry in os.listdir(drv_path):
                if entry.startswith("0000:"):
                    busy_path = os.path.join(drv_path, entry, "npu_busy_time_us")
                    if os.path.exists(busy_path):
                        self._npu_busy_path = busy_path
                        self._available = True
                    break
        pmt_root = "/sys/class/intel_pmt"
        if os.path.exists(pmt_root):
            for telem_dir in os.listdir(pmt_root):
                if not telem_dir.startswith("telem"):
                    continue
                guid_path = os.path.join(pmt_root, telem_dir, "guid")
                if os.path.exists(guid_path):
                    guid = self._read_sysfs(guid_path)
                    if guid and guid in self.PMT_GUIDS:
                        self._cpu_gen = self.PMT_GUIDS[guid]
                        break

    @staticmethod
    def _read_sysfs(path: str) -> Optional[str]:
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except PermissionError:
            try:
                r = subprocess.run(["sudo", "cat", path],
                                   capture_output=True, text=True, timeout=2)
                if r.returncode == 0:
                    return r.stdout.strip()
            except (subprocess.TimeoutExpired, OSError):
                pass
        except OSError:
            pass
        return None

    def _read_busy_time(self) -> Optional[int]:
        if not self._npu_busy_path:
            return None
        try:
            with open(self._npu_busy_path, "r") as f:
                return int(f.read().strip())
        except PermissionError:
            try:
                r = subprocess.run(["sudo", "cat", self._npu_busy_path],
                                   capture_output=True, text=True, timeout=2)
                if r.returncode == 0:
                    return int(r.stdout.strip())
            except (subprocess.TimeoutExpired, ValueError, OSError):
                pass
            return None
        except (OSError, ValueError):
            return None

    def _loop(self):
        while self._running:
            time.sleep(self._interval)
            cur_busy = self._read_busy_time()
            cur_time = time.monotonic()
            if cur_busy is not None and self._prev_busy_us is not None:
                delta_us = cur_busy - self._prev_busy_us
                delta_s = cur_time - self._prev_time
                if delta_s > 0:
                    util = max(0.0, min(100.0, 100.0 * (delta_us / 1_000_000.0) / delta_s))
                    with self._lock:
                        self._sum += util
                        self._count += 1
            self._prev_busy_us = cur_busy
            self._prev_time = cur_time


# =============================================================================
# Process/system memory
# =============================================================================

class MemoryUsageMonitor:
    """Sample process RSS and host system memory usage in the background.

    The shape intentionally mirrors the useful subset of official llm_bench
    memory fields: max RSS, max system memory, and the increase relative to the
    start of a measured stage. Values are MiB.
    """

    def __init__(self, enabled: bool = True, interval_ms: int = 100):
        self.enabled = enabled
        self._interval = max(1, interval_ms) / 1000.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._samples: list[dict[str, float]] = []
        self._baseline: Optional[dict[str, float]] = None
        self._available = False
        self._pid = os.getpid()

    def start(self):
        if not self.enabled:
            return
        self.reset()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.enabled:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def reset(self):
        if not self.enabled:
            return
        sample = self._read_sample()
        with self._lock:
            self._samples = []
            self._baseline = sample
            if sample is not None:
                self._samples.append(sample)
                self._available = True

    def current(self) -> dict:
        if not self.enabled:
            return {"available": False}
        sample = self._read_sample()
        if sample is None:
            return {"available": False}
        return {
            "available": True,
            "rss_mem_mib": sample["rss_mib"],
            "sys_mem_mib": sample["sys_mib"],
            "rss_mem_share": self._share(sample["rss_mib"], sample["total_mib"]),
            "sys_mem_share": self._share(sample["sys_mib"], sample["total_mib"]),
        }

    def summary(self) -> dict:
        if not self.enabled:
            return {"available": False}
        sample = self._read_sample()
        with self._lock:
            if sample is not None:
                self._samples.append(sample)
                self._available = True
            if not self._samples or self._baseline is None:
                return {"available": False}
            samples = list(self._samples)
            baseline = dict(self._baseline)

        max_rss = max(s["rss_mib"] for s in samples)
        max_sys = max(s["sys_mib"] for s in samples)
        total_mib = samples[-1]["total_mib"]
        return {
            "available": self._available,
            "samples": len(samples),
            "start_rss_mem_mib": baseline["rss_mib"],
            "start_sys_mem_mib": baseline["sys_mib"],
            "max_rss_mem_mib": max_rss,
            "max_sys_mem_mib": max_sys,
            "max_rss_mem_increase_mib": max(0.0, max_rss - baseline["rss_mib"]),
            "max_sys_mem_increase_mib": max(0.0, max_sys - baseline["sys_mib"]),
            "max_rss_mem_share": self._share(max_rss, total_mib),
            "max_sys_mem_share": self._share(max_sys, total_mib),
        }

    def _loop(self):
        while self._running:
            time.sleep(self._interval)
            sample = self._read_sample()
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)
                    self._available = True

    def _read_sample(self) -> Optional[dict[str, float]]:
        rss_bytes = self._read_rss_bytes()
        sys_bytes, total_bytes = self._read_system_bytes()
        if rss_bytes is None or sys_bytes is None or not total_bytes:
            return None
        return {
            "rss_mib": rss_bytes / (1024.0 * 1024.0),
            "sys_mib": sys_bytes / (1024.0 * 1024.0),
            "total_mib": total_bytes / (1024.0 * 1024.0),
            "ts": time.perf_counter(),
        }

    def _read_rss_bytes(self) -> Optional[int]:
        if psutil is not None:
            try:
                proc = psutil.Process(self._pid)
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                return rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
                return None
        return self._read_proc_status_rss(self._pid)

    @staticmethod
    def _read_proc_status_rss(pid: int) -> Optional[int]:
        try:
            with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        return int(parts[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
        return None

    @staticmethod
    def _read_system_bytes() -> tuple[Optional[int], Optional[int]]:
        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                return vm.total - vm.available, vm.total
            except OSError:
                return None, None
        values: dict[str, int] = {}
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    key, raw = line.split(":", 1)
                    if key in {"MemTotal", "MemAvailable"}:
                        values[key] = int(raw.split()[0]) * 1024
            total = values.get("MemTotal")
            available = values.get("MemAvailable")
            if total is None or available is None:
                return None, None
            return total - available, total
        except (OSError, ValueError, IndexError):
            return None, None

    @staticmethod
    def _share(value_mib: float, total_mib: float) -> float:
        return 100.0 * value_mib / total_mib if total_mib else 0.0


# =============================================================================
# Bundle helpers
# =============================================================================

class MonitorBundle:
    """Convenience wrapper that owns one of each monitor."""

    def __init__(self, enabled: bool = True,
                 cpu_interval_ms: int = 100,
                 gpu_interval_ms: int = 500,
                 npu_interval_ms: int = 500,
                 gpu_command: Optional[str] = None):
        self.enabled = enabled
        self.cpu = CpuMonitor(interval_ms=cpu_interval_ms)
        self.gpu = GpuMonitor(interval_ms=gpu_interval_ms, command=gpu_command)
        self.npu = NpuMonitor(interval_ms=npu_interval_ms)

    def start(self):
        if not self.enabled:
            return
        self.cpu.start()
        self.gpu.start()
        self.npu.start()
        time.sleep(0.3)

    def stop(self):
        if not self.enabled:
            return
        self.cpu.stop()
        self.gpu.stop()
        self.npu.stop()

    def reset(self):
        if not self.enabled:
            return
        self.cpu.reset()
        self.gpu.reset()
        self.npu.reset()

    def summary(self) -> dict:
        if not self.enabled:
            return {"cpu": {}, "gpu": {}, "npu": {}}
        return {
            "cpu": self.cpu.get_summary(),
            "gpu": self.gpu.get_summary(),
            "npu": self.npu.get_summary(),
        }
