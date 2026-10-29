import os
import time

import torch


_enabled = os.getenv("TRI_TIME_PROFILE", "0") == "1"
_start_time = None
_events = {"trimul": [], "triatt": []}
_totals = {"trimul": 0.0, "triatt": 0.0}
_calls = {"trimul": 0, "triatt": 0}


def begin():
    global _start_time
    if not _enabled:
        return
    _start_time = time.perf_counter()
    for kind in _events:
        _events[kind].clear()
        _totals[kind] = 0.0
        _calls[kind] = 0


def record(kind, fn, *args, **kwargs):
    if not _enabled or _start_time is None:
        return fn(*args, **kwargs)
    stream = torch.musa.current_stream()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record(stream)
    result = fn(*args, **kwargs)
    end.record(stream)
    _events[kind].append((start, end))
    _calls[kind] += 1
    return result


def report(label):
    if not _enabled or _start_time is None:
        return
    torch.musa.synchronize()
    for kind in _events:
        _totals[kind] += sum(start.elapsed_time(end) for start, end in _events[kind])
        _events[kind].clear()
    total_ms = (time.perf_counter() - _start_time) * 1000.0
    trimul_ms = _totals["trimul"]
    triatt_ms = _totals["triatt"]
    trimul_pct = 100.0 * trimul_ms / total_ms if total_ms else 0.0
    triatt_pct = 100.0 * triatt_ms / total_ms if total_ms else 0.0
    print(
        "TRI_TIME_PROFILE "
        f"label={label} trimul_ms={trimul_ms:.3f} "
        f"triatt_ms={triatt_ms:.3f} total_ms={total_ms:.3f} "
        f"trimul_pct={trimul_pct:.2f} triatt_pct={triatt_pct:.2f} "
        f"trimul_calls={_calls['trimul']} triatt_calls={_calls['triatt']}",
        flush=True,
    )
