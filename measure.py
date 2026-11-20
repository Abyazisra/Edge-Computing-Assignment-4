import copy
import io
import os
import threading
import time

import numpy as np
import psutil
import torch


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def state_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 2**20


def file_size_mb(path):
    return os.path.getsize(path) / 2**20


def _stats(times):
    t = np.array(times)
    return {"median_ms": float(np.median(t)), "mean_ms": float(t.mean()),
            "p95_ms": float(np.percentile(t, 95)), "std_ms": float(t.std())}


@torch.inference_mode()
def torch_cpu_latency(model, runs=100, warmup=20, threads=1, size=32):
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, 3, size, size)
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        for _ in range(warmup):
            m(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            m(x)
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(old)
    return _stats(times)


def ort_cpu_latency(session, runs=100, warmup=20, size=32):
    name = session.get_inputs()[0].name
    x = np.random.randn(1, 3, size, size).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000)
    return _stats(times)


def peak_rss_delta_mb(fn):
    """Runs fn() while a thread samples the process RSS, returns the peak growth in MB."""
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.001)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    try:
        fn()
    finally:
        stop.set()
        th.join()
    return max(0, peak[0] - base) / 2**20


@torch.inference_mode()
def torch_peak_memory(model, iters=20, size=32):
    """peak RSS growth while the model is copied and run (CPU, batch 1)"""
    x = torch.randn(1, 3, size, size)

    def work():
        m = copy.deepcopy(model).cpu().eval()
        for _ in range(iters):
            m(x)

    return peak_rss_delta_mb(work)
