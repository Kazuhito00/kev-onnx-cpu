# -*- coding: utf-8 -*-
"""One measurement per process.

The graph is fixed-length, so cost does not depend on how long the request is --
only on how many questions ride along (they all share one prefill). Runs one
configuration and exits, so ORT's arena cannot drift between configurations.
"""
import argparse
import ctypes
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kev_decide import Decider, choice, gpu_providers, noul, score  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def peak_rss_mb():
    """Peak working set in MB; argtypes are required or the handle truncates."""
    try:
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        query = kernel32.K32GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/onnx")
    ap.add_argument("--variant", default="q4f16")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--runs", type=int, default=9)
    ap.add_argument("--gpu", action="store_true",
                    help="use CUDA / DirectML if available (needs onnxruntime-gpu or -directml)")
    ap.add_argument("--questions", type=int, default=3)
    args = ap.parse_args()

    state = ("Hi, I was charged twice for my Pro subscription this month. "
             "Please refund the duplicate payment.")
    base = [choice("Which team?", ["billing", "technical", "sales"]),
            noul("The customer is angry."),
            score("How urgent?", ["low", "medium", "high", "critical"])]
    questions = [base[i % len(base)] for i in range(args.questions)]

    t0 = time.perf_counter()
    decider = Decider(ROOT / args.model, variant=args.variant,
                      intra_op_threads=args.threads,
                      providers=gpu_providers() if args.gpu else None)
    load_s = time.perf_counter() - t0

    provider = decider.session.get_providers()[0]
    if args.gpu and provider == "CPUExecutionProvider":
        print("warning: --gpu requested but running on CPU", file=sys.stderr)

    tokens = len(decider.encode(state, questions).input_ids)
    decider.decide(state, questions)  # warm
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        decider.decide(state, questions)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()

    print(json.dumps({
        "variant": args.variant,
        "provider": provider,
        "threads": args.threads,
        "runs": args.runs,
        "questions": len(questions),
        "prompt_tokens": tokens,
        "load_s": round(load_s, 2),
        "median_ms": round(statistics.median(times), 1),
        "min_ms": round(times[0], 1),
        "max_ms": round(times[-1], 1),
        "peak_rss_mb": peak_rss_mb(),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
