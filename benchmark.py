import argparse
import csv
import os

import numpy as np

import config
import measure
from data import get_loaders
from infer import ALL_MODELS, dataset_for, load_model
from onnx_infer import model_files, onnx_accuracy, open_session

COLS = ["model", "variant", "dataset", "params_M", "pt_size_MB", "onnx_size_MB", "top1_onnx",
        "top5_onnx", "torch_lat_ms", "torch_p95_ms", "ort_lat_ms", "ort_p95_ms",
        "ort_throughput_ips", "peak_mem_MB"]


def main():
    ap = argparse.ArgumentParser(description="Benchmark every exported model on CPU")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--skip_acc", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    rows = []
    for n in ALL_MODELS:
        for v in ([None] if n == "nas" else config.VARIANTS):
            onnx_file = model_files(n, v)
            if not os.path.exists(onnx_file):
                print("skip", n, v)
                continue
            model = load_model(n, v or "kd")
            sess = open_session(onnx_file, args.threads)

            tl = measure.torch_cpu_latency(model, runs=args.runs, threads=args.threads)
            ol = measure.ort_cpu_latency(sess, runs=args.runs)
            probe = np.random.randn(1, 3, 32, 32).astype(np.float32)

            def load_and_run():
                s2 = open_session(onnx_file, args.threads)
                for _ in range(30):
                    s2.run(None, {"input": probe})

            mem = measure.peak_rss_delta_mb(load_and_run)

            a1 = a5 = ""
            if not args.skip_acc:
                ds = dataset_for(n)
                _, _, tdl = get_loaders(ds, 256, args.smoke, workers=0 if args.smoke else 2)
                a1, a5 = [round(a, 2) for a in onnx_accuracy(sess, tdl)]

            row = {"model": n, "variant": v or "nas", "dataset": dataset_for(n),
                   "params_M": round(measure.count_params(model) / 1e6, 3),
                   "pt_size_MB": round(measure.state_size_mb(model), 3),
                   "onnx_size_MB": round(measure.file_size_mb(onnx_file), 3),
                   "top1_onnx": a1, "top5_onnx": a5,
                   "torch_lat_ms": round(tl["median_ms"], 3), "torch_p95_ms": round(tl["p95_ms"], 3),
                   "ort_lat_ms": round(ol["median_ms"], 3), "ort_p95_ms": round(ol["p95_ms"], 3),
                   "ort_throughput_ips": round(1000.0 / ol["median_ms"], 1),
                   "peak_mem_MB": round(mem, 2)}
            rows.append(row)
            print(row)

    with open(f"{config.RES}/benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", f"{config.RES}/benchmark.csv")


if __name__ == "__main__":
    main()
