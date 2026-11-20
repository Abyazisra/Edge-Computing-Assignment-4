import argparse
import csv
import json
import time

import optuna
import torch
import torch.nn.functional as F

import config
import measure
from data import get_loaders
from nas_space import build_network, suggest_config
from trainer import evaluate, fit, get_device, save_json

FIELDS = ["trial", "status", "reason", "val_acc", "latency_ms", "p95_ms", "size_mb",
          "params", "peak_mem_mb", "score", "train_s", "config"]


def main():
    ap = argparse.ArgumentParser(description="Hardware aware NAS with Optuna (TPE)")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=3, help="epochs per candidate")
    ap.add_argument("--train_subset", type=int, default=15000)
    ap.add_argument("--val_size", type=int, default=5000)
    ap.add_argument("--max_latency", type=float, default=10.0, help="ms, batch 1 on CPU")
    ap.add_argument("--max_size", type=float, default=1.5, help="MB of the saved weights")
    ap.add_argument("--lam", type=float, default=0.05, help="latency penalty in the score")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads used when timing")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train_dl, val_dl, _ = get_loaders("cifar10", 128, args.smoke, val_size=args.val_size if not args.smoke else 64,
                                      train_subset=args.train_subset, workers=args.workers)
    dev = get_device()
    rows = []

    def log_row(row):
        rows.append(row)
        with open(f"{config.RES}/nas_history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        save_json(f"{config.RES}/nas_history.json", rows)

    def objective(trial):
        cfg = suggest_config(trial)
        net = build_network(cfg, 10)

        # hardware measurements come first, before any training
        lat = measure.torch_cpu_latency(net, threads=args.threads)
        size = measure.state_size_mb(net)
        mem = measure.torch_peak_memory(net)
        params = measure.count_params(net)
        base = {"trial": trial.number, "latency_ms": round(lat["median_ms"], 3),
                "p95_ms": round(lat["p95_ms"], 3), "size_mb": round(size, 3),
                "params": params, "peak_mem_mb": round(mem, 2), "config": json.dumps(cfg)}

        reasons = []
        if lat["median_ms"] > args.max_latency:
            reasons.append(f"latency {lat['median_ms']:.2f}ms > {args.max_latency}ms")
        if size > args.max_size:
            reasons.append(f"size {size:.2f}MB > {args.max_size}MB")
        if reasons:
            log_row({**base, "status": "rejected", "reason": "; ".join(reasons),
                     "val_acc": "", "score": "", "train_s": 0})
            print(f"trial {trial.number}: rejected ({'; '.join(reasons)})")
            raise optuna.TrialPruned()

        t0 = time.time()
        loss_fn = lambda x, y: F.cross_entropy(net(x).float(), y)
        fit(net, train_dl, val_dl, args.epochs, 0.05, 5e-4, loss_fn, eval_every=args.epochs,
            warmup_epochs=0, label=f"nas-{trial.number}")
        acc = evaluate(net, val_dl, dev)["top1"]
        score = acc / 100.0 - args.lam * lat["median_ms"] / args.max_latency
        log_row({**base, "status": "trained", "reason": "", "val_acc": round(acc, 3),
                 "score": round(score, 4), "train_s": round(time.time() - t0, 1)})
        print(f"trial {trial.number}: acc={acc:.2f} lat={lat['median_ms']:.2f}ms size={size:.2f}MB")
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=10))
    study.optimize(objective, n_trials=args.trials)

    trained = [r for r in rows if r["status"] == "trained"]
    if not trained:
        raise SystemExit("every candidate was rejected, relax --max_latency / --max_size")
    best = max(trained, key=lambda r: r["score"])
    out = {"config": json.loads(best["config"]),
           "metrics": {k: best[k] for k in ("val_acc", "latency_ms", "p95_ms", "size_mb",
                                            "params", "peak_mem_mb", "score")},
           "constraints": {"max_latency_ms": args.max_latency, "max_size_mb": args.max_size},
           "trial": best["trial"],
           "n_trials": len(rows), "n_rejected": len(rows) - len(trained)}
    save_json(f"{config.RES}/best_arch.json", out)
    print("best architecture:", json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
