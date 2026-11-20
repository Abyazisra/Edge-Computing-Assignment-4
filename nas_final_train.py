import argparse
import json
import time

import torch
import torch.nn.functional as F

import config
import measure
from data import get_loaders
from nas_space import build_network
from trainer import evaluate, fit, get_device, save_ckpt, save_json


def main():
    ap = argparse.ArgumentParser(description="Train the best NAS architecture from scratch on CIFAR-10")
    ap.add_argument("--arch", default=f"{config.RES}/best_arch.json")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with open(args.arch) as f:
        cfg = json.load(f)["config"]

    torch.manual_seed(0)
    train_dl, _, test_dl = get_loaders("cifar10", args.batch, args.smoke, workers=args.workers)
    model = build_network(cfg, 10)
    print("params: %.3fM  size: %.2fMB" % (measure.count_params(model) / 1e6, measure.state_size_mb(model)))

    loss_fn = lambda x, y: F.cross_entropy(model(x).float(), y)
    t0 = time.time()
    hist = fit(model, train_dl, test_dl, args.epochs, args.lr, args.wd, loss_fn, label="nas-final")
    final = evaluate(model, test_dl, get_device())
    print("final test:", final)

    save_ckpt(config.ckpt_path("nas_best"), model, cfg=cfg, acc=final["top1"], epochs=args.epochs)
    save_json(f"{config.RES}/nas_best.json",
              {"final": final, "config": cfg, "epochs": args.epochs,
               "params": measure.count_params(model),
               "train_seconds": round(time.time() - t0, 1), "history": hist})


if __name__ == "__main__":
    main()
