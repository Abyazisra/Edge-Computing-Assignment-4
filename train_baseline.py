import argparse
import time

import torch
import torch.nn.functional as F

import config
from data import get_loaders
from models_student import build_student
from trainer import evaluate, fit, get_device, save_ckpt, save_json


def run_one(name, args):
    torch.manual_seed(0)
    train_dl, _, test_dl = get_loaders("cifar100", args.batch, args.smoke, workers=args.workers)
    model = build_student(name, 100)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"== baseline {name}  params={n_params/1e6:.2f}M")

    loss_fn = lambda x, y: F.cross_entropy(model(x).float(), y)
    t0 = time.time()
    hist = fit(model, train_dl, test_dl, args.epochs, args.lr, args.wd, loss_fn,
               label=f"{name}-baseline")
    final = evaluate(model, test_dl, get_device())
    print(name, "baseline test:", final)

    save_ckpt(config.ckpt_path(name, "baseline"), model, acc=final["top1"], epochs=args.epochs)
    save_json(f"{config.RES}/{name}_baseline.json",
              {"final": final, "params": n_params, "epochs": args.epochs,
               "train_seconds": round(time.time() - t0, 1), "history": hist})


def main():
    ap = argparse.ArgumentParser(description="Train student models without distillation")
    ap.add_argument("--student", default="all", choices=config.STUDENTS + ["all"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    names = config.STUDENTS if args.student == "all" else [args.student]
    for n in names:
        run_one(n, args)


if __name__ == "__main__":
    main()
