import argparse
import time

import torch

import config
from data import get_loaders
from kd_losses import HintRegressor, make_kd_loss
from models_student import build_student
from models_teacher import TeacherResNet50
from trainer import evaluate, fit, get_device, save_ckpt, save_json


def load_teacher(dev):
    ck = torch.load(config.ckpt_path("teacher_resnet50"), map_location="cpu")
    t = TeacherResNet50(100)
    t.load_state_dict(ck["model"])
    t.to(dev).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def run_one(name, teacher, args):
    torch.manual_seed(0)
    dev = get_device()
    train_dl, _, test_dl = get_loaders("cifar100", args.batch, args.smoke, workers=args.workers)
    student = build_student(name, 100)
    n_params = sum(p.numel() for p in student.parameters())

    reg = HintRegressor(student.feat_channels, teacher.feat_channels) if args.beta > 0 else None
    loss_fn = make_kd_loss(student, teacher, reg, args.T, args.alpha, args.beta)
    print(f"== distilling {name}  T={args.T} alpha={args.alpha} beta={args.beta}")

    t0 = time.time()
    hist = fit(student, train_dl, test_dl, args.epochs, args.lr, args.wd, loss_fn,
               extra=reg, label=f"{name}-kd")
    final = evaluate(student, test_dl, dev)
    print(name, "kd test:", final)

    # only the student goes to disk, the regressor is a training aid
    save_ckpt(config.ckpt_path(name, "kd"), student, acc=final["top1"], epochs=args.epochs,
              T=args.T, alpha=args.alpha, beta=args.beta)
    save_json(f"{config.RES}/{name}_kd.json",
              {"final": final, "params": n_params, "epochs": args.epochs,
               "T": args.T, "alpha": args.alpha, "beta": args.beta,
               "train_seconds": round(time.time() - t0, 1), "history": hist})


def main():
    ap = argparse.ArgumentParser(description="Knowledge distillation from ResNet-50 to small students")
    ap.add_argument("--student", default="all", choices=config.STUDENTS + ["all"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--T", type=float, default=4.0, help="softmax temperature")
    ap.add_argument("--alpha", type=float, default=0.7, help="weight of the soft target term")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of feature hint loss, 0 disables it")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    teacher = load_teacher(get_device())
    names = config.STUDENTS if args.student == "all" else [args.student]
    for n in names:
        run_one(n, teacher, args)


if __name__ == "__main__":
    main()
