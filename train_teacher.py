import argparse
import time

import torch
import torch.nn.functional as F

import config
from data import get_loaders
from models_teacher import TeacherResNet50
from trainer import evaluate, fit, get_device, save_ckpt, save_json


def main():
    ap = argparse.ArgumentParser(description="Train the ResNet-50 teacher on CIFAR-100")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run to test the script")
    args = ap.parse_args()

    torch.manual_seed(0)
    train_dl, _, test_dl = get_loaders("cifar100", args.batch, args.smoke, workers=args.workers)
    model = TeacherResNet50(100)
    print("teacher params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    loss_fn = lambda x, y: F.cross_entropy(model(x).float(), y)
    t0 = time.time()
    hist = fit(model, train_dl, test_dl, args.epochs, args.lr, args.wd, loss_fn, label="teacher")
    final = evaluate(model, test_dl, get_device())
    print("final test:", final)

    save_ckpt(config.ckpt_path("teacher_resnet50"), model, acc=final["top1"], epochs=args.epochs)
    save_json(config.RES + "/teacher_resnet50.json",
              {"final": final, "epochs": args.epochs, "train_seconds": round(time.time() - t0, 1),
               "history": hist})


if __name__ == "__main__":
    main()
