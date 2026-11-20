import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

import config
from data import get_loaders
from models_student import build_student
from nas_space import build_network
from trainer import evaluate

ALL_MODELS = config.STUDENTS + ["nas"]


def load_model(name, variant="kd"):
    """name is one of the students or 'nas'. variant only matters for students."""
    if name == "nas":
        ck = torch.load(config.ckpt_path("nas_best"), map_location="cpu")
        model = build_network(ck["cfg"], 10)
    else:
        ck = torch.load(config.ckpt_path(name, variant), map_location="cpu")
        model = build_student(name, 100)
    model.load_state_dict(ck["model"])
    return model.eval()


def dataset_for(name):
    return "cifar10" if name == "nas" else "cifar100"


def class_names(dataset):
    cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    try:
        return cls(config.DATA, train=False, download=False).classes
    except Exception:
        try:
            return cls(config.DATA, train=False, download=True).classes
        except Exception:
            return None


def load_image(path, dataset):
    mean, std = config.STATS[dataset]
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def predict(model, x, k=5):
    probs = torch.softmax(model(x), dim=1)[0]
    p, idx = probs.topk(k)
    return p.tolist(), idx.tolist()


def run_image(name, variant, path):
    ds = dataset_for(name)
    names = class_names(ds)
    model = load_model(name, variant)
    p, idx = predict(model, load_image(path, ds))
    tag = name if name == "nas" else f"{name}/{variant}"
    print(f"\n{tag}")
    for pi, ii in zip(p, idx):
        label = names[ii] if names else str(ii)
        print(f"  {label:<20s} {pi*100:5.1f}%")


def run_eval(name, variant, smoke):
    ds = dataset_for(name)
    _, _, test_dl = get_loaders(ds, 256, smoke, workers=0 if smoke else 2)
    r = evaluate(load_model(name, variant), test_dl, torch.device("cpu"))
    tag = name if name == "nas" else f"{name}/{variant}"
    print(f"{tag:<30s} top1={r['top1']:.2f}  top5={r['top5']:.2f}")


def main():
    ap = argparse.ArgumentParser(description="Inference for the trained students and the NAS model")
    ap.add_argument("--model", default="all", choices=ALL_MODELS + ["all"])
    ap.add_argument("--variant", default="kd", choices=config.VARIANTS + ["both"])
    ap.add_argument("--image", help="path to an image file")
    ap.add_argument("--evaluate", action="store_true", help="accuracy on the test split")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not args.image and not args.evaluate:
        ap.error("give --image or --evaluate")

    names = ALL_MODELS if args.model == "all" else [args.model]
    variants = config.VARIANTS if args.variant == "both" else [args.variant]
    for n in names:
        for v in ([None] if n == "nas" else variants):
            try:
                if args.image:
                    run_image(n, v, args.image)
                if args.evaluate:
                    run_eval(n, v, args.smoke)
            except FileNotFoundError:
                print(f"checkpoint for {n} {v or ''} not found, skipping")


if __name__ == "__main__":
    main()
