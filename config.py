import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
ONNX = os.path.join(ROOT, "onnx_models")
RES = os.path.join(ROOT, "results")

for _d in (DATA, CKPT, ONNX, RES):
    os.makedirs(_d, exist_ok=True)

STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}

STUDENTS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
VARIANTS = ["baseline", "kd"]


def ckpt_path(name, variant=None):
    if variant is None:
        return os.path.join(CKPT, f"{name}.pt")
    return os.path.join(CKPT, f"{name}_{variant}.pt")


def onnx_path(name, variant=None):
    base = name if variant is None else f"{name}_{variant}"
    return os.path.join(ONNX, base + ".onnx")
