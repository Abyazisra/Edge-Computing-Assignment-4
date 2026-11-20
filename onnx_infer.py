import argparse
import os

import numpy as np
import onnxruntime as ort

import config
from data import get_loaders
from infer import ALL_MODELS, class_names, dataset_for, load_image


def open_session(path, threads=1):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def onnx_accuracy(sess, loader):
    name = sess.get_inputs()[0].name
    n = c1 = c5 = 0
    for x, y in loader:
        logits = sess.run(None, {name: x.numpy()})[0]
        top = np.argsort(-logits, axis=1)[:, :5]
        y = y.numpy()
        c1 += int((top[:, 0] == y).sum())
        c5 += int((top == y[:, None]).any(axis=1).sum())
        n += len(y)
    return 100.0 * c1 / n, 100.0 * c5 / n


def model_files(name, variant):
    base = "nas_best" if name == "nas" else name
    return config.onnx_path(base, None if name == "nas" else variant)


def main():
    ap = argparse.ArgumentParser(description="CPU inference with ONNX Runtime")
    ap.add_argument("--model", default="all", choices=ALL_MODELS + ["all"])
    ap.add_argument("--variant", default="kd", choices=config.VARIANTS + ["both"])
    ap.add_argument("--image")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not args.image and not args.evaluate:
        ap.error("give --image or --evaluate")

    names = ALL_MODELS if args.model == "all" else [args.model]
    variants = config.VARIANTS if args.variant == "both" else [args.variant]
    for n in names:
        ds = dataset_for(n)
        for v in ([None] if n == "nas" else variants):
            path = model_files(n, v)
            if not os.path.exists(path):
                print("missing", path)
                continue
            sess = open_session(path)
            tag = n if v is None else f"{n}/{v}"
            if args.image:
                x = load_image(args.image, ds).numpy()
                p = softmax(sess.run(None, {"input": x})[0])[0]
                names_ = class_names(ds)
                print(f"\n{tag}")
                for i in np.argsort(-p)[:5]:
                    print(f"  {names_[i] if names_ else i:<20} {p[i]*100:5.1f}%")
            if args.evaluate:
                _, _, test_dl = get_loaders(ds, 256, args.smoke, workers=0 if args.smoke else 2)
                a1, a5 = onnx_accuracy(sess, test_dl)
                print(f"{tag:<30s} onnx top1={a1:.2f} top5={a5:.2f}")


if __name__ == "__main__":
    main()
