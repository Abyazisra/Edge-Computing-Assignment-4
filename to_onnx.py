import argparse
import os
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch

import config
from infer import ALL_MODELS, load_model


warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)


def export(model, path):
    dummy = torch.randn(1, 3, 32, 32)
    kwargs = dict(input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=17)
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, path, **kwargs)


def main():
    ap = argparse.ArgumentParser(description="Export trained models to ONNX")
    ap.add_argument("--model", default="all", choices=ALL_MODELS + ["all"])
    args = ap.parse_args()

    names = ALL_MODELS if args.model == "all" else [args.model]
    for n in names:
        for v in ([None] if n == "nas" else config.VARIANTS):
            try:
                model = load_model(n, v)
            except FileNotFoundError:
                print(f"skip {n} {v or ''}: no checkpoint")
                continue
            out = config.onnx_path(n if n != "nas" else "nas_best", v)
            export(model, out)
            onnx.checker.check_model(onnx.load(out))

            # compare against pytorch on a random batch
            x = torch.randn(4, 3, 32, 32)
            with torch.no_grad():
                ref = model(x).numpy()
            sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
            got = sess.run(None, {"input": x.numpy()})[0]
            diff = float(np.abs(ref - got).max())
            print(f"exported {os.path.basename(out)}  {os.path.getsize(out)/2**20:.2f}MB  max|diff|={diff:.2e}")


if __name__ == "__main__":
    main()
