import json
import math
import time

import torch
import torch.nn.functional as F


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    n = c1 = c5 = 0
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        total_loss += F.cross_entropy(out, y, reduction="sum").item()
        top = out.topk(min(5, out.size(1)), dim=1).indices
        c1 += (top[:, 0] == y).sum().item()
        c5 += (top == y[:, None]).any(dim=1).sum().item()
        n += y.numel()
    return {"loss": total_loss / n, "top1": 100.0 * c1 / n, "top5": 100.0 * c5 / n}


def fit(model, train_dl, eval_dl, epochs, lr, wd, loss_fn, extra=None,
        eval_every=1, warmup_epochs=1, label=""):
    """Generic SGD + cosine training loop.

    loss_fn(x, y) must return a scalar loss. `extra` is an optional module
    (for example a feature regressor) whose parameters are trained as well.
    """
    dev = get_device()
    model.to(dev)
    params = list(model.parameters())
    if extra is not None:
        extra.to(dev)
        params += list(extra.parameters())
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)

    steps_per_epoch = len(train_dl)
    total = epochs * steps_per_epoch
    warm = max(1, warmup_epochs * steps_per_epoch)

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if extra is not None:
            extra.train()
        run, seen = 0.0, 0
        for x, y in train_dl:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=use_amp):
                loss = loss_fn(x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item() * y.size(0)
            seen += y.size(0)
        row = {"epoch": ep + 1, "train_loss": run / seen}
        if eval_dl is not None and ((ep + 1) % eval_every == 0 or ep + 1 == epochs):
            row.update({f"val_{k}": v for k, v in evaluate(model, eval_dl, dev).items()})
        row["time_s"] = round(time.time() - t0, 1)
        history.append(row)
        msg = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
        print(f"[{label}] {msg}", flush=True)
    return history


def save_ckpt(path, model, **meta):
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, **meta}, path)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
