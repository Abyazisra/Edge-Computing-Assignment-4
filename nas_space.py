import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "relu6": lambda: nn.ReLU6(inplace=True),
    "silu": lambda: nn.SiLU(inplace=True),
    "hardswish": lambda: nn.Hardswish(inplace=True),
    "leaky": lambda: nn.LeakyReLU(0.1, inplace=True),
}
RESOLUTIONS = [24, 32, 40]
STEM_CHOICES = [8, 16, 24]
WIDTHS = [16, 24, 32, 48, 64, 96, 128]
KERNELS = [3, 5, 7]
BLOCK_KINDS = ["conv", "dws"]
DATA_RES = 32  # resolution of the stored images


def suggest_config(trial):
    """Define-by-run sampling with an Optuna trial."""
    cfg = {
        "resolution": trial.suggest_categorical("resolution", RESOLUTIONS),
        "act": trial.suggest_categorical("act", list(ACTIVATIONS)),
        "stem": trial.suggest_categorical("stem", STEM_CHOICES),
        "stages": [],
    }
    n_stages = trial.suggest_int("n_stages", 2, 4)
    for i in range(n_stages):
        cfg["stages"].append({
            "depth": trial.suggest_int(f"s{i}_depth", 1, 3),
            "width": trial.suggest_categorical(f"s{i}_width", WIDTHS),
            "kernel": trial.suggest_categorical(f"s{i}_kernel", KERNELS),
            "block": trial.suggest_categorical(f"s{i}_block", BLOCK_KINDS),
        })
    return cfg


def _conv_bn_act(cin, cout, k, stride, act, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        ACTIVATIONS[act](),
    )


class StdBlock(nn.Sequential):
    def __init__(self, cin, cout, k, stride, act):
        super().__init__(_conv_bn_act(cin, cout, k, stride, act))


class DwsBlock(nn.Sequential):
    """depthwise k x k followed by pointwise 1 x 1"""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__(
            _conv_bn_act(cin, cin, k, stride, act, groups=cin),
            _conv_bn_act(cin, cout, 1, 1, act),
        )


class SearchNet(nn.Module):
    """Network built from a sampled config. The resize to the searched input
    resolution happens inside forward so every model accepts 32x32 images."""

    def __init__(self, cfg, classes=10):
        super().__init__()
        self.cfg = cfg
        self.res = cfg["resolution"]
        act = cfg["act"]
        blocks = [_conv_bn_act(3, cfg["stem"], 3, 1, act)]
        c = cfg["stem"]
        for si, st in enumerate(cfg["stages"]):
            cls = DwsBlock if st["block"] == "dws" else StdBlock
            for d in range(st["depth"]):
                stride = 2 if (si > 0 and d == 0) else 1
                blocks.append(cls(c, st["width"], st["kernel"], stride, act))
                c = st["width"]
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c, classes)

    def forward(self, x):
        if self.res != DATA_RES:
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        x = self.pool(self.body(x)).flatten(1)
        return self.fc(x)


def build_network(cfg, classes=10):
    return SearchNet(cfg, classes)
