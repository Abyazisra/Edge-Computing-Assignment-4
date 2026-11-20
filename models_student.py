import torch
import torch.nn as nn


def cba(cin, cout, k=3, s=1, groups=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class InvRes(nn.Module):
    def __init__(self, cin, cout, stride, expand, k=3):
        super().__init__()
        mid = cin * expand
        body = []
        if expand != 1:
            body.append(cba(cin, mid, 1))
        body.append(cba(mid, mid, k, stride, groups=mid))
        body.append(cba(mid, cout, 1, act=False))
        self.body = nn.Sequential(*body)
        self.skip = stride == 1 and cin == cout

    def forward(self, x):
        y = self.body(x)
        return x + y if self.skip else y


class MBNet(nn.Module):
    """Shared skeleton for MobileNetV2 and EfficientNet-Lite0 (both are stacks of
    inverted residual blocks, they only differ in the block table)."""

    def __init__(self, table, stem=32, head=1280, classes=100, dropout=0.2):
        super().__init__()
        layers = [cba(3, stem, 3, 1)]
        c = stem
        for expand, k, cout, reps, stride in table:
            for j in range(reps):
                layers.append(InvRes(c, cout, stride if j == 0 else 1, expand, k))
                c = cout
        layers.append(cba(c, head, 1))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(head, classes)
        self.feat_channels = head
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, want_feat=False):
        f = self.features(x)
        out = self.fc(self.drop(torch.flatten(self.pool(f), 1)))
        return (out, f) if want_feat else out


# (expand, kernel, out channels, repeats, stride)
MBV2_TABLE = [
    (1, 3, 16, 1, 1), (6, 3, 24, 2, 1), (6, 3, 32, 3, 2), (6, 3, 64, 4, 2),
    (6, 3, 96, 3, 1), (6, 3, 160, 3, 2), (6, 3, 320, 1, 1),
]
LITE0_TABLE = [
    (1, 3, 16, 1, 1), (6, 3, 24, 2, 1), (6, 5, 40, 2, 2), (6, 3, 80, 3, 2),
    (6, 5, 112, 3, 1), (6, 5, 192, 4, 2), (6, 3, 320, 1, 1),
]


class Fire(nn.Module):
    def __init__(self, cin, squeeze, e1, e3):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(cin, squeeze, 1), nn.ReLU(inplace=True))
        self.e1 = nn.Sequential(nn.Conv2d(squeeze, e1, 1), nn.ReLU(inplace=True))
        self.e3 = nn.Sequential(nn.Conv2d(squeeze, e3, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.e1(s), self.e3(s)], dim=1)


class SqueezeNetC(nn.Module):
    def __init__(self, classes=100):
        super().__init__()
        pool = lambda: nn.MaxPool2d(3, 2, ceil_mode=True)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True), pool(),
            Fire(64, 16, 64, 64), Fire(128, 16, 64, 64), pool(),
            Fire(128, 32, 128, 128), Fire(256, 32, 128, 128), pool(),
            Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256), Fire(512, 64, 256, 256),
        )
        self.drop = nn.Dropout(0.3)
        self.cls = nn.Conv2d(512, classes, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_channels = 512

    def forward(self, x, want_feat=False):
        f = self.features(x)
        out = torch.flatten(self.pool(self.cls(self.drop(f))), 1)
        return (out, f) if want_feat else out


def build_student(name, classes=100):
    if name == "mobilenetv2":
        return MBNet(MBV2_TABLE, classes=classes)
    if name == "efficientnet_lite0":
        return MBNet(LITE0_TABLE, classes=classes)
    if name == "squeezenet":
        return SqueezeNetC(classes)
    raise ValueError(f"unknown student {name}")
