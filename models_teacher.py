import torch
import torch.nn as nn
from torchvision.models import resnet50


class TeacherResNet50(nn.Module):
    """ResNet-50 with a 3x3 stem and no max-pool so it suits 32x32 inputs."""

    def __init__(self, classes=100):
        super().__init__()
        net = resnet50(weights=None, num_classes=classes)
        net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        self.net = net
        self.feat_channels = 2048

    def forward(self, x, want_feat=False):
        n = self.net
        x = n.relu(n.bn1(n.conv1(x)))
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        feat = n.layer4(x)
        out = n.fc(torch.flatten(n.avgpool(feat), 1))
        return (out, feat) if want_feat else out
