import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_target_loss(s_logits, t_logits, T):
    """KL divergence between temperature softened distributions."""
    log_p = F.log_softmax(s_logits.float() / T, dim=1)
    q = F.softmax(t_logits.float() / T, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (T * T)


class HintRegressor(nn.Module):
    """1x1 conv that maps student feature maps to the teacher channel count."""

    def __init__(self, cin, cout):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return self.proj(x)


def feature_hint_loss(regressor, s_feat, t_feat):
    s = regressor(s_feat)
    if s.shape[-2:] != t_feat.shape[-2:]:
        s = F.adaptive_avg_pool2d(s, t_feat.shape[-2:])
    return F.mse_loss(s.float(), t_feat.float())


def make_kd_loss(student, teacher, regressor, T=4.0, alpha=0.7, beta=1.0):
    """loss = alpha * soft + (1 - alpha) * hard + beta * feature hint."""
    teacher.eval()

    def fn(x, y):
        with torch.no_grad():
            t_out, t_feat = teacher(x, want_feat=True)
        s_out, s_feat = student(x, want_feat=True)
        hard = F.cross_entropy(s_out.float(), y)
        soft = soft_target_loss(s_out, t_out, T)
        loss = alpha * soft + (1 - alpha) * hard
        if regressor is not None and beta > 0:
            loss = loss + beta * feature_hint_loss(regressor, s_feat, t_feat)
        return loss

    return fn
