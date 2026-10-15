import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def conv(inputs, outputs, kernel=3):
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel, padding=kernel//2), nn.GroupNorm(1, outputs), nn.SiLU())


class Encoder(nn.Module):
    def __init__(self, channels=3, width=32):
        super().__init__()
        self.layers = nn.Sequential(conv(channels, width), conv(width, width), conv(width, width))

    def forward(self, x):
        return self.layers(x.float())


def dice(p, y):
    return 1 - (2 * (p * y).sum() + 1e-6) / (p.sum() + y.sum() + 1e-6)


def region(p, y):
    return F.binary_cross_entropy(p.clamp(1e-6, 1-1e-6), y.float()) + dice(p, y)


def boundary(y, radius=3):
    return (F.max_pool2d(y, 2*radius+1, 1, radius) + F.max_pool2d(-y, 2*radius+1, 1, radius) > 0).float()


def warp(x, flow):
    n, _, h, w = x.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                            torch.linspace(-1, 1, w, device=x.device), indexing="ij")
    grid = torch.stack([xx, yy], -1)[None].expand(n, -1, -1, -1)
    displacement = torch.stack([flow[:, 0] * 2 / max(w-1, 1), flow[:, 1] * 2 / max(h-1, 1)], -1)
    return F.grid_sample(x, grid + displacement, mode="bilinear", padding_mode="border", align_corners=True)


def image_demo(seed, n, channels=3, mask=True):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, channels, 32, 32)).astype("float32")
    y = (x[:, :1] > .5).astype("float32") if mask else x.copy()
    return {"x": x, "y": y, "group": np.arange(n)}


def classification_demo(seed, n, dimensions=12, classes=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dimensions)).astype("float32")
    y = np.argmax(x[:, :classes], 1).astype("int64")
    return {"x": x, "y": y, "group": np.arange(n)}

