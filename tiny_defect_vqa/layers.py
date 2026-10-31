"""MobileNetV3 multiscale features and binary localization objective."""
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class Encoder(nn.Module):
    """Project stride 8/16/32 features to the shared 128-dimensional space."""

    def __init__(self, width=128, pretrained=False):
        super().__init__()
        self.features = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None).features
        self.projections = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, width, 1), nn.GroupNorm(8, width), nn.SiLU()) for channels in [24, 48, 576]])

    def forward(self, image):
        outputs = []
        for index, layer in enumerate(self.features):
            image = layer(image)
            if index in [3, 8, 12]:
                outputs.append(image)
        return [projection(value) for projection, value in zip(self.projections, outputs)]


def region(probability, target):
    """BCE plus per-image soft Dice, with nearest-neighbor target masks."""
    target = F.interpolate(target, size=probability.shape[-2:], mode="nearest")
    bce = F.binary_cross_entropy(probability.clamp(1e-7, 1-1e-7), target)
    dims = (1, 2, 3)
    dice = (2*(probability*target).sum(dims)+1e-7)/(probability.sum(dims)+target.sum(dims)+1e-7)
    return bce+1-dice.mean()
