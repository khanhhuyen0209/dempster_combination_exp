# =============================================================================
# architecture/vgg.py
# -----------------------------------------------------------------------------
# VGG16-BN matching the Re-EDL paper's exact implementation.
# Source: https://github.com/[Re-EDL repo]/architectures/vgg.py
#
# Key differences from torchvision VGG16:
#
#   1. NO AdaptiveAvgPool2d
#      Torchvision has avgpool(7,7) before the classifier designed for
#      ImageNet 224×224. The paper omits it entirely. For CIFAR-10 32×32
#      with cfg 'D' (5 MaxPool2d layers): 32/2^5 = 1×1 feature map,
#      so features flatten to 512*1*1 = 512.
#
#   2. Smaller classifier: 512→512→output  (not 4096→4096→output)
#      Paper classifier:
#        Dropout(0.5) → Linear(512,512) → ReLU
#        Dropout(0.5) → Linear(512,512) → ReLU
#        Linear(512, output_dim)
#
#   3. Dropout before each linear layer (not after)
#
#   4. Kaiming normal weight init on all Conv2d layers
#
#   5. Batch norm after every conv (vgg16_bn config)
#      Essential for stable training from scratch on CIFAR-10.
#
#   6. k_lipschitz=None — paper uses no Lipschitz constraint in experiments
#      so we omit SpectralNorm entirely.
#
# The final linear layer outputs raw logits.
# Evidence is obtained via softplus in edl_core/opinion.py.
# =============================================================================

import math
import torch.nn as nn


def _make_layers(batch_norm: bool = True) -> nn.Sequential:
    """
    Build VGG16 feature extractor (cfg 'D') with optional batch normalisation.
    For CIFAR-10 32×32 input, 5 MaxPool2d layers produce a 1×1 spatial output,
    so the flattened feature dimension is 512.
    """
    cfg = [64, 64, 'M',
           128, 128, 'M',
           256, 256, 256, 'M',
           512, 512, 512, 'M',
           512, 512, 512, 'M']
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """
    VGG16-BN for CIFAR-10, matching Re-EDL paper architecture exactly.
    Input : (N, 3, 32, 32)
    Output: (N, num_classes) raw logits
    """

    def __init__(self, num_classes: int = 10, p_drop: float = 0.5):
        super().__init__()

        # Conv feature extractor — 5 MaxPool → 1×1 spatial output for 32×32 input
        self.features = _make_layers(batch_norm=True)

        # Classifier matches paper exactly:
        #   Dropout → Linear(512,512) → ReLU → Dropout → Linear(512,512) → ReLU
        #   → Linear(512, output_dim)
        # Note: 512 input because 32×32 / 2^5 = 1×1, so 512*1*1 = 512
        self.classifier = nn.Sequential(
            nn.Dropout(p=p_drop),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=p_drop),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),   # raw logits
        )

        # Kaiming normal init on all conv layers (matches paper)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten: (N, 512) — no adaptive pooling
        x = self.classifier(x)
        return x


def build_vgg16(num_classes: int = 10, p_drop: float = 0.5) -> nn.Module:
    """
    Factory function matching the paper's vgg16_bn() call signature.
    Returns VGG16-BN with paper-exact architecture. No pretrained weights.
    """
    return VGG16(num_classes=num_classes, p_drop=p_drop)
