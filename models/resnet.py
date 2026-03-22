# =============================================================================
# architecture/resnet.py
# -----------------------------------------------------------------------------
# ResNet-18 for CIFAR-10, adapted for 32×32 input.
#
# Standard torchvision ResNet-18 is designed for ImageNet (224×224).
# For CIFAR-10 (32×32), two modifications are standard in the literature:
#   1. Replace 7×7 conv (stride=2) with 3×3 conv (stride=1) — preserves
#      spatial resolution at the first layer
#   2. Remove the first MaxPool2d — prevents over-downsampling on small inputs
#
# These modifications are standard in uncertainty/OOD papers using ResNet
# on CIFAR-10 (e.g. deep ensemble papers, MC Dropout papers).
# Output: (N, num_classes) raw logits — softplus applied in edl_core/opinion.py
# =============================================================================

import torch.nn as nn
import torchvision.models as models


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 adapted for CIFAR-10 32×32 input.
    Replaces 7×7 stem conv with 3×3 and removes first MaxPool.
    No pretrained weights — trained from scratch.
    """
    model = models.resnet18(weights=None)

    # Replace 7×7 stride-2 conv with 3×3 stride-1 for 32×32 input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                             padding=1, bias=False)

    # Remove MaxPool — would reduce 32×32 to 8×8 too aggressively
    model.maxpool = nn.Identity()

    # Replace final fc with K-class head outputting raw logits
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
