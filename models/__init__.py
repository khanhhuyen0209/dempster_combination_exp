import torch.nn as nn
from config import ID_DATASET
from .vgg    import build_vgg16
from .resnet import build_resnet18


def get_backbone(backbone_name: str = None,
                 num_classes: int = 10) -> nn.Module:
    name = backbone_name or ('vgg16_bn' if ID_DATASET == 'cifar10'
                             else 'resnet18')
    if name == 'vgg16_bn':  return build_vgg16(num_classes)
    if name == 'resnet18':  return build_resnet18(num_classes)
    raise ValueError(f"Unknown backbone: {name!r}")
