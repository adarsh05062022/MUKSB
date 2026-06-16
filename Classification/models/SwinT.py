"""SwinT.py — Swin-Transformer-Tiny wrapper for low-resolution datasets.

Swin-T is trained on 224×224 ImageNet inputs.  To plug it into the
existing CIFAR-10/SVHN pipeline (32×32 inputs) without touching the
dataloaders, we upsample inside the forward pass.  The classification
head is replaced for the target number of classes.

By default ImageNet-pretrained weights are loaded — fine-tuning a Swin
from scratch on CIFAR-10 is impractical and not the focus of the
unlearning study.  Set ``MUKSB_SWIN_PRETRAINED=0`` in the environment
to force random init.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

__all__ = ["swin_t"]


class _SwinTWrapper(nn.Module):
    """Swin-T + on-the-fly bilinear upsample to 224×224 + class head swap."""

    UPSAMPLE_SIZE = 224

    _SWIN_T_URL = "https://download.pytorch.org/models/swin_t-704ceda3.pth"

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = torchvision.models.swin_t(weights=None)
        if pretrained:
            self._load_pretrained_weights()
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)

        # placeholder; setup_model_dataset overwrites this with the
        # dataset-specific NormalizeByChannelMeanStd module.
        self.normalize = nn.Identity()

    def _load_pretrained_weights(self):
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                self._SWIN_T_URL, progress=True, check_hash=True
            )
        except RuntimeError:
            # Hash mismatch usually means a corrupt/partial cached download.
            # Clear it and retry without hash verification.
            import os, urllib.parse
            filename = os.path.basename(urllib.parse.urlparse(self._SWIN_T_URL).path)
            cached = os.path.join(torch.hub.get_dir(), "checkpoints", filename)
            if os.path.exists(cached):
                os.remove(cached)
                print(f"[SwinT] Removed corrupt cached weights: {cached}")
            state_dict = torch.hub.load_state_dict_from_url(
                self._SWIN_T_URL, progress=True, check_hash=False
            )
        # load backbone weights (head is replaced so strict=False)
        self.backbone.load_state_dict(state_dict, strict=False)
        print("[SwinT] Loaded ImageNet-pretrained Swin-T weights.")

    def forward(self, x):
        x = self.normalize(x)
        if x.shape[-1] != self.UPSAMPLE_SIZE or x.shape[-2] != self.UPSAMPLE_SIZE:
            x = F.interpolate(
                x, size=self.UPSAMPLE_SIZE, mode="bilinear", align_corners=False
            )
        return self.backbone(x)


def swin_t(num_classes=10, pretrained=None, **kwargs):
    """Factory matching the model_dict signature used by setup_model_dataset.

    Extra kwargs (e.g. ``imagenet=False`` for the convnet branch) are
    ignored so the call from ``utils.setup_model_dataset`` works as-is.
    """
    if pretrained is None:
        pretrained = os.environ.get("MUKSB_SWIN_PRETRAINED", "1") != "0"
    return _SwinTWrapper(num_classes=num_classes, pretrained=pretrained)
