"""
Loader for Madrylab/robustness-library ResNet50 checkpoints.

Used by Exp B (robustness-strength epsilon sweep) to load Salman et al.'s
ResNet50 Linf checkpoints at multiple training epsilon values.

Source: https://huggingface.co/madrylab/robust-imagenet-models
Files:  resnet50_linf_eps{0.5,1.0,2.0,4.0,8.0}.ckpt

Format details (Madrylab/robustness library):
    ckpt = {
        'model':       state_dict with prefixes:
                       'module.model.X'      <- the actual model weights (we keep)
                       'module.attacker.X'   <- adv-training attacker (we discard)
                       'module.normalizer.X' <- their built-in normalizer (we discard)
        'optimizer':   ... (we discard)
        'epoch':       ...
    }

We strip 'module.model.' prefix and load into a fresh torchvision ResNet50,
then wrap with our standard ImageNet normalization.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as T


class MadrylabResNet50Wrapper(nn.Module):
    """
    Wraps a torchvision ResNet50 with ImageNet normalization, so the model
    accepts inputs in [0, 1] (matching the convention used by other source
    models in this codebase).
    """

    def __init__(self, model: nn.Module, input_size: int = 224):
        super().__init__()
        self.model = model
        self.input_size = input_size
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            x = F.interpolate(
                x, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )
        x = self.normalize(x)
        return self.model(x)


def load_madrylab_resnet50(ckpt_path: str, device: str = "cuda:0") -> nn.Module:
    """
    Load a Madrylab/robustness ResNet50 checkpoint and return a model that
    takes inputs in [0, 1] and returns logits.

    Args:
        ckpt_path: path to .ckpt file.
        device:    target device.

    Returns:
        MadrylabResNet50Wrapper(ResNet50) on `device`, in eval mode.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Madrylab ckpt not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Madrylab format: top-level dict with 'model' key (state_dict)
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt  # raw state_dict

    # Strip 'module.model.' prefix (the actual model weights live here).
    # Skip 'module.attacker.*' (adv-training duplicate) and 'module.normalizer.*'
    # (we apply our own normalization in the wrapper).
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module.model."):
            new_sd[k[len("module.model."):]] = v

    if not new_sd:
        # Try alternative: maybe the checkpoint has no 'module.' prefix
        for k, v in sd.items():
            if k.startswith("model."):
                new_sd[k[len("model."):]] = v
        if not new_sd:
            raise ValueError(
                f"Failed to extract model parameters from {ckpt_path}; "
                f"first 5 keys were: {list(sd.keys())[:5]}"
            )

    # Build fresh torchvision ResNet50 and load weights
    model = models.resnet50(weights=None)
    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    if len(missing) > 0:
        # A few BN num_batches_tracked might be missing in older PyTorch — OK
        critical_missing = [m for m in missing if "num_batches_tracked" not in m]
        if critical_missing:
            print(f"  [Madrylab loader] WARNING: missing keys: "
                  f"{critical_missing[:5]}... ({len(critical_missing)} total)")
    if len(unexpected) > 0:
        print(f"  [Madrylab loader] Skipped {len(unexpected)} unexpected keys "
              f"(expected: attacker/normalizer modules)")

    # Wrap with normalization and move to device
    wrapped = MadrylabResNet50Wrapper(model)
    wrapped = wrapped.to(device)
    wrapped.eval()
    return wrapped
