"""
Model loading utilities for transfer attack experiments.
Wraps RobustBench model loading with consistent interface.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
from robustbench.utils import load_model

# Fix for PyTorch 2.6 weights_only=True compatibility
try:
    import easydict
    torch.serialization.add_safe_globals([easydict.EasyDict])
except (ImportError, AttributeError):
    pass  # Older PyTorch versions or easydict not installed


# Model registry with metadata
MODEL_INFO = {
    # Natural models
    "Standard": {
        "type": "natural",
        "architecture": "WideResNet-28-10",
        "robust_acc": 0.0,
    },
    # Adversarially trained models
    "Engstrom2019Robustness": {
        "type": "robust_at",
        "architecture": "ResNet-50",
        "robust_acc": 49.25,
    },
    "Rice2020Overfitting": {
        "type": "robust_at",
        "architecture": "WideResNet-34-20",
        "robust_acc": 53.42,
    },
    "Gowal2020Uncovering_70_16": {
        "type": "robust_at",
        "architecture": "WideResNet-70-16",
        "robust_acc": 57.14,
    },
    # Semi-supervised + AT
    "Carmon2019Unlabeled": {
        "type": "robust_semi",
        "architecture": "WideResNet-28-10",
        "robust_acc": 59.53,
    },
    # SOTA with diffusion data
    "Wang2023Better_WRN-28-10": {
        "type": "robust_diffusion",
        "architecture": "WideResNet-28-10",
        "robust_acc": 67.31,
    },
    # Additional Robust models for Tier-3 expansion
    "Zhang2019Theoretically": {
        "type": "robust_at",
        "architecture": "WideResNet-34-10",
        "robust_acc": 44.83,
    },
    "Sehwag2021Proxy": {
        "type": "robust_at",
        "architecture": "WideResNet-34-10",
        "robust_acc": 60.27,
    },
    "Sehwag2021Proxy_R18": {
        "type": "robust_at",
        "architecture": "ResNet-18",
        "robust_acc": 55.54,
    },
    # --- Architecture Diversity (Standard) ---
    "ResNet50_Standard": {
        "type": "natural",
        "source": "torchvision",
        "tv_name": "resnet50",
        "architecture": "ResNet-50",
        "robust_acc": 0.0
    },
    "ResNet18_Standard": {
        "type": "natural", 
        "source": "torchvision",
        "tv_name": "resnet18",
        "architecture": "ResNet-18",
        "robust_acc": 0.0
    },
    "ResNet101_Standard": {
        "type": "natural", 
        "source": "torchvision",
        "tv_name": "resnet101",
        "architecture": "ResNet-101",
        "robust_acc": 0.0
    },
    "VGG16_Standard": {
        "type": "natural",
        "source": "torchvision",
        "tv_name": "vgg16",
        "architecture": "VGG-16",
        "robust_acc": 0.0
    },
    "DenseNet121_Standard": {
        "type": "natural",
        "source": "torchvision",
        "tv_name": "densenet121",
        "architecture": "DenseNet-121",
        "robust_acc": 0.0
    },
    "ViT_B_16_Standard": {
        "type": "natural",
        "source": "torchvision",
        "tv_name": "vit_b_16",
        "architecture": "ViT-B/16",
        "robust_acc": 0.0
    },
    # --- VLM Targets ---
    "CLIP_ViT_B_32": {
        "type": "vlm",
        "source": "open_clip",
        "clip_model": "ViT-B-32",
        "clip_pretrained": "laion2b_s34b_b79k",
        "architecture": "ViT-B/32",
    },
    # ImageNet Models
    "InceptionV3": {
        "type": "natural",
        "source": "torchvision",  # torchvision weights; no RobustBench entry needed
        "tv_name": "inception_v3",
        "weights": "DEFAULT",
        "architecture": "InceptionV3",
    },
    "ResNet50": {
        "type": "natural",
        # Maps to Standard_R50.pth in RobustBench
        "robustbench_name": "Standard", 
        "architecture": "ResNet-50",
        "robust_acc": 0.0,
    },
    "Engstrom2019Robustness_ImageNet": {
        "type": "robust_at",
        "robustbench_name": "Engstrom2019Robustness",
        "architecture": "ResNet-50",
        "robust_acc": 40.0,
    },
    # --- Madrylab/Salman ε-spectrum ResNet50 (Linf, Exp B sweep) ---
    # Source: https://huggingface.co/madrylab/robust-imagenet-models
    "Salman_eps0.5": {
        "type": "robust_at",
        "source": "madrylab",
        "ckpt_filename": "resnet50_linf_eps0.5.ckpt",
        "architecture": "ResNet-50",
        "eps_train": 0.5 / 255,
    },
    "Salman_eps1.0": {
        "type": "robust_at",
        "source": "madrylab",
        "ckpt_filename": "resnet50_linf_eps1.0.ckpt",
        "architecture": "ResNet-50",
        "eps_train": 1.0 / 255,
    },
    "Salman_eps2.0": {
        "type": "robust_at",
        "source": "madrylab",
        "ckpt_filename": "resnet50_linf_eps2.0.ckpt",
        "architecture": "ResNet-50",
        "eps_train": 2.0 / 255,
    },
    "Salman_eps4.0": {
        "type": "robust_at",
        "source": "madrylab",
        "ckpt_filename": "resnet50_linf_eps4.0.ckpt",
        "architecture": "ResNet-50",
        "eps_train": 4.0 / 255,
    },
    "Salman_eps8.0": {
        "type": "robust_at",
        "source": "madrylab",
        "ckpt_filename": "resnet50_linf_eps8.0.ckpt",
        "architecture": "ResNet-50",
        "eps_train": 8.0 / 255,
    },
    # Extended ImageNet Targets
    "ViT_B_16_ImageNet": {
        "type": "natural",
        "source": "torchvision",
        "tv_name": "vit_b_16",
        "weights": "DEFAULT",
        "architecture": "ViT-B/16",
    },
    "Swin_B_ImageNet": {
        "type": "natural",
        "source": "torchvision",
        "tv_name": "swin_b",
        "weights": "DEFAULT",
        "architecture": "Swin-B",
    },
    "ConvNeXt_B_ImageNet": {
        "type": "natural",
        "source": "torchvision",
        "tv_name": "convnext_base",
        "weights": "DEFAULT",
        "architecture": "ConvNeXt-Base",
    },
    # --- Genuine CIFAR-10 standard surrogates -------------------------------
    # Trained by scripts/train_cifar10_standard.py with the same recipe as this
    # project's CIFAR-100 WRN-28-10. These REPLACE the *_Standard torchvision
    # entries for CIFAR-10 work: those are 1000-class ImageNet models and score
    # 0.0% on CIFAR-10, so they must never be used as CIFAR-10 surrogates.
    "C10_ResNet18": {"type": "natural", "source": "cifar10_local",
                     "ckpt": "c10_resnet18.pt", "architecture": "ResNet-18"},
    "C10_ResNet50": {"type": "natural", "source": "cifar10_local",
                     "ckpt": "c10_resnet50.pt", "architecture": "ResNet-50"},
    "C10_VGG16": {"type": "natural", "source": "cifar10_local",
                  "ckpt": "c10_vgg16.pt", "architecture": "VGG-16"},
    "C10_DenseNet121": {"type": "natural", "source": "cifar10_local",
                        "ckpt": "c10_densenet121.pt",
                        "architecture": "DenseNet-121"},
    # CIFAR-100 standard WRN-28-10, trained in-project (81.07% clean, see
    # models/cifar100/Linf/training_summary.txt). Not a RobustBench id, so it
    # needs its own entry -- without one it silently fell through to a KeyError
    # and the standard side of the CIFAR-100 experiment was never measured.
    "Standard_WRN28_10": {"type": "natural", "source": "cifar100_local",
                          "ckpt": "Standard_WRN28_10.pt",
                          "architecture": "WideResNet-28-10"},
}


def get_model(model_name: str, dataset: str = "cifar10", 
              threat_model: str = "Linf", device: str = "cuda:0",
              model_dir: str = None) -> torch.nn.Module:
    """
    Load a model from RobustBench model zoo.
    """
    # 1. Path Auto-detection
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_assets_models = os.path.join(project_root, "assets", "models")
    
    if model_dir is None:
        # MODEL_ROOT wins over every built-in guess, so the archive can be run
        # without editing paths; the rest are the machines we ran on.
        # Priority: MODEL_ROOT > /autodl-tmp/models > /root/autodl-tmp/models >
        #           /root/robustbench/models > ./assets/models
        env_models = os.environ.get("MODEL_ROOT")
        if env_models and os.path.exists(env_models):
            model_dir = env_models
            print(f"Using MODEL_ROOT: {model_dir}")
        elif os.path.exists("/autodl-tmp/models"):
             model_dir = "/autodl-tmp/models"
        elif os.path.exists("/root/autodl-tmp/models"):
             model_dir = "/root/autodl-tmp/models"
        elif os.path.exists("/root/robustbench/models"):
             model_dir = "/root/robustbench/models"
        elif os.path.exists(local_assets_models):
             model_dir = local_assets_models
        else:
             model_dir = os.path.join(project_root, "models")

    # Special Handling for ImageNet "Standard" -> ResNet50 (Torchvision)
    # This ensures we use the offline-friendly TVModelWrapper logic instead of RobustBench's auto-download
    info = None
    if dataset == "imagenet" and (model_name == "Standard" or model_name == "ResNet50"):
        info = {
            "source": "torchvision",
            "tv_name": "resnet50",
            "weights": "DEFAULT"
        }

    # --- CIFAR-10 redirect ---------------------------------------------------
    # The *_Standard keys are torchvision IMAGENET checkpoints with 1000-way
    # heads. On CIFAR-10 they score 0.0% and their gradients are not tied to the
    # CIFAR-10 task at all. Redirect them to the
    # genuine CIFAR-10 models so every existing script is fixed at once, rather
    # than leaving the trap in place for the next caller.
    if dataset == "cifar10":
        _C10_REDIRECT = {
            "ResNet18_Standard": "C10_ResNet18",
            "ResNet50_Standard": "C10_ResNet50",
            "VGG16_Standard": "C10_VGG16",
            "DenseNet121_Standard": "C10_DenseNet121",
        }
        if model_name in _C10_REDIRECT:
            tgt = _C10_REDIRECT[model_name]
            print(f"[redirect] CIFAR-10: {model_name} -> {tgt} "
                  f"(the torchvision entry is a 1000-class ImageNet model)")
            model_name, info = tgt, None
        elif model_name == "ViT_B_16_Standard":
            raise ValueError(
                "ViT_B_16_Standard on CIFAR-10 is a 1000-class ImageNet "
                "checkpoint (0.0% clean accuracy) and has no CIFAR-10 "
                "replacement in this project. Drop ViT from the CIFAR-10 "
                "standard pool; it remains in the ImageNet pool.")

    # Check if special "torchvision" or "clip" model
    info = info if info else MODEL_INFO.get(model_name, {})

    if info.get("source") == "cifar100_local":
        from robustbench.model_zoo.architectures.wide_resnet import WideResNet

        class _Norm100(nn.Module):
            """CIFAR-100 normalisation wrapper; the network is fed [0,1] images
            like every other surrogate in this project."""

            def __init__(self, net):
                super().__init__()
                self.net = net
                self.register_buffer("m", torch.tensor(
                    [0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1))
                self.register_buffer("s", torch.tensor(
                    [0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1))

            def forward(self, x):
                return self.net((x - self.m) / self.s)

        for base in ["/autodl-tmp/models", "/root/autodl-tmp/models",
                     model_dir, local_assets_models]:
            path = os.path.join(base, "cifar100", "Linf", info["ckpt"])
            if os.path.exists(path):
                break
        else:
            raise FileNotFoundError(f"{model_name}: {info['ckpt']} not found")
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        net = WideResNet(depth=28, num_classes=100, widen_factor=10)
        net.load_state_dict(sd, strict=True)
        print(f"Loaded CIFAR-100 surrogate {model_name} from {path}")
        return _Norm100(net).to(device).eval()

    if info.get("source") == "cifar10_local":
        # Genuine CIFAR-10 models trained by scripts/train_cifar10_standard.py.
        # The checkpoint stores its own clean accuracy, which we re-assert here:
        # a surrogate that cannot classify its dataset silently corrupts every
        # experiment downstream.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_c10train", os.path.join(project_root, "scripts",
                                      "train_cifar10_standard.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for base in ["/autodl-tmp/models", "/root/autodl-tmp/models",
                     model_dir, local_assets_models]:
            path = os.path.join(base, "cifar10", "standard", info["ckpt"])
            if os.path.exists(path):
                break
        else:
            raise FileNotFoundError(
                f"{model_name}: {info['ckpt']} not found. Run "
                f"scripts/train_cifar10_standard.py first.")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = mod.Normalized(mod.BUILDERS[ck["arch"]]())
        net.load_state_dict(ck["state_dict"], strict=True)
        acc = ck.get("clean_acc", 0.0)
        if acc < 80.0:
            raise RuntimeError(f"{model_name}: stored clean_acc={acc:.2f}% is "
                               f"implausibly low; refusing to load.")
        print(f"Loaded CIFAR-10 surrogate {model_name} "
              f"(clean acc {acc:.2f}%) from {path}")
        return net.to(device).eval()

    if info.get("source") == "torchvision":
        from torchvision import models
        import torchvision.transforms as T

        class TVModelWrapper(nn.Module):
            def __init__(self, model, input_size=224):
                super().__init__()
                self.model = model
                self.input_size = input_size
                # ImageNet Normalization
                self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
            def forward(self, x):
                # Resize to model-specific input size (299 for InceptionV3, 224 for others)
                if x.shape[-1] != self.input_size:
                    x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
                x = self.normalize(x)
                return self.model(x)

        try:
            # Check for local weights file first to force offline mode
            tv_name = info.get("tv_name", model_name.lower())
            weights_name = f"{tv_name}_weights.pth"
            
            # Check primary model_dir AND fallback local assets
            paths_to_check = [
                os.path.join(model_dir, "torchvision", weights_name),
                os.path.join(local_assets_models, "torchvision", weights_name)
            ]
            
            local_weights_path = None
            for p in paths_to_check:
                if os.path.exists(p):
                    local_weights_path = p
                    break
            
            # Load Architecture
            model_fn = getattr(models, tv_name)
            model = model_fn(weights=None) # Init empty structure
            
            if local_weights_path:
                print(f"Loading local torchvision weights: {local_weights_path}")
                state_dict = torch.load(local_weights_path)

                # DenseNet checkpoints published before torchvision 0.4 use
                # legacy key names ('...denselayer1.norm.1.weight') that no
                # longer match the module tree ('...denselayer1.norm1.weight').
                # This is the remap torchvision itself applies. Without it,
                # load_state_dict(strict=False) silently leaves the whole
                # network at its random initialisation -- which produces a randomly
                # initialised network that still passes every downstream check.
                if tv_name.startswith("densenet"):
                    pat = re.compile(
                        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\."
                        r"(?:weight|bias|running_mean|running_var))$")
                    for k in list(state_dict.keys()):
                        m = pat.match(k)
                        if m:
                            state_dict[m.group(1) + m.group(2)] = state_dict.pop(k)

                missing, unexpected = model.load_state_dict(state_dict,
                                                            strict=False)
                # Refuse a mostly-unloaded network.
                n_params = len(model.state_dict())
                if len(missing) > 0.05 * n_params:
                    raise RuntimeError(
                        f"{model_name}: {len(missing)}/{n_params} parameters "
                        f"were NOT loaded from {local_weights_path} "
                        f"(unexpected={len(unexpected)}). The checkpoint does "
                        f"not match the architecture; refusing to return a "
                        f"randomly-initialised model. First missing keys: "
                        f"{missing[:5]}")
                if missing or unexpected:
                    print(f"  [warn] {model_name}: missing={len(missing)} "
                          f"unexpected={len(unexpected)} keys (below threshold)")
            else:
                # Online Fallback (or download and save)
                print(f"Downloading torchvision weights for {model_name}...")
                weights_enum = info.get("weights", "DEFAULT")
                model = model_fn(weights=weights_enum)
            
            # Wrap for input preprocessing
            # InceptionV3 requires 299x299 input, others use 224x224
            input_size = 299 if tv_name == "inception_v3" else 224
            model = TVModelWrapper(model, input_size=input_size)
                
        except Exception as e:
            print(f"Error loading torchvision model {model_name}: {e}")
            raise e
            
    elif info.get("source") == "madrylab":
        # Madrylab/Salman ε-spectrum ResNet50 checkpoints (Exp B sweep)
        from .madrylab_loader import load_madrylab_resnet50
        ckpt_filename = info["ckpt_filename"]
        # Look in standard imagenet/Linf directory
        candidate_paths = [
            os.path.join(model_dir, "imagenet", "Linf", ckpt_filename),
            os.path.join(local_assets_models, "imagenet", "Linf", ckpt_filename),
        ]
        ckpt_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                ckpt_path = p
                break
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Madrylab ckpt {ckpt_filename} not found in any of: "
                f"{candidate_paths}"
            )
        model = load_madrylab_resnet50(ckpt_path, device=device)
        # Already moved to device + eval mode in load_madrylab_resnet50
        return model

    elif info.get("source") == "open_clip":
        import open_clip
        class CLIPWrapper(nn.Module):
            def __init__(self, model, device, dataset_name="cifar10"):
                super().__init__()
                self.model = model
                self.device = device
                # CIFAR-10 Classes
                self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                # Pre-compute text embeddings
                self.text_features = self._precompute_text_features()
                # Normalization (CLIP specific)
                self.normalize = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                           std=(0.26862954, 0.26130258, 0.27577711))

            def _precompute_text_features(self):
                tokenizer = open_clip.get_tokenizer(info["clip_model"])
                texts = [f"a photo of a {c}" for c in self.classes]
                text_tokens = tokenizer(texts).to(self.device)
                with torch.no_grad():
                    text_features = self.model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                return text_features

            def forward(self, x):
                # Resize 32x32 -> 224x224
                if x.shape[-1] < 224:
                    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                # CLIP Normalization
                x = self.normalize(x)
                
                # Image Features
                image_features = self.model.encode_image(x)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Zero-shot Logits (Cosine Similarity * 100 for scaling)
                logits = 100.0 * image_features @ self.text_features.T
                return logits

        try:
            # 1. Check for Manual Download File (assets/models/clip/{ViT-B-32.bin, open_clip_pytorch_model.bin})
            # Check primary model_dir AND fallback local assets
            paths_to_check = [
                os.path.join(model_dir, "clip", "ViT-B-32.bin"),
                os.path.join(model_dir, "clip", "open_clip_pytorch_model.bin"),
                os.path.join(local_assets_models, "clip", "ViT-B-32.bin")
            ]
            
            local_clip_path = None
            for p in paths_to_check:
                if os.path.exists(p):
                    local_clip_path = p
                    break
            
            pretrained_source = info["clip_pretrained"] 
            
            if local_clip_path:
                print(f"Loading manual CLIP weights: {local_clip_path}")
                pretrained_source = local_clip_path
            else:
                # 2. Fallback to OpenCLIP Cache (in model_dir)
                if os.path.exists(os.path.join(model_dir, "clip")):
                    os.environ["OPEN_CLIP_CACHE_DIR"] = os.path.join(model_dir, "clip")
                
            model, _, _ = open_clip.create_model_and_transforms(
                info["clip_model"], pretrained=pretrained_source
            )
            model = model.to(device) # Move to device BEFORE wrapping to enable text encoding
            
            # Wrap
            import torchvision.transforms as T # Ensure T is available
            model = CLIPWrapper(model, device)
            
        except ImportError:
            raise ImportError("open_clip not installed. Run `pip install open_clip_torch`")
        except Exception as e:
            print(f"Error loading CLIP model {model_name}: {e}")
            raise e
            
    else:
        # Default to RobustBench
        # Handle Aliasing (e.g. ResNet50 -> Standard)
        rb_name = info.get("robustbench_name", model_name)
        
        try:
            model = load_model(
                model_name=rb_name,
                dataset=dataset,
                threat_model=threat_model,
                model_dir=model_dir
            )
        except Exception as e:
            print(f"Failed to load RobustBench model {rb_name} from {model_dir}")
            raise e
        
    model = model.to(device)
    model.eval()
    return model


def get_model_info(model_name: str) -> dict:
    """Get metadata for a model."""
    return MODEL_INFO.get(model_name, {"type": "unknown", "architecture": "unknown"})


def list_available_models() -> list:
    """List all registered models."""
    return list(MODEL_INFO.keys())
