"""
Data loading utilities for transfer attack experiments.
Wraps RobustBench data loading with consistent interface.
"""
import torch
import os
from robustbench.data import load_cifar10, load_cifar100


def load_dataset(dataset: str = "cifar10", n_examples: int = 1000, 
                 data_dir: str = None) -> tuple:
    """
    Load dataset for evaluation.
    
    Args:
        dataset: One of "cifar10", "cifar100"
        n_examples: Number of examples to load
        data_dir: Directory containing dataset (auto-detected if None)
        
    Returns:
        (x_test, y_test): Tensor of images and labels
    """
    # Auto-detect data_dir: ./assets/data next to the project, else the
    # server paths below. DATA_ROOT overrides both.
    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_assets = os.path.join(project_root, "assets", "data")
        
        # DATA_ROOT wins over every built-in guess, so the archive can be run
        # without editing paths; the rest are the machines we ran on.
        env_root = os.environ.get("DATA_ROOT")
        autodl_data = "/autodl-tmp/data"
        autodl_root_data = "/root/autodl-tmp/data"

        if env_root:
            cand = os.path.join(env_root, "ImageNet") if dataset == "imagenet" \
                else env_root
            if os.path.exists(cand):
                data_dir = cand
                print(f"Using DATA_ROOT: {data_dir}")

        if data_dir is not None:
            pass
        elif dataset == "imagenet":
            # 1. Check ImageNet-specific paths, in priority order
            if os.path.exists(os.path.join(autodl_root_data, "ImageNet")):
                data_dir = os.path.join(autodl_root_data, "ImageNet")
            elif os.path.exists(os.path.join(autodl_data, "ImageNet")):
                data_dir = os.path.join(autodl_data, "ImageNet")
            elif os.path.exists("/root/robustbench/data/imagenet"):
                data_dir = "/root/robustbench/data/imagenet"
            elif os.path.exists(os.path.join(local_assets, "ImageNet")):
                # RobustBench appends '/val', so this must end in ImageNet.
                data_dir = os.path.join(local_assets, "ImageNet")
            else:
                data_dir = "/root/robustbench/data/imagenet"

            print(f"ImageNet data_dir: {data_dir}")

        else: 
            # 2. CIFAR-10/100 (Prioritize local assets)
            if os.path.exists(local_assets) and os.path.exists(os.path.join(local_assets, "CIFAR-10")):
                 data_dir = local_assets
            elif os.path.exists(autodl_data):
                 data_dir = autodl_data
                 print(f"Found AutoDL data at {data_dir}")
            else:
                # Fallback to server defaults
                if dataset == "cifar10":
                    data_dir = "/root/robustbench/data/CIFAR-10"
                elif dataset == "cifar100":
                    data_dir = "/root/robustbench/data/CIFAR-100"
                else:
                    data_dir = "/root/robustbench/data"
    
    if dataset == "cifar10":
        x_test, y_test = load_cifar10(n_examples=n_examples, data_dir=data_dir)
    elif dataset == "cifar100":
        x_test, y_test = load_cifar100(n_examples=n_examples, data_dir=data_dir)
    elif dataset == "imagenet":
        from robustbench.data import load_imagenet
        x_test, y_test = load_imagenet(n_examples=n_examples, data_dir=data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return x_test, y_test


def get_dataloader(x: torch.Tensor, y: torch.Tensor, 
                   batch_size: int = 64, shuffle: bool = False):
    """Create a DataLoader from tensors."""
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=8, pin_memory=True
    )
    return loader
