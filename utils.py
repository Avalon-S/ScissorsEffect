"""
Utility functions for transfer attack experiments.
"""
import os
import random
import numpy as np
import torch
import pandas as pd
from datetime import datetime


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cuda:0") -> torch.device:
    """Get torch device, fallback to CPU if CUDA unavailable."""
    if torch.cuda.is_available() and device_str.startswith("cuda"):
        return torch.device(device_str)
    return torch.device("cpu")


class Logger:
    """Simple logger for experiment results."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.records = []
        
    def log(self, source: str, target: str, attack: str, 
            clean_acc: float, robust_acc: float, asr: float,
            runtime: float = 0.0, n_examples: int = 0, **kwargs):
        """Log a single evaluation result."""
        record = {
            "source": source,
            "target": target,
            "attack": attack,
            "clean_acc": clean_acc,
            "robust_acc": robust_acc,
            "asr": asr,
            "runtime_sec": runtime,
            "n_examples": n_examples,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.records.append(record)
        print(f"[{attack}] {source} -> {target}: ASR={asr:.2f}%, "
              f"RobAcc={robust_acc:.2f}%, Time={runtime:.2f}s")
        
    def save(self, filename: str = None):
        """Save results to CSV."""
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.results_dir, filename)
        df = pd.DataFrame(self.records)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        return filepath
    
    def to_pivot_table(self) -> pd.DataFrame:
        """Convert results to pivot table (Attack x Target)."""
        df = pd.DataFrame(self.records)
        if len(df) == 0:
            return df
        pivot = df.pivot_table(
            values="asr", 
            index=["attack", "source"], 
            columns="target",
            aggfunc="mean"
        )
        return pivot
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return results as DataFrame."""
        return pd.DataFrame(self.records)


def compute_metrics(x_adv: torch.Tensor, x_clean: torch.Tensor, 
                    y: torch.Tensor, model: torch.nn.Module,
                    device: torch.device) -> dict:
    """Compute attack metrics (in batches to avoid OOM)."""
    model.eval()
    
    clean_hits = 0
    robust_hits = 0
    total = x_adv.size(0)
    batch_size = 100 # Conservative batch size for eval
    
    linf_norms = []
    l2_norms = []
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            x_adv_batch = x_adv[i:i+batch_size].to(device)
            x_clean_batch = x_clean[i:i+batch_size].to(device)
            y_batch = y[i:i+batch_size].to(device)
            
            # Predictions
            pred_adv = model(x_adv_batch).argmax(dim=1)
            pred_clean = model(x_clean_batch).argmax(dim=1)
            
            # Accumulate counts
            clean_hits += (pred_clean == y_batch).sum().item()
            robust_hits += (pred_adv == y_batch).sum().item()
            
            # Perturbation stats
            delta = (x_adv_batch - x_clean_batch).abs()
            linf_norms.append(delta.view(delta.size(0), -1).max(dim=1)[0].cpu())
            l2_norms.append(delta.view(delta.size(0), -1).norm(p=2, dim=1).cpu())
            
    # Calculate averages
    clean_acc = (clean_hits / total) * 100
    robust_acc = (robust_hits / total) * 100
    asr = 100 - robust_acc
    
    all_linf = torch.cat(linf_norms)
    all_l2 = torch.cat(l2_norms)
    
    return {
        "clean_acc": clean_acc,
        "robust_acc": robust_acc,
        "asr": asr,
        # linf is the worst case over the dataset, l2 the mean.
        "linf_norm": all_linf.max().item(),
        "l2_norm": all_l2.mean().item(),
    }
