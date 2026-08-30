"""
Attack wrappers for transfer attack experiments.
Uses torchattacks as the backend.
"""
import torch
import torchattacks


def get_attack(attack_name: str, model: torch.nn.Module, 
               eps: float = 8/255, steps: int = 10, alpha: float = 2/255,
               **kwargs):
    """
    Get an attack instance from torchattacks.
    
    Args:
        attack_name: Attack identifier (ifgsm, mifgsm, nifgsm, dim, vmi, tim)
        model: Source model for attack
        eps: Maximum perturbation (Linf)
        steps: Number of attack steps
        alpha: Step size
        **kwargs: Additional attack-specific parameters
        
    Returns:
        torchattacks attack instance
    """
    attack_name = attack_name.lower()
    
    if attack_name in ["ifgsm", "bim", "i-fgsm"]:
        # Iterative FGSM (BIM)
        attack = torchattacks.BIM(
            model, eps=eps, alpha=alpha, steps=steps
        )
        
    elif attack_name in ["mifgsm", "mi-fgsm"]:
        # Momentum Iterative FGSM
        decay = kwargs.get("momentum", 1.0)
        attack = torchattacks.MIFGSM(
            model, eps=eps, alpha=alpha, steps=steps, decay=decay
        )
        
    elif attack_name in ["nifgsm", "ni-fgsm"]:
        # Nesterov Iterative FGSM
        decay = kwargs.get("momentum", 1.0)
        attack = torchattacks.NIFGSM(
            model, eps=eps, alpha=alpha, steps=steps, decay=decay
        )
        
    elif attack_name in ["dim", "difgsm", "di-fgsm"]:
        # Diverse Input FGSM
        decay = kwargs.get("momentum", 1.0)
        resize_rate = kwargs.get("resize_rate", 0.9)
        diversity_prob = kwargs.get("diversity_prob", 0.5)
        attack = torchattacks.DIFGSM(
            model, eps=eps, alpha=alpha, steps=steps, decay=decay,
            resize_rate=resize_rate, diversity_prob=diversity_prob
        )
        
    elif attack_name in ["tim", "tifgsm", "ti-fgsm"]:
        # Translation Invariant FGSM
        decay = kwargs.get("momentum", 1.0)
        kernel_name = kwargs.get("kernel_name", "gaussian")
        attack = torchattacks.TIFGSM(
            model, eps=eps, alpha=alpha, steps=steps, decay=decay,
            kernel_name=kernel_name
        )
        
    elif attack_name in ["vmi", "vmifgsm", "vmi-fgsm"]:
        # Variance-tuning MI-FGSM  
        decay = kwargs.get("momentum", 1.0)
        N = kwargs.get("sample_num", 20)
        beta = kwargs.get("beta", 1.5)
        attack = torchattacks.VMIFGSM(
            model, eps=eps, alpha=alpha, steps=steps, decay=decay,
            N=N, beta=beta
        )
        
    elif attack_name in ["sinifgsm", "si-ni-fgsm", "sim"]:
        # Scale-Invariant NI-FGSM
        decay = kwargs.get("momentum", 1.0)
        m = kwargs.get("scale_copies", 5)
        attack = torchattacks.SINIFGSM(
            model, eps=eps, alpha=alpha, steps=steps, decay=decay, m=m
        )
        
    elif attack_name == "pgd":
        # Standard PGD (for sanity check on source model)
        attack = torchattacks.PGD(
            model, eps=eps, alpha=alpha, steps=steps, random_start=True
        )
        
    else:
        raise ValueError(f"Unknown attack: {attack_name}. "
                         f"Available: ifgsm, mifgsm, nifgsm, dim, tim, vmi, sinifgsm, pgd")
    
    return attack


def generate_adversarial(attack, 
                         x: torch.Tensor, y: torch.Tensor,
                         batch_size: int = 64) -> torch.Tensor:
    """
    Generate adversarial examples in batches.
    
    Args:
        attack: torchattacks.Attack instance
        x: Clean images tensor [N, C, H, W]
        y: Labels tensor [N]
        batch_size: Batch size for generation
        
    Returns:
        Adversarial examples tensor [N, C, H, W]
    """
    n = x.size(0)
    x_adv_list = []
    
    for i in range(0, n, batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        x_adv_batch = attack(x_batch, y_batch)
        x_adv_list.append(x_adv_batch.cpu())
    
    return torch.cat(x_adv_list, dim=0)


# Attack metadata for logging
ATTACK_INFO = {
    "ifgsm": {"name": "I-FGSM", "paper": "Kurakin 2016"},
    "mifgsm": {"name": "MI-FGSM", "paper": "Dong 2018"},
    "nifgsm": {"name": "NI-FGSM", "paper": "Lin 2020"},
    "dim": {"name": "DI-FGSM", "paper": "Xie 2019"},
    "tim": {"name": "TI-FGSM", "paper": "Dong 2019"},
    "vmi": {"name": "VMI-FGSM", "paper": "Wang 2021"},
    "sinifgsm": {"name": "SI-NI-FGSM", "paper": "Lin 2020"},
    "pgd": {"name": "PGD", "paper": "Madry 2018"},
}
