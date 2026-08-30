"""
Source-Adaptive p (SAP) Module

Automatically estimates optimal p for DI-FGSM based on source model properties.

Two estimators:
1. LGC (Local Gradient Consistency): measures direction consistency under perturbation
2. HF Ratio: measures high-frequency gradient energy

Usage:
    from attacks.sap import estimate_p_lgc, estimate_p_hf, SAP_DI_FGSM
"""
import torch
import torch.nn.functional as F
import numpy as np
import random


def estimate_p_lgc(model, x, y, device, K=5, sigma=1/255, alpha=1.5):
    """
    Estimate p using Local Gradient Consistency (LGC).

    Matches paper Eq. 1 and Algorithm 1:
        LGC(x) = (1/K) Σ cos(∇_x L(x), ∇_{x'_k} L(x'_k))
    where x'_k = x + ξ_k, ξ_k ~ U(-sigma, sigma)

    - LGC ≈ 1: gradients are consistent (smooth/robust) → low p
    - LGC < 0.92: gradients are divergent (noisy/standard) → high p

    Args:
        model: source model
        x: input tensor [B, C, H, W]
        y: labels [B]
        K: number of perturbation samples (default 5, matching Algorithm 1)
        sigma: perturbation scale (default 1/255, matching ε_chk in paper)
        alpha: scaling factor for p_hat

    Returns:
        p_hat: estimated p values [B]
        lgc: LGC values [B]
    """
    x = x.to(device)
    y = y.to(device)
    B = x.size(0)

    # Step 1: Compute clean gradient (reference)
    x_clean = x.clone().detach().requires_grad_(True)
    logits_clean = model(x_clean)
    loss_clean = F.cross_entropy(logits_clean, y, reduction='sum')
    model.zero_grad()
    loss_clean.backward()
    g_clean = x_clean.grad.clone().detach()  # [B, C, H, W]

    # Step 2: Compute K perturbed gradients and cosine similarity with clean
    cos_sum = torch.zeros(B, device=device)
    g_clean_flat = g_clean.view(B, -1)

    for _ in range(K):
        # Uniform noise: U(-sigma, sigma), matching paper definition
        noise = (torch.rand_like(x) * 2 - 1) * sigma
        x_pert = (x + noise).clamp(0, 1).detach().requires_grad_(True)

        # Forward and backward
        logits = model(x_pert)
        loss = F.cross_entropy(logits, y, reduction='sum')
        model.zero_grad()
        loss.backward()

        g_pert = x_pert.grad.clone().detach()
        g_pert_flat = g_pert.view(B, -1)

        # Cosine similarity per sample
        cos_sim = F.cosine_similarity(g_clean_flat, g_pert_flat, dim=1)  # [B]
        cos_sum += cos_sim

    # Compute LGC: average cosine similarity
    lgc = cos_sum / K  # [B]

    # Map to p_hat: low LGC → high p
    p_hat = alpha * (1 - lgc)
    p_hat = p_hat.clamp(0, 1)

    return p_hat, lgc


def estimate_p_hf(model, x, y, device, hf_threshold=0.5, beta=1.5):
    """
    Estimate p using High-Frequency gradient energy ratio.
    
    HF Ratio = E_high / E_total
    - High HF ratio: gradient is "spiky" (standard) → high p
    - Low HF ratio: gradient is smooth (robust) → low p
    
    Args:
        model: source model
        x: input tensor [B, C, H, W]
        y: labels [B]
        hf_threshold: fraction of max freq to consider as "high"
        beta: scaling factor
        
    Returns:
        p_hat: estimated p values [B]
        hf_ratio: HF ratio values [B]
    """
    x = x.to(device).requires_grad_(True)
    y = y.to(device)
    B, C, H, W = x.shape
    
    # Compute gradient
    logits = model(x)
    loss = F.cross_entropy(logits, y, reduction='sum')
    model.zero_grad()
    loss.backward()
    grad = x.grad.detach()  # [B, C, H, W]
    
    # Compute HF ratio per sample
    hf_ratios = []
    
    cy, cx = H // 2, W // 2
    y_coords = torch.arange(H, device=device) - cy
    x_coords = torch.arange(W, device=device) - cx
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    dist = torch.sqrt(yy.float()**2 + xx.float()**2)
    max_freq = np.sqrt(cy**2 + cx**2)
    threshold = hf_threshold * max_freq
    hf_mask = dist > threshold
    
    for b in range(B):
        hf_total = 0.0
        total_energy = 0.0
        
        for c in range(C):
            g = grad[b, c]
            fft = torch.fft.fft2(g)
            fft_shifted = torch.fft.fftshift(fft)
            magnitude = torch.abs(fft_shifted)
            
            energy = (magnitude ** 2).sum().item()
            hf_energy = (magnitude[hf_mask] ** 2).sum().item()
            
            total_energy += energy
            hf_total += hf_energy
        
        hf_ratio = hf_total / (total_energy + 1e-12)
        hf_ratios.append(hf_ratio)
    
    hf_ratio = torch.tensor(hf_ratios, device=device)
    
    # Map to p_hat: high HF → high p
    # Normalize to [0, 1] range (empirically HF ratio is usually in [0.2, 0.8])
    p_hat = beta * (hf_ratio - 0.2) / 0.6
    p_hat = p_hat.clamp(0, 1)
    
    return p_hat, hf_ratio


def di_transform(x, resize_rate=0.9):
    """DI transform: random resize + pad."""
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    
    rnd = random.randint(img_resize, img_size)
    x = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
    
    pad_top = random.randint(0, img_size - rnd)
    pad_left = random.randint(0, img_size - rnd)
    pad_bottom = img_size - rnd - pad_top
    pad_right = img_size - rnd - pad_left
    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return x


class SAP_DI_FGSM:
    """
    DI-FGSM with Source-Adaptive p (SAP).
    
    Automatically estimates p per-image based on source gradient properties.
    """
    
    def __init__(self, model, eps, alpha, steps, device,
                 estimator='lgc', K=5, sigma=1/255):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device
        self.estimator = estimator
        self.K = K
        self.sigma = sigma
        
        self.p_hat_history = []
    
    def estimate_p(self, x, y):
        """Estimate p for each sample in batch."""
        if self.estimator == 'lgc':
            p_hat, proxy = estimate_p_lgc(
                self.model, x, y, self.device, 
                K=self.K, sigma=self.sigma
            )
        elif self.estimator == 'hf':
            p_hat, proxy = estimate_p_hf(
                self.model, x, y, self.device
            )
        else:
            raise ValueError(f"Unknown estimator: {self.estimator}")
        
        return p_hat, proxy
    
    def __call__(self, x, y):
        x = x.clone().detach().to(self.device)
        y = y.to(self.device)
        B = x.size(0)
        
        # Estimate p per-image (static, computed once before attack)
        p_hat, _ = self.estimate_p(x, y)
        self.p_hat_history.extend(p_hat.cpu().numpy().tolist())
        
        x_adv = x.clone()
        m = torch.zeros_like(x)
        
        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            
            # Apply transform per-sample based on its p_hat (Graph Preserved)
            x_t_list = []
            for i in range(B):
                if random.random() < p_hat[i].item():
                    x_t_list.append(di_transform(x_adv[i:i+1]))
                else:
                    x_t_list.append(x_adv[i:i+1])
            x_t = torch.cat(x_t_list, dim=0)
            
            # Compute loss
            logits = self.model(x_t)
            loss = F.cross_entropy(logits, y, reduction='sum')
            
            # Compute gradient through the transformation
            grad = torch.autograd.grad(loss, x_adv)[0]
            
            # Normalize
            grad_norm = grad.abs().sum(dim=(1,2,3), keepdim=True) + 1e-12
            g_hat = grad / grad_norm
            
            # Momentum
            m = m + g_hat
            
            # Update
            x_adv = x_adv.detach() + self.alpha * m.sign()
            delta = torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
    
    def get_p_hat_stats(self):
        """Get statistics of estimated p values."""
        if not self.p_hat_history:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        arr = np.array(self.p_hat_history)
        return {
            "mean": arr.mean(),
            "std": arr.std(),
            "min": arr.min(),
            "max": arr.max(),
        }
    
    def reset_history(self):
        self.p_hat_history = []


class CG_DI:
    """
    CG-DI (Consistency-Guided DI): Algorithm 1 from paper.

    Binary decision rule based on LGC threshold:
        p* = 0   if LGC > τ   (robust-like, diversity harmful)
        p* = 0.8 if LGC ≤ τ   (standard-like, diversity beneficial)

    Default τ = 0.92, matching paper Algorithm 1.
    """

    def __init__(self, model, eps, alpha, steps, device,
                 calibration_data=None, batch_size=32, thresholds=None):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device
        self.thresholds = thresholds  # (tau,) single threshold or (low, high) for compat
        self.calibrated = False

        if calibration_data is not None:
            self.calibrate(*calibration_data)

        self.p_hat_history = []

    def calibrate(self, x_cal, y_cal):
        """Calibrate using Algorithm 1: LGC probe + binary decision.

        1. Compute LGC via cosine similarity (Eq. 1)
        2. Binary decision: p* = 0 if LGC > τ else 0.8
        """
        self.model.eval()
        x_cal = x_cal.to(self.device)
        y_cal = y_cal.to(self.device)

        # Compute LGC scores (cosine similarity, K=5, uniform noise)
        _, lgc_scores = estimate_p_lgc(self.model, x_cal, y_cal, self.device, K=5)
        lgc_scores = lgc_scores.cpu().numpy()

        self.mean_lgc = np.mean(lgc_scores)
        self.std_lgc = np.std(lgc_scores)

        # Single threshold τ, matching Algorithm 1
        if self.thresholds is not None:
            # Support both (tau,) and (low, high) for backward compatibility
            if isinstance(self.thresholds, (list, tuple)) and len(self.thresholds) == 2:
                # Legacy: take midpoint of (low, high) as single τ
                tau = (self.thresholds[0] + self.thresholds[1]) / 2
            else:
                tau = float(self.thresholds)
        else:
            tau = 0.92  # Paper default

        # Binary decision (Algorithm 1, line 7)
        if self.mean_lgc > tau:
            self.p_fixed = 0.0
            self.model_type = "robust"
        else:
            self.p_fixed = 0.8
            self.model_type = "standard"

        print(f"[CG-DI] LGC={self.mean_lgc:.3f}, τ={tau}, p*={self.p_fixed} ({self.model_type})")
        self.calibrated = True

    def get_p_hat(self, x, y):
        """Return the calibrated p value for all samples in batch.
        
        After calibration, we use a single p value for all samples (p_fixed),
        determined by the global HF level of the source model.
        """
        B = x.size(0)
        
        if not self.calibrated:
            # Default to 0.5 if not calibrated
            return torch.ones(B, device=self.device) * 0.5, None
        
        # Use the globally calibrated p_fixed for all samples
        p_hat = torch.ones(B, device=self.device) * self.p_fixed
        
        return p_hat, None

    def __call__(self, x, y):
        x = x.clone().detach().to(self.device)
        y = y.to(self.device)
        B = x.size(0)
        
        # 1. Estimate p
        p_hat, _ = self.get_p_hat(x, y)
        self.p_hat_history.extend(p_hat.cpu().numpy().tolist())
        
        # 2. DI-FGSM with dynamic p
        x_adv = x.clone()
        m = torch.zeros_like(x)
        
        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            
            # Apply transform per-sample based on its p_hat (Graph Preserved)
            x_t_list = []
            for i in range(B):
                if random.random() < p_hat[i].item():
                    x_t_list.append(di_transform(x_adv[i:i+1]))
                else:
                    x_t_list.append(x_adv[i:i+1])
            x_t = torch.cat(x_t_list, dim=0)
            
            # Compute loss
            logits = self.model(x_t)
            loss = F.cross_entropy(logits, y, reduction='sum')
            
            # Compute gradient through the transformation
            grad = torch.autograd.grad(loss, x_adv)[0]
            
            # Normalize
            grad_norm = grad.abs().sum(dim=(1,2,3), keepdim=True) + 1e-12
            g_hat = grad / grad_norm
            
            # Momentum
            m = m + g_hat
            
            # Update
            x_adv = x_adv.detach() + self.alpha * m.sign()
            delta = torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
    
    def get_p_hat_stats(self):
        """Get statistics of estimated p values."""
        if not self.p_hat_history:
            return {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5}
        arr = np.array(self.p_hat_history)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    
    def reset_history(self):
        self.p_hat_history = []
