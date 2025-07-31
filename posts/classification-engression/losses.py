import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# https://chatgpt.com/c/685b3d6c-5c34-8009-8484-196a0487b13e

# TODO energy score
# brier? 


def brier_loss(logits: torch.Tensor, targets: torch.Tensor):
    """
    Computes the Brier score loss for classification.
    
    Args:
        logits: Tensor of shape (B, C), raw model outputs.
        targets: Tensor of shape (B,), integer class labels.
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss or per-sample loss depending on reduction.
    """
    probs = F.softmax(logits, dim=-1)              # (B, C)
    onehot = F.one_hot(targets, num_classes=logits.size(-1)).float()  # (B, C)
    loss = (probs - onehot).pow(2).sum(dim=-1)     # (B,)
    return loss.mean()


def energy_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lamb: float = 0.5,
    p = 2,
    beta: float = 1.0,
    return_components: bool = False,
):
    """
    Energy score loss for classification.
    
    Args:
        logits: Tensor of shape (B, m, C), unnormalized outputs.
        targets: Tensor of shape (B,), integer labels.
    Returns:
        Scalar loss (or tuple if return_components=True).
    """
    B, m, C = logits.shape

    targets = F.one_hot(targets, num_classes = C).float()  # Convert targets to one-hot encoding
    targets = rearrange(targets, 'b ... -> b 1 (...)')
    preds   = rearrange(logits,   'b m ... -> b m (...)')

    # Term 1: the absolute error between the predicted and true values
    term1 = torch.linalg.vector_norm(preds - targets, ord = p, dim = 2).pow(beta).mean()

    # Term 2: pairwise absolute differences between the predicted values
    term2 = torch.tensor(0.0, device = preds.device, dtype = preds.dtype)

    if m > 1:
        # cdist is convenient. The result shape before sum is (n, m, m).
        pairwise_l1_dists = torch.cdist(preds, preds, p = p).pow(beta).mean() * m / (m - 1)
        term2 = - lamb * pairwise_l1_dists

    if return_components:
        return term1 + term2, term1, term2
    
    return term1 + term2


def js_js(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lamb: float = 0.5,
    tau: float = 1.0,
    eps: float = 1e-8,
    return_components: bool = False,
):
    B, m, C = logits.shape
    p = F.softmax(logits, dim=-1)
    logp = torch.log(p + eps)
    onehot = F.one_hot(targets, C).float()
    onehot = repeat(onehot, 'b c -> b m c', m=m)

    # Accuracy: JS(p || e_y)
    m_py = 0.5 * (p + onehot)
    logm_py = torch.log(m_py + eps)
    js_label = 0.5 * ((p * (logp - logm_py)).sum(-1) + (onehot * (torch.log(onehot + eps) - logm_py)).sum(-1))
    acc_term = js_label.mean()

    # Diversity: JS(p_i || p_j)
    p_i, p_j = rearrange(p, 'b i c -> b i 1 c'), rearrange(p, 'b j c -> b 1 j c')
    logp_i, logp_j = rearrange(logp, 'b i c -> b i 1 c'), rearrange(logp, 'b j c -> b 1 j c')
    m_ij = 0.5 * (p_i + p_j)
    logm = torch.log(m_ij + eps)

    js_ij = 0.5 * ((p_i * (logp_i - logm)).sum(-1) + (p_j * (logp_j - logm)).sum(-1))
    k_ij = torch.exp(-0.5 * js_ij / (tau**2 + eps))
    div_term = k_ij.mean()

    loss = acc_term + lamb * div_term
    return (loss, acc_term, lamb * div_term) if return_components else loss


def ce_sym_kl(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lamb: float = 0.5,
    tau: float = 1.0,
    eps: float = 1e-8,
    return_components: bool = False,
):
    """
    Kernel-based classification loss with symmetric KL and RBF kernel.
    
    Args:
        logits: Tensor of shape (B, m, C), unnormalized outputs.
        targets: Tensor of shape (B,), integer labels.
    Returns:
        Scalar loss (or tuple if return_components=True).
    """
    B, m, C = logits.shape

    # ========== ACCURACY TERM ==========
    # Flatten for cross-entropy
    logits_flat = rearrange(logits, 'b m c -> (b m) c')
    targets_rep = repeat(targets, 'b -> (b m)', m=m)
    acc_term = F.cross_entropy(logits_flat, targets_rep, reduction='mean')

    # ========== DIVERSITY TERM ==========
    logp = F.log_softmax(logits, dim=-1)  # (B, m, C)

    # Broadcast for pairwise KLs using log_target=True
    logp_i = rearrange(logp, 'b i c -> b i 1 c')  # (B, m, 1, C)
    logp_j = rearrange(logp, 'b j c -> b 1 j c')  # (B, 1, m, C)

    # Symmetric KL divergence
    kl_ij = F.kl_div(logp_j, logp_i, log_target=True, reduction='none').sum(-1)
    kl_ji = F.kl_div(logp_i, logp_j, log_target=True, reduction='none').sum(-1)
    sym_kl = kl_ij + kl_ji  # (B, m, m)

    # RBF kernel
    k_ij = torch.exp(-0.5 * sym_kl / (tau**2 + eps))  # (B, m, m)
    div_term = k_ij.mean()

    loss = acc_term + lamb * div_term
    if return_components:
        return loss, acc_term, lamb * div_term
    else:
        return loss


def ce_js(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lamb: float = 0.5,
    tau: float = 1.0,
    eps: float = 1e-8,
    return_components: bool = False,
):
    B, m, C = logits.shape
    logits_flat = rearrange(logits, 'b m c -> (b m) c')
    targets_rep = repeat(targets, 'b -> (b m)', m=m)
    acc_term = F.cross_entropy(logits_flat, targets_rep, reduction='mean')

    p = F.softmax(logits, dim=-1)
    logp = torch.log(p + eps)

    p_i, p_j = rearrange(p, 'b i c -> b i 1 c'), rearrange(p, 'b j c -> b 1 j c')
    logp_i, logp_j = rearrange(logp, 'b i c -> b i 1 c'), rearrange(logp, 'b j c -> b 1 j c')
    m_ij = 0.5 * (p_i + p_j)
    logm = torch.log(m_ij + eps)

    js_ij = 0.5 * ((p_i * (logp_i - logm)).sum(-1) + (p_j * (logp_j - logm)).sum(-1))
    k_ij = torch.exp(-0.5 * js_ij / (tau**2 + eps))
    div_term = k_ij.mean()

    loss = acc_term + lamb * div_term
    return (loss, acc_term, lamb * div_term) if return_components else loss


def ce_sym_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lamb: float = 0.3,
    eps: float = 1e-8,
    return_components: bool = False,
):
    B, m, C = logits.shape
    logits_flat = rearrange(logits, 'b m c -> (b m) c')
    targets_rep = repeat(targets, 'b -> (b m)', m=m)
    acc_term = F.cross_entropy(logits_flat, targets_rep, reduction='mean')

    p = F.softmax(logits, dim=-1)
    logp = torch.log(p + eps)

    p_i, p_j = rearrange(p, 'b i c -> b i 1 c'), rearrange(p, 'b j c -> b 1 j c')
    logp_i, logp_j = rearrange(logp, 'b i c -> b i 1 c'), rearrange(logp, 'b j c -> b 1 j c')

    ce_ij = -(p_j * logp_i).sum(-1)
    ce_ji = -(p_i * logp_j).sum(-1)
    sym_ce = 0.5 * (ce_ij + ce_ji)
    div_term = -sym_ce.mean()

    loss = acc_term + lamb * div_term
    return (loss, acc_term, lamb * div_term) if return_components else loss

    

# Baselines
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0, # 1.0, 2.0
    eps: float = 1e-8,
):
    """
    Focal loss: reduces weight of easy examples.
    Assumes logits shape (B, C), targets shape (B,)
    """
    log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
    probs = log_probs.exp()
    target_log_probs = log_probs[torch.arange(logits.size(0)), targets]
    target_probs = probs[torch.arange(logits.size(0)), targets]

    focal_weight = (1 - target_probs).clamp(min=eps).pow(gamma)
    loss = -focal_weight * target_log_probs
    return loss.mean()



def confidence_penalty_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 0.1, # 0.1, 0.5, 1.0
):
    """
    Cross entropy with confidence penalty on low entropy predictions.
    Assumes logits shape (B, C), targets shape (B,)
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='mean')
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1).mean()
    return ce_loss - beta * entropy