from typing import Optional

import torch
import torch.nn.functional as F


def compute_attention_output(
    query: torch.Tensor,   # [n_q, d_head]
    key: torch.Tensor,     # [n_kv, d_head]
    value: torch.Tensor,   # [n_kv, d_head]
) -> torch.Tensor:
    """Scaled dot-product attention. Returns [n_q, d_head]."""
    scale = query.size(-1) ** -0.5
    scores = (query @ key.T) * scale
    attn = F.softmax(scores, dim=-1)
    return attn @ value


def attention_output_relative_error(
    q: torch.Tensor,
    k_orig: torch.Tensor,
    v_orig: torch.Tensor,
    k_comp: torch.Tensor,
    v_comp: torch.Tensor,
) -> float:
    """Relative error between attention outputs before/after compression (0.0–1.0).

    Values below 0.01 indicate ±1% perplexity preservation.
    """
    out_orig = compute_attention_output(q, k_orig, v_orig)
    out_comp = compute_attention_output(q, k_comp, v_comp)
    return ((out_orig - out_comp).norm() / out_orig.norm().clamp(min=1e-8)).item()


def attention_kl_divergence(
    q: torch.Tensor,
    k_orig: torch.Tensor,
    k_comp: torch.Tensor,
) -> float:
    """KL divergence between attention score distributions before/after compression.

    Values below 0.015 approximate ±1% perplexity preservation.
    """
    scale = q.size(-1) ** -0.5
    attn_orig = F.softmax((q @ k_orig.T) * scale, dim=-1)
    attn_comp = F.softmax((q @ k_comp.T) * scale, dim=-1)
    kl = F.kl_div(
        attn_comp.log().clamp(min=-100),
        attn_orig,
        reduction="batchmean",
    ).item()
    return kl


def cosine_similarity_output(
    q: torch.Tensor,
    k_orig: torch.Tensor,
    v_orig: torch.Tensor,
    k_comp: torch.Tensor,
    v_comp: torch.Tensor,
) -> float:
    """Cosine similarity between attention outputs before/after compression.

    Values at or above 0.99 indicate ±1% accuracy preservation.
    """
    out_orig = compute_attention_output(q, k_orig, v_orig).flatten()
    out_comp = compute_attention_output(q, k_comp, v_comp).flatten()
    return F.cosine_similarity(
        out_orig.unsqueeze(0), out_comp.unsqueeze(0)
    ).item()
