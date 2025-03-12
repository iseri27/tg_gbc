from typing import Callable

import torch
from torch import Tensor

from .utils import do_nothing, trim_none, parse_r


def trim_gbc(
    cls_scores: Tensor,
    attn_weights: Tensor = None,
    r: int = 0,
    k: int = -1,
    **kwargs,
) -> Callable:
    r"""Sort the query and key based on the classification scores.
        - Classification scores $C\in\mathbb{R}^{Nq \times Nc}$
        - Attention matrix $A\in\mathbb{R}^{Nq \times Nk}$
        - Extract the maximum values from $C$: $\hat{C} = \max(C, \text{dim}=1) \in \mathbb{R}^{Nq}$
        - Calculate the importance of Key:
            \[
                S_j = \sum_{i=0}^{{Nq}-1}\times A_{i,j}\times \hat{C}_{i}
            \]
    Args:
        - cls_scores (Tensor): $[B, Nq, Nc]$
        - attn_weights (Tensor): $[B, Nq, Nk]$
    """

    if r <= 0:
        return do_nothing

    B, Nq, Nk = attn_weights.shape

    with torch.no_grad():
        # [B, Nq, num_classes] -> [B, Nq]
        cls_scores = cls_scores.max(dim=-1).values.sigmoid()

        if k > 0:
            _, cls_indices = (-cls_scores).sort(dim=-1)
            cls_indices = cls_indices[:, :k]
            cls_indices_expand = cls_indices.unsqueeze(-1).expand(-1, -1, Nk)
            a = torch.gather(attn_weights, index=cls_indices_expand, dim=1)
            c = torch.gather(cls_scores, index=cls_indices, dim=1)[..., None]
            scores = a * c
        else:
            scores = attn_weights * cls_scores[..., None]

        # scores: [B, Nk]
        scores = torch.sum(scores, dim=1)
        # indices: [B, Nk]
        indices = scores.sort().indices

    def _trim(x: Tensor):
        """
        Args:
            x (Tensor): [B, Nk, E]
        Returns:
            Tensor: [B, Nk-r, E]
        """
        if not isinstance(x, Tensor):
            return x
        B, N, E = x.shape
        x = torch.gather(x, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, E))
        x = x[:, r:, :]
        return x

    return _trim


def build_gbc(r: int, n: int, k: int, layers: int = 6, *args, **kwargs):
    """
    Returns:
        trim_func, r_list, n, flag
    """
    tgtg_info = dict(r=parse_r(r, n, layers), n=n, k=k)

    if r <= 0 or n <= 0:
        tgtg_info["enable"] = False
        return trim_none, tgtg_info

    tgtg_info["enable"] = True

    return trim_gbc, tgtg_info
