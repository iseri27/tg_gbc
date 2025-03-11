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
    r"""通过分类分数对 query 与 key 进行排序
    分类分数 C\in\mathbb R^{N_q\times N_C}
    注意力矩阵 A\in\mathbb R^{N_q\times N_k}
    挑选出 C 中的最大值 Cm = max(C, dim=1) \in \mathbb R^{N_q}
    根据 Cm 对 query 进行裁剪
    计算 Key 的重要性：
        S_j = \sum\limits_{i=0}^{N_q-1} Cm_i\times A_{i,j}
    Args:
        cls_scores (Tensor): [B, Nq, num_classes]
        attn_weights (Tensor): [B, Nq, Nk]
    """

    if r <= 0:
        return do_nothing

    select = kwargs.get("cls_select", "max")

    B, Nq, Nk = attn_weights.shape

    with torch.no_grad():
        # [B, Nq, num_classes] -> [B, Nq]
        if select == "mean":
            cls_scores = cls_scores.mean(dim=-1).sigmoid()
        elif select == "min":
            cls_scores = cls_scores.min(dim=-1).values.sigmoid()
        else:
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
        # 将 indices 用于维度 dim 的重排
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
