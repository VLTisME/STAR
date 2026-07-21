"""Pure score manipulation for PE -> X-VLM two-stage retrieval.

PE is the global retriever.  X-VLM only changes the ordering of PE's fixed Top-K
candidates, which keeps the two model environments independent and makes every
postprocessing ablation reproducible from cached scores.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..metrics import full_report, rank_of_gt
from .assignment import apply_gale_shapley, gale_shapley_match, greedy_sca, top1_conflict_count


def minmax_per_query(scores: Tensor) -> Tensor:
    """Normalize candidate scores independently without producing NaNs on ties."""
    low = scores.amin(dim=1, keepdim=True)
    high = scores.amax(dim=1, keepdim=True)
    return (scores - low) / (high - low).clamp_min(1e-8)


def rank_candidates(candidate_indices: Tensor, candidate_scores: Tensor) -> tuple[Tensor, Tensor]:
    """Return gallery IDs and aligned scores in descending candidate-score order."""
    positions = torch.argsort(candidate_scores, dim=1, descending=True)
    return (
        torch.gather(candidate_indices, 1, positions),
        torch.gather(candidate_scores, 1, positions),
    )


def apply_postprocess(order: Tensor, ordered_scores: Tensor, method: str) -> Tensor:
    if method == "none":
        return order
    if method == "greedy_sca":
        resolved, _ = greedy_sca(order, ordered_scores)
        return resolved
    if method == "gale_shapley":
        matched = gale_shapley_match(order, ordered_scores)
        return apply_gale_shapley(order, matched)
    raise ValueError(f"Unknown postprocess method: {method}")


def scores_from_order(pe_scores: Tensor, order: Tensor) -> Tensor:
    """Make ``order`` the exact Top-K while preserving PE's tail ordering.

    The input candidates are PE Top-K, so every unlisted item was already below
    the candidate set.  Giving ordered candidates a small score band above PE's
    row maximum yields an exact full-gallery metric calculation without storing a
    dense cross-encoder score matrix.
    """
    if pe_scores.ndim != 2 or order.ndim != 2 or pe_scores.size(0) != order.size(0):
        raise ValueError("pe_scores and order must be [Q, G] and [Q, K]")
    result = pe_scores.clone().float()
    q = torch.arange(order.size(0), device=result.device).unsqueeze(1)
    maximum = result.amax(dim=1, keepdim=True)
    k = order.size(1)
    # All candidates are strictly above the PE tail, with deterministic ranks.
    rank_band = torch.arange(k, 0, -1, device=result.device, dtype=result.dtype).unsqueeze(0)
    result[q, order] = maximum + rank_band / float(k + 1)
    return result


def evaluate_order(pe_scores: Tensor, gt_index: Tensor, order: Tensor) -> dict[str, float | int]:
    scores = scores_from_order(pe_scores, order)
    report = full_report(scores, gt_index, ks=(1, 5, 10, 50, 200))
    ranks = rank_of_gt(scores, gt_index)
    return {
        **report,
        "candidate_coverage": float((ranks <= order.size(1)).float().mean().item()),
        "gt_at_rank_2": int((ranks == 2).sum().item()),
        "top1_conflicts": top1_conflict_count(order),
    }
