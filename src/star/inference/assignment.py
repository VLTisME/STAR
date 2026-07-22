"""Candidate-list postprocessing for the clean dev/report protocol.

All functions operate only on a frozen score matrix and candidate list. Their
hyperparameters and selection are governed by the dev split; report is called
only after the method is frozen.
"""
from __future__ import annotations

import torch
from torch import Tensor


def top1_conflict_count(order: Tensor) -> int:
    """Number of duplicate rank-one gallery assignments."""
    top1 = order[:, 0]
    return int(top1.numel() - torch.unique(top1).numel())


def _promote(order: Tensor, chosen: Tensor) -> Tensor:
    new_order = order.clone()
    for query_index, value in enumerate(chosen.tolist()):
        if value < 0:
            continue
        row = [int(candidate) for candidate in order[query_index].tolist()]
        if value not in row:
            continue
        new_order[query_index] = torch.tensor(
            [value] + [candidate for candidate in row if candidate != value], dtype=order.dtype
        )
    return new_order


def greedy_sca(order: Tensor, scores: Tensor) -> tuple[Tensor, Tensor]:
    """A deterministic greedy conflict resolver used as an SCA-style ablation."""
    if order.shape != scores.shape:
        raise ValueError("order and scores must have identical [queries, candidates] shapes")
    assigned = order[:, 0].clone()
    holders: dict[int, list[int]] = {}
    for query_index, gallery_index in enumerate(assigned.tolist()):
        holders.setdefault(int(gallery_index), []).append(query_index)
    # A gallery may initially be held by several queries.  Track multiplicity:
    # removing one losing claim must not make the image available while another
    # query still holds it.
    occupancy: dict[int, int] = {}
    for value in assigned.tolist():
        gallery_index = int(value)
        occupancy[gallery_index] = occupancy.get(gallery_index, 0) + 1
    for queries in holders.values():
        if len(queries) < 2:
            continue
        _, *losers = sorted(queries, key=lambda q: float(scores[q, 0]), reverse=True)
        for query_index in losers:
            old = int(assigned[query_index])
            replacement = next(
                (int(candidate) for candidate in order[query_index].tolist()
                 if int(candidate) != old and occupancy.get(int(candidate), 0) == 0),
                None,
            )
            if replacement is not None:
                assigned[query_index] = replacement
                occupancy[old] -= 1
                if occupancy[old] == 0:
                    del occupancy[old]
                occupancy[replacement] = occupancy.get(replacement, 0) + 1
    return _promote(order, assigned), assigned


def gale_shapley_match(order: Tensor, scores: Tensor) -> Tensor:
    """Deferred acceptance over score-ordered candidate lists."""
    if order.shape != scores.shape:
        raise ValueError("order and scores must have identical [queries, candidates] shapes")
    query_count, candidate_count = order.shape
    next_choice = [0] * query_count
    holder: dict[int, int] = {}
    held_score: dict[int, float] = {}
    free = list(range(query_count))
    while free:
        query_index = free.pop()
        while next_choice[query_index] < candidate_count:
            position = next_choice[query_index]
            gallery_index = int(order[query_index, position])
            score = float(scores[query_index, position])
            next_choice[query_index] += 1
            if gallery_index not in holder:
                holder[gallery_index] = query_index
                held_score[gallery_index] = score
                break
            if score > held_score[gallery_index]:
                rejected = holder[gallery_index]
                holder[gallery_index] = query_index
                held_score[gallery_index] = score
                free.append(rejected)
                break
    matched = torch.full((query_count,), -1, dtype=torch.long)
    for gallery_index, query_index in holder.items():
        matched[query_index] = gallery_index
    return matched


def apply_gale_shapley(order: Tensor, matched: Tensor) -> Tensor:
    """Promote a Gale-Shapley match to rank one without changing list membership."""
    if matched.numel() != order.size(0):
        raise ValueError("matched must contain one gallery index per query")
    return _promote(order, matched)
