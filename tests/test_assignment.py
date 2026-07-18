import torch

from star.inference import apply_gale_shapley, gale_shapley_match, greedy_sca, top1_conflict_count


def test_gale_shapley_resolves_a_duplicate_top1():
    order = torch.tensor([[0, 1, 2], [0, 2, 1]])
    scores = torch.tensor([[0.9, 0.5, 0.1], [0.8, 0.7, 0.2]])
    matched = gale_shapley_match(order, scores)
    resolved = apply_gale_shapley(order, matched)
    assert top1_conflict_count(order) == 1
    assert top1_conflict_count(resolved) == 0
    assert resolved[0, 0].item() == 0
    assert resolved[1, 0].item() == 2


def test_greedy_sca_preserves_candidate_membership():
    order = torch.tensor([[0, 1, 2], [0, 2, 1]])
    scores = torch.tensor([[0.9, 0.5, 0.1], [0.8, 0.7, 0.2]])
    resolved, _ = greedy_sca(order, scores)
    assert top1_conflict_count(resolved) == 0
    for source, target in zip(order, resolved):
        assert set(source.tolist()) == set(target.tolist())
