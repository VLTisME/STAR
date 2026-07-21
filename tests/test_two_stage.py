import torch

from star.inference.two_stage import evaluate_order, minmax_per_query, rank_candidates, scores_from_order


def test_minmax_per_query_handles_a_tied_row():
    scores = torch.tensor([[2.0, 2.0], [1.0, 3.0]])
    result = minmax_per_query(scores)
    assert torch.isfinite(result).all()
    assert torch.equal(result[0], torch.zeros(2))
    assert torch.equal(result[1], torch.tensor([0.0, 1.0]))


def test_ranked_candidates_override_only_the_topk():
    pe = torch.tensor([[0.9, 0.8, 0.2, 0.1]])
    order = torch.tensor([[1, 0]])
    final = scores_from_order(pe, order)
    assert torch.argsort(final, descending=True).tolist() == [[1, 0, 2, 3]]


def test_candidate_metrics_keep_the_pe_tail_for_outside_gt():
    pe = torch.tensor([[0.9, 0.8, 0.2, 0.1]])
    order = torch.tensor([[1, 0]])
    report = evaluate_order(pe, torch.tensor([2]), order)
    assert report["R@1"] == 0.0
    assert report["R@5"] == 1.0


def test_rank_candidates_keeps_indices_aligned_with_scores():
    ids = torch.tensor([[8, 3, 5]])
    scores = torch.tensor([[0.1, 0.8, 0.4]])
    order, ordered = rank_candidates(ids, scores)
    assert order.tolist() == [[3, 5, 8]]
    assert ordered.tolist() == [[0.8, 0.4, 0.1]]
