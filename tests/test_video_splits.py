from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_module():
    import sys

    path = Path(__file__).parents[1] / "scripts" / "make_video_disjoint_splits.py"
    spec = spec_from_file_location("make_video_disjoint_splits", path)
    module = module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cross_video_hard_pair_stays_together():
    module = load_module()
    rows = [
        {"image_id": "a", "hard_i_id": "b", "video_id": 1},
        {"image_id": "b", "hard_i_id": "a", "video_id": 2},
        {"image_id": "c", "hard_i_id": "c", "video_id": 3},
        {"image_id": "d", "hard_i_id": "d", "video_id": 4},
    ]
    splits, _, summary = module.build_split(rows, module.DEFAULT_RATIOS, seed=2026)
    assert splits["1"] == splits["2"]
    assert summary["hard_pair_leakage"] == 0


def test_missing_hard_target_fails():
    module = load_module()
    rows = [{"image_id": "a", "hard_i_id": "missing", "video_id": 1}]
    try:
        module.build_split(rows, module.DEFAULT_RATIOS, seed=2026)
    except ValueError as exc:
        assert "hard targets" in str(exc)
    else:
        raise AssertionError("expected missing hard target to fail")
