import importlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

merge = importlib.import_module("scripts.merge_dual_gpu_benchmarks")


def _two_episode_csv(path: Path, p1: str, p2: str, v1: float, v2: float, rid: int) -> None:
    df = pd.DataFrame(
        {
            "policy": [p1, p1, p1, p2, p2, p2],
            "scenario": ["Cache Stampede"] * 3 + ["Cache Stampede"] * 3,
            "episode_index": [0, 1, 2] * 2,
            "reward": [v1, v1, v1, v2, v2, v2],
            "success": [False] * 6,
        }
    )
    df["training_run_id"] = rid
    df["mttr"] = float("nan")
    df["adversary_suppression"] = float("nan")
    df["unsafe_rate"] = float("nan")
    df["trained_policy_source"] = "n/a"
    df["trained_policy_checkpoint"] = "n/a"
    df.to_csv(path, index=False)


def test_dedup_baselines_identical():
    with tempfile.TemporaryDirectory() as d:
        d1 = Path(d) / "r1.csv"
        d2 = Path(d) / "r2.csv"
        b = 10.0
        t1 = 5.0
        t2 = 7.0
        p = ("baseline_frozen", "trained_grpo")
        _two_episode_csv(d1, p[0], p[1], b, t1, 1)
        _two_episode_csv(d2, p[0], p[1], b, t2, 2)
        a1 = merge.load_episodes(d1, 1)
        a2 = merge.load_episodes(d2, 2)
        out, info = merge.dedup_baselines(a1, a2)
        assert info.get("deduped") is True
        assert len(out) == 3 + 3 + 3  # 3 baseline + 2 * 3 trained
        assert out[out["policy"] == "trained_grpo"].shape[0] == 6


def test_merge_produces_artifacts():
    r1 = Path("data/benchmark_ingest/run1_episodes_from_fixture.csv")
    r2 = Path("data/benchmark_ingest/run2_episodes_from_fixture.csv")
    if not r1.exists() or not r2.exists():
        pytest.skip("Fixtures not present")
    with tempfile.TemporaryDirectory() as d:
        outd = Path(d) / "m"
        merge.run_merge(
            r1, r2, outd, copy_newone_stats=Path("newone/statistical_test.json") if Path("newone/statistical_test.json").exists() else None
        )
        assert (outd / "benchmark_episodes_long.csv").exists()
        st = json.loads((outd / "statistical_test_merged.json").read_text())
        assert "p_value" in st
        assert (outd / "normalization_config.json").exists()
