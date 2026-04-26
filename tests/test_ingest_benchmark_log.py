import importlib
from pathlib import Path

ibl = importlib.import_module("scripts.ingest_benchmark_log")


SAMPLE = """
[bench] Benchmarking: baseline_frozen
  Cache Stampede ep0: reward=-258.62, success=False
  Cache Stampede ep1: reward=-261.18, success=False
[bench] Benchmarking: trained_grpo
  Cache Stampede ep0: reward=-100.0, success=False
"""


def test_parse_paste_text_rows():
    rows = ibl.parse_paste_text(SAMPLE)
    assert len(rows) == 3
    assert rows[0]["policy"] == "baseline_frozen"
    assert rows[0]["episode_index"] == 0
    assert rows[2]["reward"] == -100.0
    assert rows[2]["success"] is False


def test_to_dataframe_sort():
    rows = ibl.parse_paste_text(SAMPLE)
    df = ibl.to_dataframe(rows)
    assert "policy" in df.columns
    assert len(df) == 3


def test_sample_file_exists():
    p = Path(__file__).resolve().parent.parent / "data/benchmark_ingest/sample_paste.log"
    assert p.exists()
    t = p.read_text()
    r = ibl.parse_paste_text(t)
    assert len(r) > 0
