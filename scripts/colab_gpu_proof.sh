#!/usr/bin/env bash
set -euo pipefail

echo "== AIC Colab GPU proof run =="
echo "Working directory: $(pwd)"

python3 --version
nvidia-smi || true

if [ ! -d ".venv" ]; then
  python3.12 -m venv .venv || python3 -m venv .venv
fi

./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo "== Dependency check =="
./.venv/bin/python run_hackathon.py verify

echo "== GRPO training (GPU expected) =="
./.venv/bin/python run_hackathon.py grpo

echo "== Benchmark and demo artifacts =="
./.venv/bin/python scripts/run_final_benchmark.py
./.venv/bin/python run_hackathon.py plots demo

echo "== Export validation on GRPO checkpoint =="
./.venv/bin/python eval/test_export.py --source checkpoints/grpo

echo "== Final evidence manifest =="
./.venv/bin/python run_hackathon.py verify

echo "Done. Key outputs:"
echo "- checkpoints/grpo/"
echo "- logs/eval/policy_benchmark.jsonl"
echo "- results/benchmark_summary.csv"
echo "- results/reward_curve.png"
echo "- results/verifier_pass_rate.png"
echo "- results/before_after_demo.md"
echo "- results/evidence_manifest.json"
