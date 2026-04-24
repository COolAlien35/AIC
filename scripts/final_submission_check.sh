#!/bin/bash
# scripts/final_submission_check.sh
# Run this before hitting submit to verify everything is in place.

echo "=================================="
echo "AIC FINAL SUBMISSION VERIFICATION"
echo "=================================="

PASS=0
FAIL=0

check() {
    if eval "$2"; then
        echo "✅ $1"
        PASS=$((PASS+1))
    else
        echo "❌ $1"
        FAIL=$((FAIL+1))
    fi
}

# Data
check "SFT examples ≥ 400" "[ \$(wc -l < artifacts/sft/orchestrator_sft.jsonl 2>/dev/null || echo 0) -ge 400 ]"

# Checkpoints
check "SFT checkpoint exists" "[ -d checkpoints/sft ]"
check "GRPO checkpoint exists" "[ -d checkpoints/grpo ]"
check "Trained model NOT tiny-gpt2" "! grep -q 'tiny-gpt2' checkpoints/sft/sft_metadata.json 2>/dev/null"

# Results
check "Benchmark CSV exists" "[ -f results/benchmark_summary.csv ]"
check "Statistical test exists" "[ -f results/statistical_test.json ]"
check "Reward curve plot exists" "[ -f results/reward_curve.png ]"
check "Verifier pass rate plot exists" "[ -f results/verifier_pass_rate.png ]"
check "Evidence manifest exists" "[ -f results/evidence_manifest.json ]"

# Logs
check "GRPO training logs exist" "[ -f logs/grpo_progress.jsonl ]"

# Code quality
check "Tests pass" "python -m pytest tests/ -q --tb=no 2>/dev/null | grep -q 'passed'"

# README
check "README has no 'tiny-gpt2'" "! grep -q 'tiny-gpt2' README.md"
check "README has no placeholder FILL" "! grep -q '\[FILL\]' README.md"

echo ""
echo "=================================="
echo "Result: $PASS passed, $FAIL failed"

if [ $FAIL -eq 0 ]; then
    echo "🏆 SUBMISSION READY — GO WIN"
else
    echo "⚠️  FIX $FAIL ISSUES BEFORE SUBMITTING"
fi
echo "=================================="
