# 🏅 PHASE 7 — PRIZE ALIGNMENT & SUBMISSION POLISH
> **Time: 30 minutes | Maximum ROI — every word maps to a prize category**

---

## Prize Targeting Map

Don't submit generically. Every sentence in your README and pitch should link directly to a prize.

| Prize | Key Requirement | Your Evidence |
|-------|----------------|---------------|
| **Fleet AI** (Multi-agent) | Coordinated agent decisions under uncertainty | 6 specialist agents + adversarial + verifier, GRPO reward proves coordination improves |
| **Halluminate** (Adversary detection) | Detecting and rejecting adversarial inputs | `override_adversary=True` decisions in benchmark, per-agent trust calibration |
| **Patronus AI** (Safety + eval) | Evaluation framework + safety constraints | Recovery Verifier gate + 248 tests + statistical benchmark + `BRUTAL_HACKATHON_AUDIT.md` |
| **Scaler AI** (Enterprise RAG) | Retrieval-augmented reasoning | Knowledge agent + runbook retrieval in orchestration loop |

---

## Submission Package Checklist

```
□ GitHub repo           — All code, clean commit history
□ checkpoints/grpo/     — Trained model checkpoint (THE PROOF)
□ results/benchmark_summary.csv     — Real numbers, 3 policies × 30 episodes
□ results/statistical_test.json     — t-test + Cohen's d
□ results/reward_curve.png          — Visual proof of learning
□ results/policy_comparison.png     — Before/after comparison
□ results/evidence_manifest.json    — Complete artifact index
□ README.md             — No false claims, no [FILL], real numbers
□ BRUTAL_HACKATHON_AUDIT.md         — Include it. Shows intellectual maturity.
□ HuggingFace Space URL — Live demo with trained model toggle
□ Video demo link       — 3 minutes, structured (see pitch script below)
```

---

## The 3-Minute Pitch Script

Use this structure exactly. Each block has a job.

### 0:00–0:30 — THE PROBLEM
> *"Modern incident response requires orchestrating dozens of specialist agents under adversarial conditions, schema uncertainty, and SLA pressure. Humans are too slow. Rule-based systems are too rigid. We need an agent that learns."*

### 0:30–1:30 — THE ARCHITECTURE
> *"We built AIC: 6 specialist agents plus an adversarial agent plus a Recovery Verifier, all governed by a trained Orchestrator that learns via GRPO with 8-component reward decomposition and automatic reward hacking detection."*

**[Show architecture diagram here]**

### 1:30–2:15 — THE PROOF
> *"After GRPO training, the Orchestrator goes from [X]% to [Y]% success rate. Reward improvement: +[X] units. p-value: [X]. Cohen's d: [X] — a [large/medium] effect size. This is statistically significant."*

**[Show reward curve + benchmark table here]**

### 2:15–3:00 — THE DEMO
> *"Here's a live cascading failure. The baseline picks the adversarial agent's recommendation. Our trained model detects it and overrides it — recovering the system within SLA."*

**[Live Gradio demo with toggle ON here]**

---

## Final Verification Script

Run this before hitting submit:

```bash
#!/bin/bash
# scripts/final_submission_check.sh

echo "=================================="
echo "AIC FINAL SUBMISSION VERIFICATION"
echo "=================================="

PASS=0; FAIL=0

check() {
    if eval "$2"; then
        echo "✅ $1"; PASS=$((PASS+1))
    else
        echo "❌ $1"; FAIL=$((FAIL+1))
    fi
}

# Data
check "SFT examples ≥ 500" "[ $(wc -l < artifacts/sft/orchestrator_sft.jsonl) -ge 500 ]"

# Checkpoints
check "SFT checkpoint exists" "[ -d checkpoints/sft ]"
check "GRPO checkpoint exists" "[ -d checkpoints/grpo ]"
check "Trained model NOT tiny-gpt2" "! grep -q 'tiny-gpt2' checkpoints/sft/sft_metadata.json 2>/dev/null"

# Results
check "Benchmark CSV exists" "[ -f results/benchmark_summary.csv ]"
check "Statistical test exists" "[ -f results/statistical_test.json ]"
check "Trained policy in benchmark" "grep -q 'trained_grpo' results/benchmark_summary.csv 2>/dev/null"
check "Reward curve plot exists" "[ -f results/reward_curve.png ]"
check "Policy comparison plot exists" "[ -f results/policy_comparison.png ]"
check "Evidence manifest exists" "[ -f results/evidence_manifest.json ]"

# Logs
check "GRPO training logs exist" "[ -f logs/grpo_progress.jsonl ]"
check "GRPO has >50 log entries" "[ $(wc -l < logs/grpo_progress.jsonl 2>/dev/null || echo 0) -ge 50 ]"

# Code quality
check "248 tests pass" "python -m pytest tests/ -q --tb=no 2>/dev/null | grep -q 'passed'"

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
```

```bash
chmod +x scripts/final_submission_check.sh
./scripts/final_submission_check.sh
```

---

## The Veteran's Last Word

**1. One strong proof beats ten weak signals.**  
A single clean `reward_delta = +47.3` with `p = 0.003` is worth more than fifteen features.

**2. Intellectual honesty is a competitive advantage.**  
`BRUTAL_HACKATHON_AUDIT.md` — include it. Teams that audit themselves are rare. Judges remember them.

**3. The demo must show adversary detection.**  
That moment where baseline picks the adversarial recommendation and YOUR model overrides it — that is your WOW moment. Engineer the demo scenario to show that contrast clearly.

**4. Name your components with precision.**  
Don't say "we prevent reward hacking." Say:  
*"We implement RLVR with process-based reward decomposition (R1–R8), where R6 specifically penalizes adversarial action selection and R8 tracks progress monotonicity to detect reward inflation. Here are the audit logs."*

**5. The trained model is the submission. Everything else is context.**  
If judges could keep only one file, it should be the GRPO checkpoint and the benchmark result that proves it works.

---

## ✅ Phase 7 Completion Criteria

- [ ] Final submission check script: **0 failures**
- [ ] Every README placeholder replaced with real numbers
- [ ] Pitch script rehearsed — under 3 minutes
- [ ] `BRUTAL_HACKATHON_AUDIT.md` included in repo
- [ ] HuggingFace Space URL live and working

**→ If you have time: [BONUS_PHASES.md](BONUS_PHASES.md)**
