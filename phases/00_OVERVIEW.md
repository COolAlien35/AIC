# 🏆 AIC HACKATHON — EXECUTION ROADMAP
> Broken from `LEGENDARY_MASTERPLAN.md` into sequential implementation phases.

---

## 🧠 The One-Line Strategy

> You don't win by having the best architecture. You win by **proving your agent learns.**

```
PROOF = Trained Model + Benchmark Δ + Statistical Test + Evidence Bundle + Clean README
```

---

## 🗺️ Architecture at a Glance

```
Specialist Agents (db, infra, app, network, security)
        + Adversarial Agent (injects deceptive recommendations)
        + Knowledge/Analysis Agents
                │
                ▼
  Orchestrator Agent  ──────────────────────> AICEnvironment
  (THE MODEL BEING TRAINED)                  WorldState | Verifier | RewardEngine (R1–R8)
                │                                         │
                └──────── Traces ──────────────────────────┘
                                    │
              SFT Data ─────────────┼───── GRPO Training ──── Evidence Bundle
```

**The Orchestrator must learn to:**
1. Select correct specialist under normal conditions
2. Override adversarial recommendations
3. Detect schema drift and adjust confidence
4. Prioritize by risk, SLA urgency, and business impact

---

## ⏱️ Time Budget

| Phase | File | Activity | Time | Who |
|-------|------|----------|------|-----|
| **0** | `PHASE_0_EMERGENCY_TRIAGE.md` | Code fixes (blocking everything) | 1.5 hrs | Dev (local) |
| **1** | `PHASE_1_COLAB_SETUP.md` | GPU environment setup | 20 min | Dev (Colab) |
| **2** | `PHASE_2_SFT_TRAINING.md` | SFT on Qwen2.5 3B | 2 hrs | GPU (idle) |
| **3** | `PHASE_3_GRPO_TRAINING.md` | GRPO reinforcement training | 5–7 hrs | GPU (idle) |
| **4** | `PHASE_4_BENCHMARK.md` | Benchmark + statistical proof | 1 hr | GPU |
| **5** | `PHASE_5_EVIDENCE_BUNDLE.md` | Plots + README surgery | 45 min | Dev (parallel) |
| **6** | `PHASE_6_EXPORT_DEMO.md` | Export weights + demo wiring | 45 min | Dev (parallel) |
| **7** | `PHASE_7_PRIZE_SUBMISSION.md` | Prize alignment + pitch | 30 min | Team |
| **B** | `BONUS_PHASES.md` | Ablation, heatmap, HuggingFace | 4+ hrs | If time permits |

> **Wall-clock net time ≈ 8 hours** — Phases 5 & 6 run in parallel while GPU handles 2–4.

---

## 🛑 Stop Points (If Short on Time)

```
STOP A — 8 hrs  →  Phases 0–4  →  60% win probability
STOP B — 10 hrs →  + Phases 5–6 →  75% win probability
STOP C — 13 hrs →  + Phase 7    →  80% win probability
STOP D — 18 hrs →  + Bonus      →  85–90% win probability
```

---

## ✅ Final Submission Checklist

- [ ] `artifacts/sft/orchestrator_sft.jsonl` — 600+ examples, 6 scenarios
- [ ] `checkpoints/sft/` — Qwen2.5 3B SFT checkpoint
- [ ] `checkpoints/grpo/` — Trained GRPO model + `training_summary.json`
- [ ] `exports/aic-orchestrator-trained/` — Merged weights, inference-ready
- [ ] `results/benchmark_summary.csv` — 3 policies × 30 episodes
- [ ] `results/statistical_test.json` — t-test + Cohen's d
- [ ] `results/reward_curve.png`
- [ ] `results/policy_comparison.png`
- [ ] `results/evidence_manifest.json`
- [ ] `logs/grpo_progress.jsonl` — 50+ entries
- [ ] `README.md` — Real numbers, no `[FILL]`, no `tiny-gpt2`
- [ ] `BRUTAL_HACKATHON_AUDIT.md` — Include it. Judges love intellectual honesty.
- [ ] 248 tests passing
- [ ] HuggingFace Space live demo link
- [ ] 3-minute video demo

---

*Execute phases in sequence. Do not skip Phase 0.*
