# Changes Plan (Post Phase 3)

This file lists what to do **after Phase 3 is implemented** and what to do to make AIC the strongest possible submission.

---

## 1) Immediate Priority (Must Do Next)

These are mandatory to convert training progress into proof.

### 1.1 Benchmark and Statistical Proof (Phase 4)
- Run full benchmark across all policies and scenarios.
- Ensure outputs are generated:
  - `results/benchmark_summary.csv`
  - `results/benchmark_by_scenario.csv`
  - `results/statistical_test.json`
- Gate criteria:
  - `trained_grpo` appears in benchmark outputs.
  - Trained policy reward is better than frozen baseline.
  - Target significance: `p_value < 0.05` and meaningful Cohen's d.

### 1.2 Evidence Bundle and README Alignment (Phase 5)
- Generate and verify:
  - `results/reward_curve.png`
  - `results/policy_comparison.png`
  - `results/evidence_manifest.json`
- Update `README.md` with only real numbers from current artifacts.
- Remove stale placeholders and old model references.
- Ensure README claims exactly match generated results.

### 1.3 Export + Demo Readiness (Phase 6)
- Export merged model to:
  - `exports/aic-orchestrator-trained/`
- Validate JSON decision quality on adversarial test prompt.
- Confirm demo has trained-vs-baseline toggle and works deterministically.

### 1.4 Submission Hardening (Phase 7)
- Run final verification script with zero failures.
- Rehearse 3-minute demo around one reproducible "wow" scenario:
  - Baseline follows bad recommendation.
  - Trained model overrides adversary and recovers within SLA.

---

## 2) High-Impact Upgrades (Make It Best)

After the required phases are green, do these for stronger score and robustness.

### 2.1 Training Quality Improvements
- Increase GRPO steps beyond minimal target (e.g., 150 -> 300+) if GPU budget allows.
- Run multiple seeds and report average + variance (not single-run cherry-pick).
- Add curriculum schedule tuning:
  - Harder adversarial frequency in later steps.
  - More schema drift in late-stage training.
- Add early-stop logic on reward plateau and save best checkpoint by eval metric.

### 2.2 Reward and Safety Refinement
- Audit each reward component contribution (R1-R8) to detect reward hacking.
- Add penalties for:
  - Contradictory reasoning vs selected action.
  - Repeated low-impact actions.
- Track safety KPIs separately:
  - Unsafe action rate
  - Verifier veto rate
  - Adversary override precision/recall

### 2.3 Evaluation Depth
- Add ablation table:
  - baseline
  - SFT-only
  - GRPO without curriculum
  - full GRPO + curriculum
- Add per-scenario heatmap for reward/success.
- Add adversarial override analysis by scenario and severity.

### 2.4 Reliability and Reproducibility
- Add one-command reproducible pipeline (train -> eval -> plots -> manifest).
- Pin dependency versions for stable reruns.
- Save full run metadata:
  - git commit
  - seed
  - hardware info
  - config snapshot
- Add CI checks to fail on missing required artifacts.

### 2.5 Demo and Storytelling Quality
- Build one polished scenario script with predictable telemetry progression.
- Show side-by-side metrics trend:
  - reward
  - health
  - SLA status
  - adversary suppression
- Keep demo deterministic with fixed seed and predefined scenario config.

---

## 3) Model Evolution Roadmap (After Core Completion)

Only do this once all required artifacts and proof are already strong.

- Move from current Qwen size to larger model class if budget permits.
- Keep same eval harness and compare using identical seeds/scenarios.
- Report tradeoff table:
  - quality gain
  - latency
  - memory
  - cost

---

## 4) Definition of Done (True "All Done")

Project is considered complete when all are true:

- Phase 4 outputs exist and trained policy improvement is statistically supported.
- Phase 5 visual and manifest artifacts are present and README is fully aligned.
- Phase 6 export exists and demo can switch baseline/trained reliably.
- Phase 7 verification script passes with zero blocking issues.
- Submission evidence is reproducible from documented commands.

---

## 5) Suggested Execution Order (Fastest Path)

1. Finish Phase 4 and lock benchmark numbers.
2. Finish Phase 5 and update README with exact numbers.
3. Finish Phase 6 and validate demo toggle behavior.
4. Finish Phase 7 and run final submission checks.
5. Run high-impact upgrades (ablations, multi-seed, heatmaps) if time remains.

