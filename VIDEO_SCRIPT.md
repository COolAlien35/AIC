# AIC — 2-minute video walkthrough

A 5-beat storyboard for the rubric-mandated **<2-minute YouTube video**.
Total target length: **120 seconds** (5 beats × ~24s). Designed to be
recorded in **one take** with QuickTime + a single mic.

> **Recording target:** unlisted YouTube link pasted into README Quick Links.

## 1) Pre-flight (do this once, ~5 min)

* Open these tabs side-by-side in Chrome at 1280×720 zoom 100%:
  1. `https://github.com/COolAlien35/AIC` (README, scrolled to "Quick links")
  2. `https://huggingface.co/spaces/KINGKK007/aic-openenv-env` (Docker Space, post-build)
  3. `results/grpo_reward_curve.png` (zoomed)
  4. `aic/env/scenario_registry.py` and `aic/env/reward_engine.py` open in Cursor split view
  5. `results/benchmark_summary_normalized.csv` open in Cursor
* Terminal ready with `./.venv/bin/python` activated and these commands typed but **not** run:
  ```bash
  curl https://kingkk007-aic-openenv-env.hf.space/health
  ./.venv/bin/python scripts/score_tasks.py --episodes 1
  ```
* Make sure these results files exist locally so the cuts don't break:
  - `results/grpo_reward_curve.png`
  - `results/grpo_training_summary.json`
  - `results/benchmark_by_task_grader.csv`
  - `logs/grpo_progress.jsonl`
* Hide your dock, mute notifications, close every other window.
* Mic check: speak the word "incident" - peak should sit at -12 dB.

## 2) Storyboard (5 beats × 24s = 120s)

### Beat 1 — The problem (0:00 → 0:24)
**Voiceover (~55 words):**
> "Production outages aren't single failures, they cascade. A cache evicts,
> the DB pool saturates, latency goes p99-shaped, error rate explodes - and
> the on-call engineer reads a runbook that was written for a world that
> doesn't exist anymore. Worse: telemetry lies. A field gets renamed, a
> metric goes NaN, and your dashboards confidently mislead you."

**Screen:** Cursor split — `aic/env/scenario_registry.py` showing the 6
scenarios. Highlight the `telemetry_corruption_rules` block on Schema
Migration Disaster (NaN blackout / field rename / unit shift).

**B-roll cue:** quick cursor flick from "Cache Stampede" to "Schema
Migration Disaster" to "Credential Compromise". 3 scenarios in 4 seconds.

### Beat 2 — The environment (0:24 → 0:48)
**Voiceover (~55 words):**
> "AIC is an OpenEnv environment. Six specialist agents propose actions
> every step - DB, infra, app, network, security, and an adversary that
> lies on purpose. A deterministic Recovery Verifier gates every action by
> risk and blast radius. Twelve KPIs evolve through a service-topology
> DAG. The action is a structured `OrchestratorDecision` JSON."

**Screen:** Cursor → `openenv.yaml` (show `reset_method`, `step_method`,
**`state_method`**, `tasks:` block with 3 ids), then split to
`aic/schemas/actions.py` showing the `OrchestratorDecision` model.

**Then:** terminal — type and run:
```
curl https://kingkk007-aic-openenv-env.hf.space/health
```
Show `{"status":"ok","active_envs":0}` returning live from the HF Space.

### Beat 3 — The reward (0:48 → 1:12)
**Voiceover (~55 words):**
> "Reward is verifiable. Eight components: outcome, SLA, trust calibration,
> explanation quality, format compliance, verifier veto, reasoning
> coherence, and progress. Plus a confident-and-wrong penalty so the
> policy can't just shout. Each task also has a deterministic 0-to-1
> grader - the headline metric judges use to compare policies."

**Screen:** Cursor → `aic/env/reward_engine.py` (highlight R1-R8) → split
to `aic/tasks/task_db_pool_recovery.py` (show `def grade(trace) -> float`
returning `[0,1]`).

**Then:** terminal:
```
./.venv/bin/python scripts/score_tasks.py --episodes 1
```
Show the 3 task graders printing 0-1 scores in real time.

### Beat 4 — Real training (1:12 → 1:36)
**Voiceover (~55 words):**
> "Training is real. Eighty GRPO steps on a Colab T4 GPU using TRL plus
> Unsloth, six point one nine hours of wall clock. Reward improved from
> minus fifteen point one zero to minus ten point two four - a four point
> eight six delta on a brutal reward surface. No synthetic curves, no
> projected numbers. The raw JSONL log is committed in the repo."

**Screen:** `results/grpo_reward_curve.png` zoomed full screen, then
zoom-out to `logs/grpo_progress.jsonl` open in Cursor showing the JSONL
rows with real `step`, `reward`, `loss`, `kl`, `elapsed_minutes`.

**Then:** flash `results/grpo_training_summary.json` for 1 second:
`{"total_steps": 80, "initial_reward": -15.10, "final_reward": -10.24, "reward_delta": +4.86, "training_time_minutes": 371.3}`.

### Beat 5 — The result + the ask (1:36 → 2:00)
**Voiceover (~55 words):**
> "Live HF Space, three graded tasks, real GRPO run, OpenAI baseline
> script, and a CPU-safe sixty-second reproduce path - all linked from
> the README quick-links. Pull the environment from `KINGKK007 /
> aic-openenv-env`. GitHub at the URL on screen. Thanks for watching."

**Screen:** Final card (full screen, dark background, big white text):

```
AIC — Adaptive Incident Choreographer

OpenEnv:    huggingface.co/spaces/KINGKK007/aic-openenv-env
GitHub:     github.com/COolAlien35/AIC
Reward:     real GRPO   -15.10  ->  -10.24   (Δ +4.86, 80 steps, 6.19 h)
Tasks:      3 graders   0.0 — 1.0   easy / medium / hard
```

Hold for 4 seconds. Fade out.

## 3) Recording shot list (in order)

| # | Source | Duration | Notes |
|---|--------|----------|-------|
| 1 | Cursor — `aic/env/scenario_registry.py` | 24s | scroll, highlight telemetry_corruption_rules |
| 2 | Cursor split — `openenv.yaml` ⇆ `aic/schemas/actions.py` | 12s | show state_method + OrchestratorDecision |
| 3 | Terminal — `curl .../health` | 6s | live HF Space response |
| 4 | Cursor — `aic/env/reward_engine.py` | 8s | scroll R1-R8 |
| 5 | Cursor split — `aic/tasks/task_db_pool_recovery.py` ⇆ `task_adversarial_misroute.py` | 8s | show grade() functions |
| 6 | Terminal — `score_tasks.py --episodes 1` | 8s | live grader output |
| 7 | `results/grpo_reward_curve.png` full screen | 12s | zoom in/out |
| 8 | `logs/grpo_progress.jsonl` in Cursor | 8s | scroll JSONL rows |
| 9 | `results/grpo_training_summary.json` | 4s | flash totals |
| 10 | Final card | 4s | hold + fade |

## 4) Recording protocol

1. Press `Cmd+Shift+5` → "Record Selected Portion" → drag a 1280×720 box
   over the screen area you'll use throughout. Lock the box.
2. Click **Options → Microphone → MacBook Pro Microphone**.
3. Click **Record**. Count "1, 2, 3" silently in your head, then start.
4. **One take.** If you stumble, keep going - 90% of the value is in the
   first take. Never restart.
5. Stop with `Cmd+Ctrl+Esc`.
6. Open in QuickTime, **Edit → Trim** to remove dead air at start/end.
7. **File → Export As → 1080p**. Save as `aic_walkthrough_v1.mp4`.

## 5) YouTube upload

1. youtube.com/upload → drag the file.
2. Title: `AIC — Adaptive Incident Choreographer (OpenEnv hackathon)`.
3. Description (paste verbatim):

```
2-minute walkthrough of the AIC OpenEnv environment for the Meta OpenEnv hackathon.

OpenEnv environment Space (judges pull this URL):
https://huggingface.co/spaces/KINGKK007/aic-openenv-env

Live Gradio demo:
https://huggingface.co/spaces/KINGKK007/aic-incident-command-center

Source code:
https://github.com/COolAlien35/AIC

Real GRPO training: 80 steps on a Colab T4, 6.19 h wall-clock,
reward improved -15.10 -> -10.24 (Δ +4.86). Final loss 0.0026.
Three deterministic 0.0-1.0 task graders (easy / medium / hard).
Reward function: 8 verifiable components + R9 over-confidence penalty + verifier veto.
```

4. Visibility: **Unlisted**.
5. Click **Publish**, copy the share URL.
6. Paste into the README "Quick links" table → row "2-minute video walkthrough".
7. Commit and push: `git add README.md && git commit -m "add YouTube walkthrough URL" && git push`.

Done. Total time end-to-end: ~45 minutes.
