---
title: AIC Results Dashboard
emoji: "📊"
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Static results dashboard (merged benchmark + GRPO curves)
tags:
  - dashboard
  - openenv
  - reinforcement-learning
  - grpo
  - trl
  - unsloth
  - fastapi
---

# 📊 AIC — Results Dashboard

This Space hosts the **static results dashboard** for the AIC project.
It serves the files in `dashboard/site/` (HTML/CSS/JS + `data.js` + plots) with a
lightweight nginx container — it loads instantly and requires no backend.

## Links

- **GitHub (source of truth)**: https://github.com/COolAlien35/AIC
- **Judge-facing OpenEnv env Space** (canonical): https://huggingface.co/spaces/KINGKK007/aic-training
- **Judge runtime URL**: https://kingkk007-aic-training.hf.space

## What’s inside

- Headline benchmark comparison (mean reward + CI)
- GRPO training curves (reward / loss / KL)
- Per-scenario uplift table + figures
- 0–1 task graders (rubric-aligned)

> This Space is **dashboard-only**. It does not run the OpenEnv environment server.
> For evaluation, judges should use the canonical environment Space above.

