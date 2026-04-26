# 📊 AIC Results Dashboard (static)

A clean, professional, single-page **static HTML dashboard** built on top of the
merged benchmark and the real GRPO training log.

> No backend. No build step. No framework lag. Just open `index.html`.

## Run it

The simplest path — open the file directly:

```bash
open dashboard/site/index.html              # macOS
xdg-open dashboard/site/index.html          # Linux
start dashboard\site\index.html             # Windows
```

If your browser blocks local images served from `file://` (Firefox sometimes does),
serve the directory with Python's stdlib HTTP server instead:

```bash
python3 -m http.server -d dashboard/site 8000
# then visit http://localhost:8000
```

## Refresh the data

The dashboard reads pre-computed JSON via `data.js`. To regenerate it from the
canonical results files (`results/benchmark_merged/`, `results/grpo_training_summary.json`,
`logs/grpo_progress.jsonl`, etc.):

```bash
./.venv/bin/python dashboard/site/build_data.py
```

This copies the latest plots into `dashboard/site/plots/` as well.

## What you'll see

| Section | Source data |
|---|---|
| Hero KPIs (uplift, Cohen's d, p-value, GRPO Δ) | `results/benchmark_merged/statistical_test_merged.json` · `results/grpo_training_summary.json` |
| Headline bar (mean reward + 95 % CI) | `results/benchmark_merged/benchmark_summary_merged.csv` |
| GRPO training (reward · loss · KL) | `logs/grpo_progress.jsonl` |
| Per-scenario uplift + grouped bars + table | `results/benchmark_merged/benchmark_by_scenario_merged.csv` |
| Statistical evidence panel | `results/benchmark_merged/statistical_test_merged.json` |
| Task graders (3 cards, easy/medium/hard) | `results/inference_summary.json` · `results/benchmark_by_task_grader.csv` |
| Figure gallery (12 figures) | `results/benchmark_merged/plots/*.png` |

## Files

```
dashboard/site/
├── index.html       ← entry point
├── styles.css       ← dark, professional theme, no animation lag
├── main.js          ← Chart.js bindings + table renderers
├── data.js          ← embedded JSON (rebuild with build_data.py)
├── build_data.py    ← regenerates data.js + refreshes plots/
├── plots/           ← copy of results/benchmark_merged/plots/*.png
└── README.md        ← this file
```

## Why static (and not Streamlit)

The repo also ships an internal `dashboard/app.py` Streamlit app for live trajectory
inspection. This static version is intended for **external reviewers / hackathon
judges** — opens instantly, no Python kernel boot, no rerun-on-every-click jank, and
the data is locked to the merged source-of-truth so it can never drift.
