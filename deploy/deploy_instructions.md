# 🚢 Deployment Instructions — AIC

AIC ships **two Hugging Face Spaces** plus a self-host Docker option. This doc tells you
which one to use and how to push.

| Space | What it is | Who needs it |
|---|---|---|
| [`KINGKK007/aic-openenv-env`](https://huggingface.co/spaces/KINGKK007/aic-openenv-env) | **Canonical OpenEnv environment server** (Docker, FastAPI on port 7860) | Hackathon judges — they pull this URL to evaluate the env |
| [`KINGKK007/aic-incident-command-center`](https://huggingface.co/spaces/KINGKK007/aic-incident-command-center) | Interactive Gradio walkthrough demo | Anyone who wants to click through an episode in the browser |

The repo also contains everything you need to run the env locally via Docker or
`uvicorn`.

---

## 1) Push the canonical OpenEnv Space (judges pull this)

The build context is [`hf_env_space/`](../hf_env_space/) — a thin Docker wrapper that
clones the source repo and starts `aic.server.env_api:app` on port 7860.

```bash
# 1. Create the Space (one-time, on huggingface.co/new-space)
#    SDK: Docker · Hardware: CPU Basic · Visibility: Public · port: 7860

# 2. Push the wrapper from this repo
git remote add hf-env https://huggingface.co/spaces/KINGKK007/aic-openenv-env
git subtree push --prefix=hf_env_space hf-env main
```

Wait for the build to go green, then smoke-test:

```bash
HOST="https://kingkk007-aic-openenv-env.hf.space"
curl -s "$HOST/health"
ENV_ID=$(curl -sX POST "$HOST/reset" -H 'Content-Type: application/json' \
  -d '{"episode_id":0,"base_seed":42,"fault_mode":"cascading_failure"}' | jq -r .env_id)
curl -s "$HOST/state/$ENV_ID" | jq '.state | {step,scenario_name,health_score}'
```

Append the live response to `results/hf_space_smoke.log` so judges can see it without
hitting the URL themselves.

---

## 2) Push the Gradio demo Space (optional, for click-through reviewers)

```bash
# Create the Space (one-time, on huggingface.co/new-space)
#    SDK: Gradio · SDK Version: 4.44.0 · Hardware: CPU Basic

git remote add hf-demo https://huggingface.co/spaces/KINGKK007/aic-incident-command-center

# The Gradio entry point at the repo root is app.py. The Space-level
# README is hf_space_readme.md — copy it into place before pushing.
cp hf_space_readme.md README.md
git push hf-demo main
git checkout README.md   # restore the canonical project README
```

### Optional Space-level secrets

For the full LLM-backed agent path (`use_llm_agents=True`), set in the Space's **Settings → Variables and secrets**:

| Name | Purpose |
|---|---|
| `OPENAI_API_KEY` | Powers `scripts/openai_baseline.py` if a reviewer runs the OpenAI baseline live in Spaces. |
| `ANTHROPIC_API_KEY` | Powers Claude-backed sub-agents when `use_llm_agents=True`. |

Both are **optional** — the demo runs end-to-end with heuristic specialists by default.

---

## 3) Self-host the FastAPI environment with Docker

The repo-root [`Dockerfile`](../Dockerfile) builds the OpenEnv FastAPI server (the same
image the canonical Space runs).

```bash
docker build -t aic-env-api .
docker run --rm -p 8000:8000 aic-env-api
```

Smoke test:

```bash
curl -s http://localhost:8000/health
# → {"status":"ok","active_envs":0}

ENV_ID=$(curl -sX POST http://localhost:8000/reset \
  -H 'Content-Type: application/json' \
  -d '{"episode_id":0,"base_seed":42,"fault_mode":"cascading_failure"}' | jq -r .env_id)

curl -s http://localhost:8000/state/$ENV_ID | jq '.state | keys'
```

For HF Docker Spaces, `Dockerfile` must live at the repo root (it already does).

---

## 4) Run the FastAPI server without Docker (fastest local dev)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_env_server.py     # starts uvicorn on http://localhost:8000

# or, equivalently:
uvicorn aic.server.env_api:app --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Notes |
|---|---|---|
| `POST` | `/reset` | Body: `{episode_id, base_seed, fault_mode, scenario_id?}` → returns `{env_id, observation}` |
| `POST` | `/step` | Body: `{env_id, action}` where `action` is an `OrchestratorDecision` |
| `GET`  | `/state/{env_id}` | Full structured state (rubric requirement) |
| `GET`  | `/render/{env_id}` | ANSI render |
| `DELETE` | `/env/{env_id}` | Free resources |
| `GET`  | `/health` | Liveness |

---

## 5) Run the Gradio demo container locally

The demo's container definition is in [`deploy/Dockerfile`](Dockerfile) and serves
`app.py` on port 7860.

```bash
docker build -f deploy/Dockerfile -t aic-gradio-demo .
docker run --rm -p 7860:7860 aic-gradio-demo
# open http://localhost:7860
```

---

## 6) OpenEnv compliance check

`AICEnvironment` subclasses `openenv.env.Env` and exposes the full contract:

```python
from openenv.env import Env
from aic.env.aic_environment import AICEnvironment

assert issubclass(AICEnvironment, Env)

env = AICEnvironment(episode_id=0, base_seed=42, fault_mode="cascading_failure")
obs = env.reset(seed=42)
state = env.state()        # required by the latest OpenEnv spec
ansi = env.render()        # ANSI render
```

Manifest: [`openenv.yaml`](../openenv.yaml). Validate with:

```bash
./.venv/bin/python -m openenv validate openenv.yaml > results/openenv_validate.log
```

(The repo also ships `results/openenv_validate.log` from the most recent run.)

---

## 7) Push the trained model to the Hub

```bash
# 1. Export and merge the GRPO LoRA adapter
./.venv/bin/python eval/test_export.py --source checkpoints/grpo --merge

# 2. Push to your namespace
./.venv/bin/python eval/test_export.py \
    --source checkpoints/grpo \
    --push --hub-repo <your-username>/aic-orchestrator
```

Then update `app.py` (or your inference path) to load the trained model.

---

## 8) Monitoring dashboard (optional)

```bash
streamlit run dashboard/app.py
```

The dashboard surfaces:

- live tracking of all 12 KPIs
- trust-score evolution per agent
- reward decomposition (R1–R8 + R9)
- agent recommendation history per step
- episode replay

---

*Companion docs:* [`README.md`](../README.md) · [`DESIGN.md`](../DESIGN.md) ·
[`COLAB_GPU_RUNBOOK.md`](../COLAB_GPU_RUNBOOK.md) · [`openenv.yaml`](../openenv.yaml)
