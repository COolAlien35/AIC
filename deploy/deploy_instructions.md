# Deployment Instructions — AIC on HuggingFace Spaces

## Option 1: Gradio SDK (Recommended)

HuggingFace Spaces natively supports Gradio apps. The `app.py` at the repo root
is already configured as a Gradio application.

### Steps

1. **Create a new Space** on [huggingface.co/new-space](https://huggingface.co/new-space):
   - SDK: **Gradio**
   - SDK Version: **4.44.0**
   - Hardware: **CPU Basic** (free tier works)

2. **Push the repo**:
   ```bash
   # Add HF Space as a remote
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/AIC

   # Copy the HF Space README as the root README for the space
   cp hf_space_readme.md README.md

   # Push
   git push hf main
   ```

3. **Verify** at `https://huggingface.co/spaces/YOUR_USERNAME/AIC`

### Environment Variables (Optional)

If using LLM-backed agents, set in Space settings:
- `ANTHROPIC_API_KEY` — for Claude-backed sub-agents

## Option 2: Docker SDK

Use the `deploy/Dockerfile` for a custom Docker-based Space:

1. Create a Space with SDK: **Docker**
2. Ensure `Dockerfile` is at the repo root (or copy from `deploy/`)
3. Push to the Space remote

## Option 3: FastAPI Server

For programmatic access, use the FastAPI environment server:

```bash
# Run locally
python scripts/run_env_server.py

# API endpoints
POST /reset   — Create and reset a new environment
POST /step    — Step the environment with an action
GET  /render/{env_id}  — Render current state
GET  /health  — Health check
```

## OpenEnv Registration

AICEnvironment is OpenEnv-compliant. To register with an OpenEnv registry:

```python
from openenv.env import Env
from aic.env.aic_environment import AICEnvironment

# Verify compliance
assert issubclass(AICEnvironment, Env)

# The environment exposes:
# - state_space: dict describing observation schema
# - action_space: dict describing action schema  
# - episode_max_length: int (20 steps)
```

## Trained Model Deployment

To serve the trained model alongside the environment:

1. Export the LoRA adapter:
   ```bash
   python eval/test_export.py --source checkpoints/grpo --merge
   ```

2. Push model to HF Hub:
   ```bash
   python eval/test_export.py --source checkpoints/grpo --push --hub-repo YOUR_USERNAME/aic-orchestrator
   ```

3. Update `app.py` to load the trained model for inference.

## Monitoring

The Streamlit dashboard provides real-time monitoring:

```bash
streamlit run dashboard/app.py
```

Features:
- Live metric tracking (12 KPIs)
- Trust score evolution
- Reward decomposition (R1–R8)
- Agent recommendation history
- Episode replay
