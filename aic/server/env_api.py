"""FastAPI service exposing reset/step/render operations for AICEnvironment."""
from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aic.env.aic_environment import AICEnvironment


app = FastAPI(title="AIC OpenEnv Service", version="0.1.0")
_ENV_REGISTRY: dict[str, AICEnvironment] = {}


class ResetRequest(BaseModel):
    episode_id: int = 0
    base_seed: int = 42
    fault_mode: str = "cascading_failure"
    use_llm_agents: bool = False


class StepRequest(BaseModel):
    env_id: str
    action: dict | str = Field(default_factory=dict)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "active_envs": len(_ENV_REGISTRY)}


@app.post("/reset")
def reset_env(request: ResetRequest) -> dict:
    env = AICEnvironment(
        episode_id=request.episode_id,
        base_seed=request.base_seed,
        fault_mode=request.fault_mode,
        use_llm_agents=request.use_llm_agents,
        manage_trust_scores=False,
    )
    obs = env.reset()
    env_id = str(uuid4())
    _ENV_REGISTRY[env_id] = env
    return {"env_id": env_id, "observation": obs}


@app.post("/step")
def step_env(request: StepRequest) -> dict:
    env = _ENV_REGISTRY.get(request.env_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Unknown env_id")
    obs, reward, done, info = env.step(request.action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/render/{env_id}")
def render_env(env_id: str) -> dict:
    env = _ENV_REGISTRY.get(env_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Unknown env_id")
    return {"render": env._render_ansi()}


@app.delete("/env/{env_id}")
def delete_env(env_id: str) -> dict:
    env = _ENV_REGISTRY.pop(env_id, None)
    if env is None:
        raise HTTPException(status_code=404, detail="Unknown env_id")
    return {"deleted": env_id}
