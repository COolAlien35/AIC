"""FastAPI service exposing reset/step/render operations for AICEnvironment."""
from __future__ import annotations

import textwrap
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Human-friendly landing page for HF Spaces."""
    endpoints_text = "\n".join(
        [
            "GET    /health",
            "POST   /reset",
            "POST   /step",
            "GET    /state/{env_id}",
            "GET    /render/{env_id}",
            "DELETE /env/{env_id}",
        ]
    )
    curl_text = "\n".join(
        [
            "HOST=\"https://kingkk007-aic-training.hf.space\"",
            "curl -s \"$HOST/health\"",
            "ENV_ID=$(curl -sX POST \"$HOST/reset\" -H 'Content-Type: application/json' \\",
            "  -d '{\"episode_id\":0,\"base_seed\":42,\"fault_mode\":\"cascading_failure\"}' | jq -r .env_id)",
            "curl -s \"$HOST/state/$ENV_ID\" | jq '.state | {step, health_score, is_within_sla}'",
        ]
    )
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AIC OpenEnv Environment</title>
  <style>
    :root {{
      --bg:#0b0f0c; --card:#11160f; --line:rgba(255,255,255,0.10);
      --text:#e7ece5; --muted:#9aa59a; --green:#34d399; --amber:#fbbf24;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Inter, Roboto, Helvetica, Arial, sans-serif;
    }}
    body {{
      margin:0; background: radial-gradient(ellipse at 20% -10%, rgba(16,185,129,0.10), transparent 55%), var(--bg);
      color:var(--text); font-family:var(--sans); line-height:1.55;
    }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 32px 22px 56px; }}
    .hero {{ display:flex; gap:14px; align-items:center; margin-bottom: 18px; }}
    .dot {{ width:10px; height:10px; border-radius:999px; background:var(--green); box-shadow:0 0 18px rgba(52,211,153,0.35); }}
    h1 {{ font-size: 22px; margin: 0; letter-spacing:-0.01em; }}
    p {{ color: var(--muted); margin: 10px 0 0; }}
    .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 18px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px 16px; }}
    .k {{ font-family: var(--mono); font-size: 11px; letter-spacing: 0.10em; text-transform: uppercase; color: var(--muted); }}
    .v {{ margin-top: 8px; font-family: var(--mono); font-size: 12.5px; white-space: pre-wrap; }}
    a {{ color: var(--green); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ font-family: var(--mono); font-size: 12.5px; }}
    .pill {{ display:inline-flex; gap:8px; align-items:center; border:1px solid rgba(52,211,153,0.25); background:rgba(52,211,153,0.08); color:var(--green); padding: 4px 10px; border-radius: 999px; font-family: var(--mono); font-size: 12px; }}
    .warn {{ border-color: rgba(251,191,36,0.35); background: rgba(251,191,36,0.08); color: var(--amber); }}
    @media (max-width: 820px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <span class="dot" aria-hidden="true"></span>
      <div>
        <h1>AIC — Adaptive Incident Choreographer (OpenEnv Environment)</h1>
        <p>This Space serves the judge-facing FastAPI environment (OpenEnv reset/step/state).</p>
      </div>
    </div>

    <div class="pill">Status: <code>ok</code> · Active envs: <code>{len(_ENV_REGISTRY)}</code></div>
    <div style="height:10px"></div>
    <div class="pill warn">Tip: use <a href="/docs">/docs</a> for interactive API explorer</div>

    <div class="grid">
      <div class="card">
        <div class="k">Quick endpoints</div>
        <div class="v">{endpoints_text}</div>
      </div>
      <div class="card">
        <div class="k">Curl smoke test</div>
        <div class="v">{curl_text}</div>
      </div>
      <div class="card">
        <div class="k">Project links</div>
        <div class="v">
GitHub: <a href="https://github.com/COolAlien35/AIC" target="_blank" rel="noopener">COolAlien35/AIC</a>\n
Dashboard: <a href="https://huggingface.co/spaces/KINGKK007/aic-results-dashboard" target="_blank" rel="noopener">KINGKK007/aic-results-dashboard</a>\n
Trained adapter: <a href="https://huggingface.co/COolAlien35/aic-grpo-adapter-14" target="_blank" rel="noopener">COolAlien35/aic-grpo-adapter-14</a>
        </div>
      </div>
      <div class="card">
        <div class="k">Notes</div>
        <div class="v">This is an environment server (not a UI). Use the dashboard Space for visuals, and /docs for interactive API testing.</div>
      </div>
    </div>
  </div>
</body>
</html>
"""
    return html

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


@app.get("/state/{env_id}")
def state_env(env_id: str) -> dict:
    """Return the structured environment state (OpenEnv ``state`` method)."""
    env = _ENV_REGISTRY.get(env_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Unknown env_id")
    return {"env_id": env_id, "state": env.state()}


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
