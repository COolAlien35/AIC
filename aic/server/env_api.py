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
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b0f0c;
      --bg2: #0f1510;
      --card: rgba(17, 22, 15, 0.78);
      --line: rgba(255,255,255,0.10);
      --line2: rgba(255,255,255,0.14);
      --text: #e7ece5;
      --muted: #aab2a7;
      --muted2: #6f7a6c;
      --green: #34d399;
      --green2: #10b981;
      --amber: #fbbf24;
      --shadow: 0 1px 0 rgba(255,255,255,0.03), 0 18px 44px rgba(0,0,0,0.35);
      --mono: \"JetBrains Mono\", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    }}
    body {{
      margin: 0;
      background:
        radial-gradient(ellipse at 15% -10%, rgba(16,185,129,0.10), transparent 55%),
        radial-gradient(ellipse at 85% 110%, rgba(20,184,166,0.06), transparent 60%),
        linear-gradient(180deg, var(--bg2), var(--bg));
      color: var(--text);
      font-family: var(--sans);
      line-height: 1.55;
    }}
    .wrap {{ max-width: 1040px; margin: 0 auto; padding: 34px 22px 64px; }}

    .top {{
      display: flex; align-items: flex-start; justify-content: space-between; gap: 16px;
      padding: 18px 18px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(11,15,12,0.35);
      backdrop-filter: blur(12px) saturate(140%);
      -webkit-backdrop-filter: blur(12px) saturate(140%);
      box-shadow: var(--shadow);
    }}
    .brand {{ display: flex; gap: 12px; align-items: center; }}
    .mark {{
      width: 34px; height: 34px; border-radius: 10px;
      background: linear-gradient(135deg, var(--green2), var(--green));
      box-shadow: 0 0 0 1px rgba(52,211,153,0.30), 0 0 24px rgba(52,211,153,0.14);
      position: relative;
      flex: 0 0 auto;
    }}
    .mark:after {{
      content: \"\";
      position: absolute;
      inset: 8px;
      border-radius: 6px;
      background: rgba(11,15,12,0.85);
      box-shadow: inset 0 0 0 1.5px rgba(52,211,153,0.55);
    }}
    h1 {{ font-size: 20px; margin: 0; letter-spacing: -0.02em; }}
    .sub {{ color: var(--muted2); margin-top: 6px; font-size: 13px; }}
    .actions {{ display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }}
    .btn {{
      display: inline-flex; align-items: center; gap: 8px;
      padding: 8px 12px;
      border-radius: 10px;
      border: 1px solid var(--line2);
      background: rgba(255,255,255,0.02);
      color: var(--text);
      text-decoration: none;
      font-size: 12.5px;
      font-weight: 600;
      transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
    }}
    .btn:hover {{
      transform: translateY(-1px);
      border-color: rgba(52,211,153,0.28);
      background: rgba(52,211,153,0.06);
    }}
    .btn.primary {{
      border-color: rgba(52,211,153,0.30);
      background: rgba(52,211,153,0.10);
      color: var(--green);
    }}

    .meta {{ margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap; }}

    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 16px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 16px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px) saturate(140%);
      -webkit-backdrop-filter: blur(10px) saturate(140%);
    }}
    .k {{
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      color: var(--muted2);
    }}
    .v {{ margin-top: 10px; font-family: var(--mono); font-size: 12.5px; white-space: pre-wrap; color: rgba(231,236,229,0.92); }}
    a {{ color: var(--green); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ font-family: var(--mono); font-size: 12.5px; }}
    .pill {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      border: 1px solid rgba(52,211,153,0.25);
      background: rgba(52,211,153,0.08);
      color: var(--green);
      padding: 6px 10px;
      border-radius: 999px;
      font-family: var(--mono);
      font-size: 12px;
    }}
    .pill code {{ color: inherit; }}
    .warn {{ border-color: rgba(251,191,36,0.35); background: rgba(251,191,36,0.08); color: var(--amber); }}
    @media (max-width: 900px) {{
      .top {{ flex-direction: column; align-items: stretch; }}
      .actions {{ justify-content: flex-start; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="brand">
        <span class="mark" aria-hidden="true"></span>
        <div>
          <h1>AIC — Adaptive Incident Choreographer (OpenEnv Environment)</h1>
          <div class="sub">Judge-facing FastAPI env server (reset/step/state/render). No heavy UI—use the dashboard Space for visuals.</div>
        </div>
      </div>
      <div class="actions">
        <a class="btn primary" href="/docs">Open API docs</a>
        <a class="btn" href="/health">Health</a>
        <a class="btn" href="https://huggingface.co/spaces/KINGKK007/aic-results-dashboard" target="_blank" rel="noopener">Results dashboard</a>
        <a class="btn" href="https://github.com/COolAlien35/AIC" target="_blank" rel="noopener">GitHub</a>
      </div>
    </div>

    <div class="meta">
      <div class="pill">Status: <code>ok</code> · Active envs: <code>{len(_ENV_REGISTRY)}</code></div>
      <div class="pill warn">Tip: <code>/docs</code> is interactive; this page is quick links + smoke test</div>
    </div>

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
