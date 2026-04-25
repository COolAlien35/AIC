# aic/utils/logging_utils.py
"""
Structured JSON Lines logging for AIC episodes.
Each episode produces a .jsonl file with one StepRecord per line,
plus a summary JSON on finalization.
"""
import json
import time
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class StepRecord:
    """Per-step record logged during an episode."""
    episode_id: int
    step: int
    timestamp: float
    world_state: dict[str, float]
    agent_recommendations: dict[str, str]
    orchestrator_action: str
    reward_components: dict[str, float]
    reward_total: float
    trust_scores: dict[str, float]
    schema_drift_active: bool
    schema_drift_type: Optional[str]
    deadlock_detected: bool
    extra: dict[str, Any] = field(default_factory=dict)


class EpisodeLogger:
    """Logs every step of an episode to a JSON Lines file."""

    def __init__(self, log_dir: str = "logs", episode_id: int = 0):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id
        self.episode_path = self.log_dir / f"episode_{episode_id:04d}.jsonl"
        self.steps: list[StepRecord] = []

    def log_step(self, record: StepRecord) -> None:
        """Append a step record as a JSON line to the episode file."""
        self.steps.append(record)
        with open(self.episode_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def finalize(self, total_reward: float, success: bool) -> dict:
        """Write episode summary JSON and return the summary dict."""
        active_agents: list[str] = []
        for step in self.steps:
            agents = step.extra.get("active_agents") if isinstance(step.extra, dict) else None
            if isinstance(agents, list):
                for a in agents:
                    if isinstance(a, str) and a not in active_agents:
                        active_agents.append(a)
        summary = {
            "episode_id": self.episode_id,
            "total_steps": len(self.steps),
            "total_reward": total_reward,
            "success": success,
            "timestamp": time.time(),
            "active_agents": active_agents,
        }
        summary_path = self.log_dir / f"episode_{self.episode_id:04d}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary


def load_episode(log_dir: str, episode_id: int) -> list[dict]:
    """Load all step records for an episode from its JSONL file."""
    path = Path(log_dir) / f"episode_{episode_id:04d}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No log found for episode {episode_id}")
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records
