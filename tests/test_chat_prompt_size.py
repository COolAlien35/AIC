"""Regression test: the compact chat prompt must fit within
``max_prompt_length=1024`` for every canonical scenario. The original outage
silently truncated a 2.5k-token prompt to 256 tokens; this guards against
that re-emerging.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from aic.env.aic_environment import AICEnvironment  # noqa: E402
from aic.training.prompting import (  # noqa: E402
    build_chat_messages_compact,
    build_compact_user_text,
)
from aic.training.scenario_contract import (  # noqa: E402
    CANONICAL_SCENARIO_IDS,
    SCENARIO_TRAINING_META,
)


def _approx_token_count(text: str) -> int:
    """Cheap token estimator: ~4 chars per token for English/JSON-ish text.

    Sufficient for a regression bound; does not require downloading a
    tokenizer at test time.
    """
    return max(1, len(text) // 4)


@pytest.mark.parametrize("scenario_id", CANONICAL_SCENARIO_IDS)
def test_compact_user_text_fits_1024(scenario_id):
    meta = SCENARIO_TRAINING_META[scenario_id]
    env = AICEnvironment(
        episode_id=0,
        base_seed=42,
        fault_mode=meta.fault_injector_mode,
        use_llm_agents=False,
        manage_trust_scores=False,
        scenario_id=scenario_id,
    )
    obs = env.reset()
    text = build_compact_user_text(obs)
    approx_tokens = _approx_token_count(text)
    assert approx_tokens <= 900, (
        f"Compact user text for scenario {scenario_id} ({meta.scenario_name}) "
        f"is too large (~{approx_tokens} tokens). Trim more fields in prompting.py."
    )
    messages = build_chat_messages_compact(obs)
    total_chars = sum(len(m["content"]) for m in messages)
    approx_total_tokens = _approx_token_count(
        " ".join(m["content"] for m in messages)
    )
    assert approx_total_tokens <= 1024, (
        f"Compact chat messages for scenario {scenario_id} (~{approx_total_tokens} "
        f"tokens) exceed 1024. total_chars={total_chars}"
    )
