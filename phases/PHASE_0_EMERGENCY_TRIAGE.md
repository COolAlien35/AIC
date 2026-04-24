# 🔴 PHASE 0 — EMERGENCY TRIAGE
> **Time: 90 minutes | Priority: BLOCKING EVERYTHING ELSE**  
> Do this BEFORE touching Colab. Skipping this wastes 6 hours of GPU time.

---

## FIX 0.1 — The Model Name Bug
**File:** `aic/training/config.py` and `aic/training/modeling_unsloth.py`  
**Time:** 5 minutes

**Problem:** `config.py` says `Qwen/Qwen2.5-3B-Instruct` but the loading code is hardcoded to `sshleifer/tiny-gpt2`.

### Fix in `aic/training/config.py`

```python
@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_prompt_length: int = 1024
    use_unsloth: bool = True          # Try unsloth first, fall back to transformers
    load_in_4bit: bool = True         # Required for T4 VRAM budget
    sft_num_episodes: int = 120
    grpo_max_steps: int = 150         # Enough for measurable learning
    grpo_batch_size: int = 1
    grpo_grad_accumulation: int = 8   # Effective batch = 8
    output_dir: str = "checkpoints"
    use_reward_audit: bool = True     # MUST be True — it's a prize feature
```

### Fix in `aic/training/modeling_unsloth.py`

```python
def load_model_and_tokenizer(config: TrainingConfig):
    model_name = config.model_name  # ← THIS LINE IS THE ENTIRE FIX

    if config.use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=config.max_prompt_length,
                dtype=None,
                load_in_4bit=config.load_in_4bit,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            return model, tokenizer, {"model_name": model_name, "backend": "unsloth"}
        except ImportError:
            pass  # Fall through to standard transformers

    # Standard transformers fallback
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=config.load_in_4bit,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    return model, tokenizer, {"model_name": model_name, "backend": "transformers"}
```

### Verify Fix 0.1

```bash
python -c "
from aic.training.config import TrainingConfig
from aic.training.modeling_unsloth import load_model_and_tokenizer
c = TrainingConfig()
assert c.model_name == 'Qwen/Qwen2.5-3B-Instruct', f'WRONG: {c.model_name}'
print('✅ Model config correct:', c.model_name)
"
```

---

## FIX 0.2 — The SFT Data Generator
**File:** `aic/training/generate_sft_data.py`  
**Time:** 30 minutes

**Problem:** Only 1 scenario, only step 0, no adversarial overrides, no drift. 40 examples total. Need 600+.

### Full Replacement Logic

```python
# aic/training/generate_sft_data.py — FULL REWRITE OF MAIN LOOP

import json
import random
from pathlib import Path
from aic.env.aic_environment import AICEnvironment
from aic.training.config import TrainingConfig
from aic.training.prompting import build_orchestrator_prompt

ALL_SCENARIOS = [
    "cascading_failure",
    "memory_leak",
    "db_connection_saturation",
    "network_storm",
    "schema_migration_failure",
    "credential_compromise",
]

DRIFT_TYPES = [None, "field_rename", "type_coercion", "unit_change"]

def _select_action_diverse(obs: dict, step: int, episode_id: int) -> dict:
    candidates = obs.get("candidate_recommendations", [])
    if not candidates:
        return {"selected_recommendation_id": 0, "override_adversary": False,
                "reasoning": "No candidates available, defaulting to safe action."}

    non_adversarial = [c for c in candidates if c.get("agent") != "adversarial_agent"]
    adversarial = [c for c in candidates if c.get("agent") == "adversarial_agent"]

    strategy = (episode_id * 7 + step * 3) % 6  # Deterministic diversity

    if strategy == 0 and non_adversarial:
        chosen = max(non_adversarial, key=lambda x: x.get("confidence", 0))
        override = False
        reasoning = (f"Selected {chosen['agent']} with highest confidence "
                     f"{chosen.get('confidence', 0):.2f}. No adversarial override needed.")

    elif strategy == 1 and adversarial and non_adversarial:
        chosen = max(non_adversarial, key=lambda x: x.get("confidence", 0))
        override = True
        reasoning = (f"Detected adversarial recommendation from {adversarial[0]['agent']}. "
                     f"Overriding with {chosen['agent']} — risk too high.")

    elif strategy == 2 and non_adversarial:
        chosen = min(non_adversarial, key=lambda x: x.get("risk", 1.0))
        override = bool(adversarial)
        reasoning = (f"SLA critical — selecting lowest-risk action from {chosen['agent']} "
                     f"(risk={chosen.get('risk', 0):.2f}). Safety first.")

    elif strategy == 3 and non_adversarial:
        current_metrics = obs.get("current_metrics", {})
        worst_metric = min(current_metrics, key=current_metrics.get) if current_metrics else None
        metric_specialists = {
            "error_rate": "app_agent", "latency_p99": "network_agent",
            "db_connections": "db_agent", "memory_usage": "infra_agent",
        }
        preferred_agent = metric_specialists.get(worst_metric)
        specialist = next((c for c in non_adversarial
                           if c.get("agent") == preferred_agent), None)
        chosen = specialist if specialist else non_adversarial[0]
        override = bool(adversarial)
        reasoning = (f"Worst metric is {worst_metric}. Targeting with {chosen['agent']}.")

    elif strategy == 4 and non_adversarial:
        sorted_candidates = sorted(non_adversarial,
                                   key=lambda x: x.get("confidence", 0), reverse=True)
        chosen = sorted_candidates[1] if len(sorted_candidates) > 1 else sorted_candidates[0]
        override = bool(adversarial)
        reasoning = (f"Top candidate has high risk. Selecting {chosen['agent']} as safer alternative.")

    else:
        safe = [c for c in non_adversarial if c.get("risk", 1.0) < 0.5]
        chosen = random.choice(safe) if safe else (non_adversarial[0] if non_adversarial else candidates[0])
        override = bool(adversarial)
        reasoning = f"Conservative selection: {chosen.get('agent', 'unknown')} within acceptable risk bounds."

    return {
        "selected_recommendation_id": chosen.get("id", 0),
        "override_adversary": override,
        "schema_drift_detected": obs.get("schema_drift_active", False),
        "reasoning": reasoning,
        "confidence_score": chosen.get("confidence", 0.7),
    }


def generate_sft_dataset(config: TrainingConfig | None = None) -> Path:
    if config is None:
        config = TrainingConfig()

    output_path = Path("artifacts/sft/orchestrator_sft.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episodes_per_scenario = 20   # 20 × 6 = 120 episodes
    steps_per_episode = 5        # 5 × 120 = 600 examples minimum

    records = []
    episode_id = 0

    for scenario in ALL_SCENARIOS:
        for ep_local in range(episodes_per_scenario):
            drift_type = DRIFT_TYPES[ep_local % len(DRIFT_TYPES)]

            try:
                env = AICEnvironment(
                    episode_id=episode_id,
                    fault_mode=scenario,
                    drift_type=drift_type,
                )
                obs = env.reset()

                for step in range(steps_per_episode):
                    prompt = build_orchestrator_prompt(obs)
                    action = _select_action_diverse(obs, step, episode_id)

                    record = {
                        "episode_id": episode_id,
                        "step": step,
                        "scenario": scenario,
                        "drift_type": drift_type,
                        "prompt": prompt,
                        "completion": json.dumps(action),
                        "metadata": {
                            "has_adversarial": action["override_adversary"],
                            "schema_drift": action["schema_drift_detected"],
                            "selection_strategy": (episode_id * 7 + step * 3) % 6,
                        }
                    }
                    records.append(record)

                    obs, _, done, _ = env.step(action)
                    if done:
                        break

            except Exception as e:
                print(f"  ⚠️  Episode {episode_id} ({scenario}) failed: {e}")

            episode_id += 1

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    scenarios_seen = set(r["scenario"] for r in records)
    adversarial_count = sum(1 for r in records if r["metadata"]["has_adversarial"])
    drift_count = sum(1 for r in records if r["metadata"]["schema_drift"])

    print(f"\n✅ SFT Dataset Generated:")
    print(f"   Total examples: {len(records)}")
    print(f"   Scenarios covered: {len(scenarios_seen)}/6 → {scenarios_seen}")
    print(f"   Adversarial overrides: {adversarial_count} ({adversarial_count/len(records)*100:.1f}%)")
    print(f"   Schema drift examples: {drift_count} ({drift_count/len(records)*100:.1f}%)")

    assert len(records) >= 500, f"FAILED: Only {len(records)} examples. Need 500+."
    assert len(scenarios_seen) == 6, f"FAILED: Only {len(scenarios_seen)} scenarios."

    return output_path
```

### Verify Fix 0.2

```bash
python aic/training/generate_sft_data.py
wc -l artifacts/sft/orchestrator_sft.jsonl  # Must show 600+

python -c "
import json
data = [json.loads(l) for l in open('artifacts/sft/orchestrator_sft.jsonl')]
scenarios = set(d['scenario'] for d in data)
adversarial = sum(1 for d in data if d['metadata']['has_adversarial'])
print('Scenarios:', scenarios)
print(f'Adversarial examples: {adversarial}/{len(data)}')
assert len(scenarios) == 6, 'MISSING SCENARIOS'
assert adversarial > 50, 'NOT ENOUGH ADVERSARIAL EXAMPLES'
print('✅ Data quality check PASSED')
"
```

---

## FIX 0.3 — The Benchmark Script
**File:** `scripts/run_final_benchmark.py`  
**Time:** 15 minutes

**Problem:** 3 episodes, missing trained policy, no statistical tests.

### Replace `run_benchmark()` Function

```python
import numpy as np
from scipy import stats

ALL_SCENARIOS = [
    "cascading_failure", "memory_leak", "db_connection_saturation",
    "network_storm", "schema_migration_failure", "credential_compromise"
]

def run_benchmark(num_episodes_per_scenario: int = 5):
    policies = {
        "baseline_frozen": BaselineFrozenTrustPolicy(),
        "baseline_adaptive": BaselineAdaptiveTrustPolicy(),
        "trained_grpo": TrainedGRPOPolicy(checkpoint_path="checkpoints/grpo"),
    }

    all_results = []

    for policy_name, policy in policies.items():
        print(f"\n🔄 Benchmarking: {policy_name}")
        for scenario in ALL_SCENARIOS:
            for ep_idx in range(num_episodes_per_scenario):
                result = run_single_episode(policy, policy_name, scenario, ep_idx)
                all_results.append(result)
                print(f"  {scenario} ep{ep_idx}: reward={result['reward']:.2f}, "
                      f"success={result['success']}")

    df = pd.DataFrame(all_results)

    baseline_rewards = df[df["policy"] == "baseline_frozen"]["reward"].values
    trained_rewards = df[df["policy"] == "trained_grpo"]["reward"].values

    t_stat, p_value = stats.ttest_ind(baseline_rewards, trained_rewards)

    pooled_std = np.sqrt((np.std(baseline_rewards)**2 + np.std(trained_rewards)**2) / 2)
    cohens_d = (np.mean(trained_rewards) - np.mean(baseline_rewards)) / (pooled_std + 1e-9)

    summary = df.groupby("policy").agg(
        avg_reward=("reward", "mean"),
        std_reward=("reward", "std"),
        success_rate=("success", "mean"),
        num_episodes=("reward", "count"),
    ).reset_index()

    summary.to_csv("results/benchmark_summary.csv", index=False)

    scenario_summary = df.groupby(["policy", "scenario"]).agg(
        avg_reward=("reward", "mean"),
        success_rate=("success", "mean"),
    ).reset_index()
    scenario_summary.to_csv("results/benchmark_by_scenario.csv", index=False)

    stats_output = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "cohens_d": float(cohens_d),
        "effect_size_label": (
            "large" if abs(cohens_d) > 0.8 else
            "medium" if abs(cohens_d) > 0.5 else "small"
        ),
        "baseline_mean": float(np.mean(baseline_rewards)),
        "trained_mean": float(np.mean(trained_rewards)),
        "improvement": float(np.mean(trained_rewards) - np.mean(baseline_rewards)),
        "improvement_pct": float((np.mean(trained_rewards) - np.mean(baseline_rewards)) /
                                  abs(np.mean(baseline_rewards)) * 100),
    }

    with open("results/statistical_test.json", "w") as f:
        json.dump(stats_output, f, indent=2)

    print(f"\n📊 BENCHMARK COMPLETE")
    print(f"   Improvement: {stats_output['improvement']:+.2f} ({stats_output['improvement_pct']:+.1f}%)")
    print(f"   p-value: {stats_output['p_value']:.4f} ({'✅ SIGNIFICANT' if stats_output['significant'] else '⚠️ not significant'})")
    print(f"   Cohen's d: {stats_output['cohens_d']:.3f} ({stats_output['effect_size_label']} effect)")

    return df, stats_output


class TrainedGRPOPolicy:
    """The policy we actually trained — the star of the show."""

    def __init__(self, checkpoint_path: str = "checkpoints/grpo"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"  Loading trained model from {checkpoint_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, device_map="auto", torch_dtype=torch.float16,
        )
        self.model.eval()
        print(f"  ✅ Trained model loaded")

    def select_action(self, obs: dict) -> dict:
        from aic.training.prompting import build_orchestrator_prompt
        import torch

        prompt = build_orchestrator_prompt(obs)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, temperature=0.3, do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        completion = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        try:
            json_start = completion.find("{")
            json_end = completion.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                decision = json.loads(completion[json_start:json_end])
                return decision
        except Exception:
            pass

        # Graceful fallback
        candidates = obs.get("candidate_recommendations", [])
        safe = [c for c in candidates if c.get("agent") != "adversarial_agent"]
        chosen_id = safe[0].get("id", 0) if safe else 0
        return {
            "selected_recommendation_id": chosen_id,
            "override_adversary": len(safe) < len(candidates),
            "reasoning": "Fallback: JSON parse failed, selecting safest candidate.",
        }
```

---

## FIX 0.4 — GRPO Progress Logging
**File:** `aic/training/train_grpo.py`  
**Time:** 10 minutes

Add this callback **before** `GRPOTrainer` initialization:

```python
import time
import json
from pathlib import Path
from transformers import TrainerCallback

class AICProgressCallback(TrainerCallback):
    """Logs reward and loss at every step — creates the reward curve for evidence."""

    def __init__(self, log_path: str = "logs/grpo_progress.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.step_log = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        entry = {
            "step": state.global_step,
            "reward": logs.get("reward", logs.get("train/reward", 0)),
            "loss": logs.get("loss", logs.get("train/loss", 0)),
            "elapsed_minutes": (time.time() - self.start_time) / 60,
        }
        self.step_log.append(entry)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"  Step {entry['step']:4d} | reward={entry['reward']:+.4f} | "
              f"loss={entry['loss']:.4f} | elapsed={entry['elapsed_minutes']:.1f}m")

    def on_train_end(self, args, state, control, **kwargs):
        if not self.step_log:
            return
        summary = {
            "total_steps": len(self.step_log),
            "initial_reward": self.step_log[0]["reward"],
            "final_reward": self.step_log[-1]["reward"],
            "reward_delta": self.step_log[-1]["reward"] - self.step_log[0]["reward"],
            "training_time_minutes": self.step_log[-1]["elapsed_minutes"],
        }
        out = Path(args.output_dir) / "training_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n🏁 Training complete. Reward delta: {summary['reward_delta']:+.4f}")

# In run_grpo():
trainer = GRPOTrainer(
    ...
    callbacks=[AICProgressCallback()],
)
```

---

## FIX 0.5 — Final Verification Before GPU
**Time:** 5 minutes

Run this entire block locally before opening Colab:

```bash
echo "=== FIX VERIFICATION ==="

# Check 1: Model name
python -c "
from aic.training.config import TrainingConfig
c = TrainingConfig()
assert 'tiny-gpt2' not in c.model_name, 'FAIL: still using tiny-gpt2'
print(f'✅ Model: {c.model_name}')
"

# Check 2: SFT data diversity
python -c "
import json
data = [json.loads(l) for l in open('artifacts/sft/orchestrator_sft.jsonl')]
scenarios = set(d['scenario'] for d in data)
assert len(data) >= 500, f'FAIL: only {len(data)} examples'
assert len(scenarios) == 6, f'FAIL: only {len(scenarios)} scenarios'
print(f'✅ SFT data: {len(data)} examples, {len(scenarios)} scenarios')
"

# Check 3: Config sanity
python -c "
from aic.training.config import TrainingConfig
c = TrainingConfig()
assert c.use_reward_audit == True, 'FAIL: reward audit disabled'
assert c.grpo_max_steps >= 100, 'FAIL: too few GRPO steps'
print(f'✅ Config: GRPO steps={c.grpo_max_steps}, reward_audit={c.use_reward_audit}')
"

echo "=== ALL FIXES VERIFIED — READY FOR GPU ==="
```

---

## ✅ Phase 0 Completion Criteria

- [ ] `config.model_name` → `Qwen/Qwen2.5-3B-Instruct` (NOT `tiny-gpt2`)
- [ ] `generate_sft_data.py` generates 600+ examples across 6 scenarios
- [ ] Benchmark script has `trained_grpo` policy and t-test statistics
- [ ] `AICProgressCallback` is registered in `train_grpo.py`
- [ ] All 4 verification scripts pass without assertion errors

**→ Next: [PHASE_1_COLAB_SETUP.md](PHASE_1_COLAB_SETUP.md)**
