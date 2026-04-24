co# Quick Fixes Needed (Code Changes)

**Purpose**: Specific code changes to make training actually work  
**Time**: 1-2 hours of coding + 8-10 hours of GPU training

---

## 1. Fix SFT Data Generation (30 mins)

### File: `aic/training/generate_sft_data.py`

**Current Problem**: Only generates 40 examples from 1 scenario

**Fix**:
```python
# CHANGE THIS:
def generate_sft_dataset(config: TrainingConfig | None = None) -> Path:
    if config is None:
        config = TrainingConfig()
    
    # OLD: Only 32 episodes, one scenario
    for episode_id in range(config.sft_num_episodes):  # 32 episodes
        env = AICEnvironment(
            episode_id=episode_id,
            fault_mode=config.fault_mode,  # Always "cascading_failure"
            ...
        )

# TO THIS:
def generate_sft_dataset(config: TrainingConfig | None = None) -> Path:
    if config is None:
        config = TrainingConfig()
    
    # NEW: 100+ episodes, all scenarios
    scenarios = [
        "cascading_failure",
        "memory_leak", 
        "db_connection_saturation",
        "network_storm",
        "schema_migration_failure",
        "credential_compromise"
    ]
    
    episodes_per_scenario = 20  # 20 * 6 = 120 total
    episode_id = 0
    
    for scenario in scenarios:
        for _ in range(episodes_per_scenario):
            env = AICEnvironment(
                episode_id=episode_id,
                fault_mode=scenario,  # Rotate through scenarios
                drift_type="field_rename" if episode_id % 3 == 0 else None,  # Add drift
                ...
            )
            
            # Generate multiple steps per episode (not just step 0)
            obs = env.reset()
            for step in range(min(5, SLA_STEPS)):  # First 5 steps
                # Generate training example
                record = {...}
                f.write(json.dumps(record) + "\n")
                
                # Step environment
                action = _generate_diverse_action(obs, step)  # Not always same agent
                obs, _, done, _ = env.step(action)
                if done:
                    break
            
            episode_id += 1

def _generate_diverse_action(obs, step):
    """Generate diverse actions, not always network_agent"""
    candidates = obs["candidate_recommendations"]
    
    # Vary selection strategy
    if step % 4 == 0:
        # Select highest confidence non-adversarial
        return max((c for c in candidates if c["agent"] != "adversarial_agent"), 
                   key=lambda x: x["confidence"])
    elif step % 4 == 1:
        # Override adversarial if present
        adv = next((c for c in candidates if c["agent"] == "adversarial_agent"), None)
        if adv:
            return {"override_adversary": True, ...}
    elif step % 4 == 2:
        # Select by target metrics
        return _select_by_metrics(candidates, obs["current_metrics"])
    else:
        # Random valid selection
        return random.choice([c for c in candidates if c["risk"] < 0.5])
```

---

## 2. Fix Model Selection (5 mins)

### File: `aic/training/config.py`

**Current Problem**: Code says Qwen but actually uses tiny-gpt2

**Fix**:
```python
# CHANGE THIS:
@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"  # Says this but doesn't use it

# ENSURE THIS IS ACTUALLY USED:
# In run_sft.py and train_grpo.py, verify:
model, tokenizer, _ = load_model_and_tokenizer(config)

# In modeling_unsloth.py, ensure:
def load_model_and_tokenizer(config):
    model_name = config.model_name  # NOT hardcoded "sshleifer/tiny-gpt2"
    
    if config.use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,  # Use config value
                max_seq_length=config.max_prompt_length,
                dtype=None,
                load_in_4bit=True,
            )
        except ImportError:
            # Fallback to transformers
            pass
    
    # Standard transformers path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)  # Use config
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer, {"model_name": model_name}
```

---

## 3. Fix Benchmark Sample Size (5 mins)

### File: `scripts/run_final_benchmark.py`

**Current Problem**: Only 3 episodes per policy

**Fix**:
```python
# CHANGE THIS:
def run_benchmark(num_episodes=3):  # Too small!

# TO THIS:
def run_benchmark(num_episodes=30):  # Statistically meaningful
    
    # Also add all scenarios
    scenarios = [
        "cascading_failure",
        "memory_leak",
        "db_connection_saturation", 
        "network_storm",
        "schema_migration_failure",
        "credential_compromise"
    ]
    
    results = []
    for policy_name in ["baseline_frozen", "baseline_adaptive", "trained_grpo"]:
        for scenario in scenarios:
            for episode_id in range(num_episodes // len(scenarios)):
                # Run episode
                result = run_single_episode(policy_name, scenario, episode_id)
                results.append(result)
    
    # Add statistical testing
    from scipy import stats
    baseline_rewards = [r["reward"] for r in results if r["policy"] == "baseline_frozen"]
    trained_rewards = [r["reward"] for r in results if r["policy"] == "trained_grpo"]
    
    t_stat, p_value = stats.ttest_ind(baseline_rewards, trained_rewards)
    
    summary = {
        "baseline_mean": np.mean(baseline_rewards),
        "trained_mean": np.mean(trained_rewards),
        "improvement": np.mean(trained_rewards) - np.mean(baseline_rewards),
        "p_value": p_value,
        "significant": p_value < 0.05
    }
    
    return results, summary
```

---

## 4. Add Trained Policy to Benchmark (15 mins)

### File: `aic/evals/benchmark_suite.py`

**Current Problem**: No trained policy in benchmark

**Fix**:
```python
# ADD THIS:
class TrainedGRPOPolicy:
    """Policy using trained GRPO model"""
    
    def __init__(self, checkpoint_path="checkpoints/grpo"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model.eval()
    
    def select_action(self, obs):
        from aic.training.prompting import build_orchestrator_prompt
        from aic.schemas.actions import OrchestratorDecision
        
        prompt = build_orchestrator_prompt(obs)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True
            )
        
        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            # Extract JSON from completion
            json_start = completion.find("{")
            json_end = completion.rfind("}") + 1
            json_str = completion[json_start:json_end]
            decision = OrchestratorDecision.model_validate_json(json_str)
            return decision.model_dump()
        except Exception:
            # Fallback to safe action
            return {"selected_recommendation_id": 0, "override_adversary": False}

# UPDATE benchmark runner:
def run_benchmark_suite():
    policies = {
        "baseline_frozen": BaselineFrozenTrustPolicy(),
        "baseline_adaptive": BaselineAdaptiveTrustPolicy(),
        "trained_grpo": TrainedGRPOPolicy(),  # ADD THIS
    }
    ...
```

---

## 5. Fix README Claims (15 mins)

### File: `README.md`

**Changes Needed**:

```markdown
# REMOVE THESE FALSE CLAIMS:

❌ "640 records in `artifacts/sft/`"
❌ "GRPO training setup ✅"  (if not run)
❌ "Model export validation ✅" (if not validated)

# REPLACE WITH HONEST STATEMENTS:

✅ "SFT dataset: XXX examples across 6 scenarios"
✅ "Model: Qwen/Qwen2-0.5B-Instruct (500M parameters)"
✅ "Training: X hours on T4 GPU"
✅ "Benchmark: 30 episodes per policy across 6 scenarios"

# UPDATE BENCHMARK TABLE:

| Policy | Episodes | Avg Reward | Success Rate | Improvement |
|--------|----------|------------|--------------|-------------|
| baseline_frozen | 30 | -287.4 | 0.0% | baseline |
| baseline_adaptive | 30 | -291.6 | 0.0% | -1.5% |
| **trained_grpo** | **30** | **-XXX.X** | **X.X%** | **+X.X%** |

# ADD STATISTICAL SIGNIFICANCE:

**Statistical Test**: t-test, p-value = 0.XXX (significant at α=0.05)
**Effect Size**: Cohen's d = X.XX (medium/large effect)
```

---

## 6. Add Training Progress Logging (10 mins)

### File: `aic/training/train_grpo.py`

**Add This**:
```python
def run_grpo(config: TrainingConfig | None = None) -> Path:
    ...
    
    # ADD: Progress tracking
    progress_log = []
    
    class ProgressCallback:
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                progress_log.append({
                    "step": state.global_step,
                    "reward": logs.get("reward", 0),
                    "loss": logs.get("loss", 0),
                    "timestamp": time.time()
                })
                
                # Save progress periodically
                if state.global_step % 10 == 0:
                    with open("logs/grpo_progress.jsonl", "a") as f:
                        f.write(json.dumps(progress_log[-1]) + "\n")
    
    trainer = GRPOTrainer(
        ...
        callbacks=[ProgressCallback()],
    )
    
    trainer.train()
    
    # Save final progress summary
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump({
            "total_steps": len(progress_log),
            "final_reward": progress_log[-1]["reward"],
            "reward_improvement": progress_log[-1]["reward"] - progress_log[0]["reward"],
            "training_time_hours": (progress_log[-1]["timestamp"] - progress_log[0]["timestamp"]) / 3600
        }, f, indent=2)
```

---

## 7. Add Evidence Manifest Generator (15 mins)

### File: `scripts/generate_evidence_manifest.py` (NEW)

```python
#!/usr/bin/env python3
"""Generate evidence manifest for hackathon submission"""
import json
from pathlib import Path
import pandas as pd

def generate_manifest():
    manifest = {
        "project": "Adaptive Incident Choreographer (AIC)",
        "submission_date": "2026-04-24",
        "evidence": {}
    }
    
    # 1. Training data
    sft_path = Path("artifacts/sft/orchestrator_sft.jsonl")
    if sft_path.exists():
        with open(sft_path) as f:
            num_examples = sum(1 for _ in f)
        manifest["evidence"]["sft_data"] = {
            "path": str(sft_path),
            "num_examples": num_examples,
            "status": "✅" if num_examples >= 500 else "⚠️"
        }
    
    # 2. Trained model
    grpo_path = Path("checkpoints/grpo")
    if grpo_path.exists():
        metadata_path = grpo_path / "grpo_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            manifest["evidence"]["trained_model"] = {
                "path": str(grpo_path),
                "model_name": metadata.get("model_name"),
                "status": "✅"
            }
    
    # 3. Benchmark results
    benchmark_path = Path("results/benchmark_summary.csv")
    if benchmark_path.exists():
        df = pd.read_csv(benchmark_path)
        trained_row = df[df["policy"] == "trained_grpo"]
        if not trained_row.empty():
            manifest["evidence"]["benchmark"] = {
                "path": str(benchmark_path),
                "trained_success_rate": float(trained_row["success_rate"].iloc[0]),
                "trained_avg_reward": float(trained_row["avg_reward"].iloc[0]),
                "status": "✅"
            }
    
    # 4. Plots
    for plot_name in ["reward_curve.png", "verifier_pass_rate.png"]:
        plot_path = Path("results") / plot_name
        if plot_path.exists():
            manifest["evidence"][plot_name] = {
                "path": str(plot_path),
                "status": "✅"
            }
    
    # Save manifest
    with open("results/evidence_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Generate markdown version
    md = "# Evidence Manifest\n\n"
    for key, value in manifest["evidence"].items():
        status = value.get("status", "❌")
        md += f"- {status} **{key}**: `{value['path']}`\n"
    
    with open("results/evidence_manifest.md", "w") as f:
        f.write(md)
    
    print("Evidence manifest generated!")
    print(f"Complete: {sum(1 for v in manifest['evidence'].values() if v.get('status') == '✅')}/{len(manifest['evidence'])}")

if __name__ == "__main__":
    generate_manifest()
```

---

## 🚀 Execution Order

### Step 1: Code Changes (1-2 hours)
```bash
# 1. Fix SFT data generation
# Edit aic/training/generate_sft_data.py (changes above)

# 2. Fix model selection  
# Edit aic/training/config.py and modeling_unsloth.py

# 3. Fix benchmark
# Edit scripts/run_final_benchmark.py

# 4. Add trained policy
# Edit aic/evals/benchmark_suite.py

# 5. Add progress logging
# Edit aic/training/train_grpo.py

# 6. Create evidence generator
# Create scripts/generate_evidence_manifest.py

# 7. Test locally
python aic/training/generate_sft_data.py
# Verify: wc -l artifacts/sft/orchestrator_sft.jsonl shows 500+
```

### Step 2: GPU Training (8-10 hours)
```bash
# On Colab with T4 GPU:

# 1. Upload code
# 2. Install requirements
pip install -r requirements.txt

# 3. Run SFT
python aic/training/run_sft.py

# 4. Run GRPO
python aic/training/train_grpo.py

# 5. Run benchmark
python scripts/run_final_benchmark.py --episodes 30

# 6. Generate evidence
python scripts/generate_evidence_manifest.py
```

### Step 3: Update Documentation (30 mins)
```bash
# 1. Fix README claims
# 2. Add real benchmark results
# 3. Update evidence manifest
# 4. Commit and push
```

---

## ✅ Verification Checklist

After making changes, verify:

- [ ] `wc -l artifacts/sft/orchestrator_sft.jsonl` shows 500+
- [ ] `cat checkpoints/sft/sft_metadata.json` shows Qwen (not tiny-gpt2)
- [ ] `ls checkpoints/grpo/` shows trained model files
- [ ] `cat results/benchmark_summary.csv` includes trained_grpo row
- [ ] `cat results/evidence_manifest.json` shows all ✅
- [ ] README claims match actual artifacts
- [ ] No false "✅" marks in documentation

---

**Time Investment**: 2 hours coding + 10 hours GPU = 12 hours total  
**Result**: Submission-ready project with proof of training

**DO THESE FIXES BEFORE TRAINING TO AVOID WASTED GPU TIME**
