# 🎯 HACKATHON WIN-READY TODO LIST

**Last Updated**: April 24, 2026  
**Current Status**: Architecture Complete, Training Proof Missing  
**Time to Win-Ready**: 48 hours (with GPU access)

---

## 🚨 CRITICAL PATH (MUST DO - Blocks Submission)

### 1. Generate Proper SFT Training Data ⏰ 30 mins
**Priority**: P0 - CRITICAL  
**Blocks**: All training  
**Current**: 40 examples, 1 scenario  
**Target**: 500+ examples, all 6 scenarios

```bash
# Update generate_sft_data.py to cover all scenarios
python aic/training/generate_sft_data.py \
  --num_episodes 100 \
  --scenarios all \
  --include_adversarial_overrides \
  --include_schema_drift

# Verify output
wc -l artifacts/sft/orchestrator_sft.jsonl  # Should show 500+
```

**Acceptance Criteria**:
- [ ] 500+ training examples generated
- [ ] All 6 fault modes covered (not just cascading_failure)
- [ ] Examples include adversarial overrides (override_adversary=true)
- [ ] Examples include schema drift detection
- [ ] Diverse recommendation selections (not all network_agent)

---

### 2. Run SFT Training on Real Model ⏰ 2 hours
**Priority**: P0 - CRITICAL  
**Blocks**: GRPO training  
**Current**: tiny-gpt2 (124M toy model)  
**Target**: Qwen/Qwen2-0.5B-Instruct (500M)

```bash
# Update config.py
# model_name = "Qwen/Qwen2-0.5B-Instruct"  # NOT tiny-gpt2

# Run on Colab with T4 GPU (free tier)
python aic/training/run_sft.py
```

**Acceptance Criteria**:
- [ ] Model trained on Qwen 0.5B or Llama 3.2 1B (NOT tiny-gpt2)
- [ ] Training completes without OOM errors
- [ ] Checkpoint saved to `checkpoints/sft/`
- [ ] Training loss shows convergence
- [ ] Metadata file shows correct model name

---

### 3. Run GRPO Training ⏰ 4-6 hours
**Priority**: P0 - CRITICAL  
**Blocks**: Core value proposition  
**Current**: Never run, no checkpoints  
**Target**: Trained GRPO checkpoint with measurable improvement

```bash
# Generate GRPO prompts (if not exists)
python aic/training/train_grpo.py --generate-prompts-only

# Run GRPO training on GPU
python aic/training/train_grpo.py \
  --max_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4
```

**Acceptance Criteria**:
- [ ] GRPO training completes successfully
- [ ] Checkpoint saved to `checkpoints/grpo/`
- [ ] Training logs show reward improvement over episodes
- [ ] Curriculum progression logged (easy → medium → hard)
- [ ] Reward audit logs generated in `logs/audit/`
- [ ] grpo_metadata.json created with training details

---

### 4. Run Comprehensive Benchmark ⏰ 1 hour
**Priority**: P0 - CRITICAL  
**Blocks**: Results proof  
**Current**: 3 episodes per policy, 0% success  
**Target**: 30+ episodes, statistical significance

```bash
# Run benchmark with trained model
python scripts/run_final_benchmark.py \
  --episodes 30 \
  --scenarios all \
  --policies baseline_frozen,baseline_adaptive,trained_grpo
```

**Acceptance Criteria**:
- [ ] 30+ episodes per policy
- [ ] All 6 scenarios tested
- [ ] Trained policy shows >0% success rate
- [ ] Trained policy outperforms baseline (statistical test)
- [ ] Results saved to `results/benchmark_summary.csv`
- [ ] Per-scenario breakdown included

---

### 5. Fix README Claims ⏰ 30 mins
**Priority**: P0 - CRITICAL  
**Blocks**: Credibility  
**Current**: False claims (640 examples, completed GRPO)  
**Target**: Honest, accurate documentation

**Changes Needed**:
```markdown
# REMOVE these false claims:
- "640 records in artifacts/sft/" → Update to actual count
- "GRPO training setup ✅" → Change to "GRPO training completed ✅"
- "Model export validation ✅" → Only mark ✅ after validation

# ADD these honest statements:
- Actual SFT dataset size: XXX examples
- Training completed on: Qwen 0.5B / Llama 3.2 1B
- Benchmark results: X% success rate improvement
- Training time: X hours on T4 GPU
```

**Acceptance Criteria**:
- [ ] All quantitative claims match actual artifacts
- [ ] No "✅" marks for incomplete items
- [ ] Training results section updated with real numbers
- [ ] Benchmark table shows trained policy results
- [ ] Model size correctly stated

---

## 🔥 HIGH PRIORITY (Strongly Recommended)

### 6. Export and Validate Trained Model ⏰ 30 mins
**Priority**: P1 - HIGH  
**Why**: Proves model is deployment-ready

```bash
# Export trained GRPO model
python aic/training/export_model.py \
  --source checkpoints/grpo \
  --target exports/aic-trained

# Validate export
python eval/test_export.py \
  --source exports/aic-trained \
  --test_inference
```

**Acceptance Criteria**:
- [ ] Model exported successfully
- [ ] Reload test passes
- [ ] Inference test produces valid outputs
- [ ] Export size documented
- [ ] Inference speed benchmarked

---

### 7. Generate Training Evidence Bundle ⏰ 15 mins
**Priority**: P1 - HIGH  
**Why**: Makes judging easy

```bash
# After training completes
python run_hackathon.py verify plots demo
```

**Acceptance Criteria**:
- [ ] `results/evidence_manifest.json` created
- [ ] `results/evidence_manifest.md` created
- [ ] Reward curve shows training progression
- [ ] Verifier pass rate comparison (before/after)
- [ ] Before/after demo with real trained model

---

### 8. Document Reward Audit Results ⏰ 15 mins
**Priority**: P1 - HIGH  
**Why**: Proves anti-hacking works

```bash
# Analyze audit logs
python -c "
import json
from pathlib import Path

audit_files = list(Path('logs/audit').glob('audit_ep*.jsonl'))
total_episodes = len(audit_files)
flagged = sum(1 for f in audit_files if 'flagged' in f.read_text())

print(f'Total episodes: {total_episodes}')
print(f'Flagged for hacking: {flagged}')
print(f'Clean rate: {(total_episodes-flagged)/total_episodes*100:.1f}%')
"
```

**Acceptance Criteria**:
- [ ] Audit summary added to README
- [ ] Detection rate documented
- [ ] Example of flagged episode shown
- [ ] Mitigation strategy explained

---

### 9. Add Trained Model to Gradio Demo ⏰ 1 hour
**Priority**: P1 - HIGH  
**Why**: Live demo is compelling

```python
# Update app.py to load trained model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("checkpoints/grpo")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/grpo")

def step_with_trained_model(obs):
    prompt = build_orchestrator_prompt(obs)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    decision = tokenizer.decode(outputs[0])
    return decision
```

**Acceptance Criteria**:
- [ ] Gradio app loads trained model
- [ ] "Use Trained Model" toggle added
- [ ] Side-by-side comparison (baseline vs trained)
- [ ] Inference works without errors

---

## ⚡ MEDIUM PRIORITY (Nice to Have)

### 10. Ablation Studies ⏰ 2 hours
**Priority**: P2 - MEDIUM  
**Why**: Shows what components matter

```bash
# Test different configurations
python scripts/run_final_benchmark.py --policy baseline_frozen
python scripts/run_final_benchmark.py --policy sft_only
python scripts/run_final_benchmark.py --policy sft_plus_grpo
python scripts/run_final_benchmark.py --policy grpo_no_curriculum
python scripts/run_final_benchmark.py --policy grpo_no_audit
```

**Acceptance Criteria**:
- [ ] Baseline vs SFT-only vs SFT+GRPO comparison
- [ ] With/without curriculum comparison
- [ ] With/without reward audit comparison
- [ ] Results table in README

---

### 11. Improve Visualizations ⏰ 1 hour
**Priority**: P2 - MEDIUM  
**Why**: Makes results clearer

```python
# Add to generate_plots.py
- Reward components breakdown (R1-R8 over time)
- Trust score evolution per agent
- Success rate by scenario
- Curriculum tier progression
```

**Acceptance Criteria**:
- [ ] Reward components plot
- [ ] Trust evolution plot
- [ ] Per-scenario success rates
- [ ] Curriculum progression chart

---

### 12. Add Statistical Significance Testing ⏰ 30 mins
**Priority**: P2 - MEDIUM  
**Why**: Proves improvement is real

```python
from scipy import stats

baseline_rewards = [...]  # from benchmark
trained_rewards = [...]   # from benchmark

t_stat, p_value = stats.ttest_ind(baseline_rewards, trained_rewards)
print(f"p-value: {p_value:.4f}")
print(f"Significant: {p_value < 0.05}")
```

**Acceptance Criteria**:
- [ ] T-test on reward distributions
- [ ] P-value < 0.05 for significance
- [ ] Effect size (Cohen's d) calculated
- [ ] Results added to README

---

## 📊 EXECUTION TIMELINE

### Minimum Viable (48 hours to submission-ready)

| Hour | Task | Deliverable |
|------|------|-------------|
| 0-0.5 | Generate SFT data | 500+ examples |
| 0.5-2.5 | Run SFT training | Qwen 0.5B checkpoint |
| 2.5-8.5 | Run GRPO training | Trained GRPO checkpoint |
| 8.5-9.5 | Run benchmark | 30 episodes × 3 policies |
| 9.5-10 | Update README | Accurate claims |
| 10-10.5 | Export model | Validated export |
| 10.5-11 | Generate evidence | Plots + manifest |
| 11-11.5 | Document audit | Audit summary |

**Total**: 11.5 hours (mostly GPU training time)

---

### Competitive (72 hours to strong submission)

Add to above:
- Hour 12-13: Add trained model to demo
- Hour 14-16: Ablation studies
- Hour 17-18: Better visualizations
- Hour 18-19: Statistical testing
- Hour 19-20: Polish documentation

---

## 🎯 SUCCESS METRICS

### Minimum for Submission
- [ ] Trained model exists (not tiny-gpt2)
- [ ] Success rate > 0% (currently 0%)
- [ ] Trained outperforms baseline (any amount)
- [ ] README claims match artifacts
- [ ] Evidence manifest generated

### Competitive Submission
- [ ] Success rate improvement > 10%
- [ ] Statistical significance (p < 0.05)
- [ ] Ablation studies show component value
- [ ] Live demo with trained model
- [ ] Comprehensive audit evidence

### Prize-Winning Submission
- [ ] Success rate improvement > 25%
- [ ] Strong effect size (Cohen's d > 0.8)
- [ ] All ablations documented
- [ ] Publication-quality plots
- [ ] Deployment-ready artifacts

---

## 🚀 QUICK START (Next 30 Minutes)

```bash
# 1. Generate proper SFT data
python aic/training/generate_sft_data.py --num_episodes 100

# 2. Open Colab notebook
# - Upload code to Colab
# - Enable T4 GPU
# - Install requirements

# 3. Update config
# Edit aic/training/config.py:
# model_name = "Qwen/Qwen2-0.5B-Instruct"

# 4. Start SFT training
python aic/training/run_sft.py

# 5. While training, fix README
# Remove false claims, prepare results section
```

---

## 📋 CHECKLIST FOR SUBMISSION

### Before Submitting
- [ ] GRPO training completed successfully
- [ ] Benchmark shows trained > baseline
- [ ] README updated with real results
- [ ] All plots regenerated with trained model
- [ ] Evidence manifest created
- [ ] Export validated
- [ ] Demo works with trained model
- [ ] No false claims in documentation

### Submission Package
- [ ] GitHub repo with all code
- [ ] `checkpoints/grpo/` with trained model
- [ ] `results/` with plots and benchmarks
- [ ] `BRUTAL_HACKATHON_AUDIT.md` (shows self-awareness)
- [ ] `README.md` with honest results
- [ ] HuggingFace Space demo link
- [ ] Video demo (optional but recommended)

---

## 🎬 FINAL NOTES

### If You Can't Get GPU Access
1. **Be Brutally Honest**: "Training pipeline validated, GPU execution pending"
2. **Pivot Framing**: "Production-grade RL platform for incident response"
3. **Emphasize Architecture**: Multi-agent orchestration, safety guarantees
4. **Show Scaffolding Quality**: 248 tests, comprehensive design

### If You Have 48 Hours + GPU
1. **Execute Critical Path**: Items 1-5 above
2. **Don't Skip Benchmark**: Need proof of improvement
3. **Fix README First**: Credibility matters
4. **Document Everything**: Evidence manifest is key

### If You Have 72 Hours + GPU
1. **Do Everything Above**
2. **Add Ablations**: Show what components matter
3. **Polish Demo**: Live trained model is compelling
4. **Statistical Rigor**: Prove significance

---

**Bottom Line**: You have excellent architecture. Now prove it works with training.

**Time Investment**: 11.5 hours (mostly GPU time) to go from "incomplete" to "submission-ready"

**Win Probability**: 
- As-is: 5%
- With critical path: 60%
- With high priority: 80%

**DO NOT SUBMIT WITHOUT COMPLETING CRITICAL PATH (Items 1-5)**
