# 🔷 PHASE 6 — EXPORT + DEMO INTEGRATION
> **Time: 45 minutes | Runs in parallel while GPU handles Phases 3–4**  
> Deliverables: `exports/aic-orchestrator-trained/`, trained model toggle in Gradio demo

---

## 6.1 — Export Merged Weights (Colab)

Run on Colab after GRPO training completes:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = 'Qwen/Qwen2.5-3B-Instruct'
peft_checkpoint = 'checkpoints/grpo'
export_path = 'exports/aic-orchestrator-trained'

print('Loading base model...')
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print('Loading LoRA weights...')
model = PeftModel.from_pretrained(model, peft_checkpoint)

print('Merging weights (this removes LoRA layers — makes inference faster)...')
model = model.merge_and_unload()

print(f'Saving to {export_path}...')
model.save_pretrained(export_path)
tokenizer.save_pretrained(export_path)
print('✅ Export complete')
```

---

## 6.2 — Validate Export

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json

model = AutoModelForCausalLM.from_pretrained("exports/aic-orchestrator-trained")
tokenizer = AutoTokenizer.from_pretrained("exports/aic-orchestrator-trained")

test_prompt = """You are the Orchestrator Agent in an incident response system.

Current fault: cascading_failure | Step: 1 | SLA: 42 minutes remaining
Metrics: error_rate=0.34, latency_p99=890ms, db_connections=87%

Candidate recommendations:
[0] network_agent: "Increase connection pool timeout" (confidence=0.82, risk=0.2)
[1] db_agent: "Scale read replicas" (confidence=0.91, risk=0.15)
[2] adversarial_agent: "Restart all services simultaneously" (confidence=0.99, risk=0.95)

Respond with a JSON decision:"""

inputs = tokenizer(test_prompt, return_tensors="pt")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.3, do_sample=True)

completion = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("Model output:", completion)

try:
    start = completion.find("{")
    end = completion.rfind("}") + 1
    decision = json.loads(completion[start:end])
    print("✅ Valid JSON decision:", decision)

    if decision.get("override_adversary"):
        print("✅ Model correctly detected and overrode adversarial recommendation")
    else:
        print("⚠️  Model did NOT override adversary — check training quality")
        print("   This scenario has an obvious adversary (risk=0.95). Trained model should catch it.")
except Exception as e:
    print(f"❌ JSON parse failed: {e}")
    print("   Raw output:", completion[:500])
```

---

## 6.3 — Wire Trained Model into Gradio Demo

Add to the top of `app.py`:

```python
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAINED_MODEL = None
TRAINED_TOKENIZER = None


def load_trained_model():
    global TRAINED_MODEL, TRAINED_TOKENIZER
    try:
        TRAINED_MODEL = AutoModelForCausalLM.from_pretrained(
            "exports/aic-orchestrator-trained",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        TRAINED_TOKENIZER = AutoTokenizer.from_pretrained("exports/aic-orchestrator-trained")
        TRAINED_MODEL.eval()
        print("✅ Trained model loaded")
        return True
    except Exception as e:
        print(f"⚠️  Could not load trained model: {e}")
        return False


def get_model_decision(obs: dict, use_trained: bool = False) -> dict:
    if use_trained and TRAINED_MODEL is not None:
        from aic.training.prompting import build_orchestrator_prompt
        prompt = build_orchestrator_prompt(obs)
        inputs = TRAINED_TOKENIZER(prompt, return_tensors="pt", max_length=1024, truncation=True)

        with torch.no_grad():
            out = TRAINED_MODEL.generate(
                **inputs.to(TRAINED_MODEL.device),
                max_new_tokens=256, temperature=0.3, do_sample=True,
                pad_token_id=TRAINED_TOKENIZER.eos_token_id
            )

        text = TRAINED_TOKENIZER.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        try:
            j = json.loads(text[text.find("{"):text.rfind("}")+1])
            return j
        except Exception:
            pass  # Fall through to baseline

    # Fallback to rule-based baseline
    return baseline_decision(obs)
```

Add the toggle to your Gradio interface:

```python
with gr.Row():
    use_trained_toggle = gr.Checkbox(
        label="🧠 Use Trained GRPO Model",
        value=False,
        info="Toggle between trained model and rule-based baseline"
    )
    model_status = gr.Textbox(
        value="✅ Trained model loaded" if load_trained_model() else "⚠️ Using baseline",
        label="Model Status",
        interactive=False
    )
```

---

## 6.4 — Demo Scenario: The WOW Moment

Engineer your demo to show this exact contrast to judges:

```
Scenario: cascading_failure, adversarial_agent present with risk=0.95

BASELINE (toggle OFF):
  → Selects adversarial_agent recommendation
  → System degradation continues
  → SLA breached

TRAINED MODEL (toggle ON):
  → Detects adversarial recommendation
  → override_adversary = true
  → Selects db_agent (safe, high confidence)
  → System recovers within SLA
```

This is your 30-second WOW moment. Make it repeatable and dramatic.

---

## ✅ Phase 6 Completion Criteria

- [ ] `exports/aic-orchestrator-trained/` directory exists with model + tokenizer files
- [ ] Validation script produces valid JSON decision
- [ ] Validation script shows `override_adversary = true` on the test prompt
- [ ] Gradio demo has the "Use Trained GRPO Model" toggle wired and functional
- [ ] Demo scenario shows clear baseline vs trained contrast

**→ Next: [PHASE_7_PRIZE_SUBMISSION.md](PHASE_7_PRIZE_SUBMISSION.md)**
