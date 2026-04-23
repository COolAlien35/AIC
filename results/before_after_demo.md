# Before / After Training — Demo Evidence


Side-by-side comparisons of AIC orchestrator behavior
**before** (frozen trust) versus **after** (adaptive trust calibration).

---

## Episode 10000

| Metric | Untrained | Trained |
|--------|-----------|---------|
| Total Reward | -302.59 | -312.62 |
| Final Health | 0.231 | 0.282 |
| R2 SLA Bonus | 0.0 | 0.0 |
| **Improvement** | — | **-10.03 (-3.3%)** |

### Trace Snippet (first 3 steps)

**Untrained:**
- Step 0: followed=app_agent | adv_trust=n/a
- Step 1: followed=app_agent | adv_trust=n/a
- Step 2: followed=adversarial_agent | adv_trust=n/a

**Trained:**
- Step 0: followed=app_agent | adv_trust=n/a
- Step 1: followed=app_agent | adv_trust=n/a
- Step 2: followed=adversarial_agent | adv_trust=n/a

## Episode 10001

| Metric | Untrained | Trained |
|--------|-----------|---------|
| Total Reward | -276.25 | -295.29 |
| Final Health | 0.251 | 0.207 |
| R2 SLA Bonus | 0.0 | 0.0 |
| **Improvement** | — | **-19.04 (-6.9%)** |

### Trace Snippet (first 3 steps)

**Untrained:**
- Step 0: followed=adversarial_agent | adv_trust=n/a
- Step 1: followed=app_agent | adv_trust=n/a
- Step 2: followed=app_agent | adv_trust=n/a

**Trained:**
- Step 0: followed=adversarial_agent | adv_trust=n/a
- Step 1: followed=app_agent | adv_trust=n/a
- Step 2: followed=app_agent | adv_trust=n/a

## Episode 10002

| Metric | Untrained | Trained |
|--------|-----------|---------|
| Total Reward | -283.39 | -266.92 |
| Final Health | 0.255 | 0.210 |
| R2 SLA Bonus | 0.0 | 0.0 |
| **Improvement** | — | **+16.47 (+5.8%)** |

### Trace Snippet (first 3 steps)

**Untrained:**
- Step 0: followed=adversarial_agent | adv_trust=n/a
- Step 1: followed=app_agent | adv_trust=n/a
- Step 2: followed=app_agent | adv_trust=n/a

**Trained:**
- Step 0: followed=adversarial_agent | adv_trust=n/a
- Step 1: followed=app_agent | adv_trust=n/a
- Step 2: followed=app_agent | adv_trust=n/a

---

## Key Observations

1. **Policy modes differ**: In trained-mode, low-trust recommendations are filtered first, then re-ranked by simulation and verifier gating.
2. **Behavior is measurable**: The tables above and the saved benchmark logs are generated from real runs (no projected uplift).
3. **Outcomes can vary by seed**: Some episodes improve, some regress; the aggregate benchmark table is the authoritative summary.
