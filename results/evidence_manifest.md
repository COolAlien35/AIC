# Evidence Manifest

- Generated at (UTC): `2026-04-23T18:16:04.553151+00:00`
- Invoked tasks: `verify`
- Unsloth mode: `missing_fallback_to_transformers`
- Torch backends: `cuda=False` `mps=True`

## Artifacts

| Path | Exists | Size (bytes) | SHA256 |
|------|--------|--------------|--------|
| `results/reward_curve.png` | `True` | `98005` | `23ae0629a0aa359f95baff8d2c28e65dfac2e3673c325a050c5af68c8c619b0e` |
| `results/verifier_pass_rate.png` | `True` | `47108` | `9b7dc865d8114210622e792d6d7e702bbe434723f2c5bf01b5c12975150c270c` |
| `results/before_after_demo.md` | `True` | `2352` | `9901c0c4e1f9bf638d1ab00335d96092c7426138521a28bff8e6636f64706965` |
| `logs/eval/policy_benchmark.jsonl` | `True` | `1202` | `efce88a25ecf34120a324ae5ddd680edd5d7d5589b01c08cec3a05755dcf0d3c` |
| `results/benchmark_summary.csv` | `True` | `249` | `a8eda593adfde70d8aee9ee079f2517af0717ffea54d489c2e8ee421fcae4427` |
| `checkpoints/sft/sft_metadata.json` | `True` | `94` | `049e5152c398e1212a766a41bdf70189ace03958659e5f904b997be0dc5c7a99` |
| `checkpoints/grpo/grpo_metadata.json` | `True` | `76` | `d61c67a0c6053ee931b4a7065f54bdb744a1f60f5ae4438ffb2782a1fa32d76c` |

## Repro Commands

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run_hackathon.py verify plots demo
./.venv/bin/python run_hackathon.py sft
```

## Optional GPU Path

```bash
./.venv/bin/python run_hackathon.py grpo
```
