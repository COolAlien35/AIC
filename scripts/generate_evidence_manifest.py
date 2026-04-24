#!/usr/bin/env python3
"""
Generate hackathon evidence manifest.
Run AFTER all training and benchmarking is complete.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check(path: Path, condition: bool = True, min_count: int = 0) -> tuple:
    exists = path.exists()
    if not exists:
        return "❌", {"path": str(path), "status": "MISSING"}

    info = {"path": str(path), "size_kb": round(path.stat().st_size / 1024, 1)}

    if path.suffix == ".jsonl":
        count = sum(1 for _ in open(path))
        info["record_count"] = count
        ok = count >= min_count
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        info["keys"] = list(data.keys())[:5]
        ok = condition
    elif path.suffix == ".csv":
        try:
            import pandas as pd
            df = pd.read_csv(path)
            info["rows"] = len(df)
            info["columns"] = list(df.columns)
            ok = condition
        except Exception:
            ok = False
    elif path.suffix in (".png", ".jpg"):
        ok = True
    else:
        ok = True

    return ("✅" if ok else "⚠️"), info


def generate_manifest():
    print("📋 Generating Evidence Manifest...\n")

    evidence = {}

    # 1. SFT Training Data
    status, info = check(Path("artifacts/sft/orchestrator_sft.jsonl"), min_count=400)
    evidence["sft_training_data"] = {"status": status, **info}

    # 2. SFT Checkpoint
    sft_meta = Path("checkpoints/sft/sft_metadata.json")
    if sft_meta.exists():
        meta = json.loads(sft_meta.read_text())
        evidence["sft_checkpoint"] = {
            "status": "✅" if "Qwen" in meta.get("model_name", "") or "Llama" in meta.get("model_name", "") else "⚠️",
            "model_name": meta.get("model_name"),
            "path": "checkpoints/sft/",
        }
    else:
        evidence["sft_checkpoint"] = {"status": "❌", "path": "checkpoints/sft/"}

    # 3. GRPO Checkpoint
    grpo_path = Path("checkpoints/grpo")
    if grpo_path.exists() and any(grpo_path.iterdir()):
        training_summary = grpo_path / "training_summary.json"
        if training_summary.exists():
            summary = json.loads(training_summary.read_text())
            evidence["grpo_checkpoint"] = {
                "status": "✅",
                "path": str(grpo_path),
                "total_steps": summary.get("total_steps"),
                "reward_delta": summary.get("reward_delta"),
                "training_time_minutes": summary.get("training_time_minutes"),
            }
        else:
            evidence["grpo_checkpoint"] = {"status": "⚠️", "path": str(grpo_path), "note": "missing training_summary.json"}
    else:
        evidence["grpo_checkpoint"] = {"status": "❌", "path": "checkpoints/grpo/"}

    # 4. Benchmark Results
    bench_path = Path("results/benchmark_summary.csv")
    if bench_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(bench_path)
            evidence["benchmark"] = {
                "status": "✅",
                "path": str(bench_path),
                "policies": list(df["policy"].unique()) if "policy" in df.columns else [],
                "rows": len(df),
            }
        except Exception:
            evidence["benchmark"] = {"status": "⚠️", "path": str(bench_path)}
    else:
        evidence["benchmark"] = {"status": "❌", "path": str(bench_path)}

    # 5. Statistical Test
    stats_path = Path("results/statistical_test.json")
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        evidence["statistical_test"] = {
            "status": "✅" if stats.get("significant") else "⚠️",
            "p_value": stats.get("p_value"),
            "cohens_d": stats.get("cohens_d"),
            "effect_size": stats.get("effect_size_label"),
            "significant": stats.get("significant"),
        }
    else:
        evidence["statistical_test"] = {"status": "❌"}

    # 6. Plots
    for plot in ["reward_curve.png", "policy_comparison.png", "verifier_pass_rate.png"]:
        p = Path("results") / plot
        evidence[f"plot_{plot.replace('.png','')}"] = {
            "status": "✅" if p.exists() else "❌",
            "path": str(p),
        }

    # 7. GRPO Training Logs
    log_path = Path("logs/grpo_progress.jsonl")
    if log_path.exists():
        entries = [json.loads(l) for l in open(log_path) if l.strip()]
        evidence["grpo_training_logs"] = {
            "status": "✅",
            "total_log_entries": len(entries),
            "first_reward": entries[0]["reward"] if entries else None,
            "last_reward": entries[-1]["reward"] if entries else None,
        }
    else:
        evidence["grpo_training_logs"] = {"status": "❌"}

    # 8. Reward Audit Logs
    audit_dir = Path("logs/audit")
    if audit_dir.exists():
        audit_files = list(audit_dir.glob("*.jsonl"))
        evidence["reward_audit_logs"] = {
            "status": "✅" if audit_files else "⚠️",
            "num_audit_files": len(audit_files),
            "path": str(audit_dir),
        }
    else:
        evidence["reward_audit_logs"] = {"status": "❌"}

    # Save manifests
    Path("results").mkdir(exist_ok=True)

    full_manifest = {
        "project": "Adaptive Incident Choreographer (AIC)",
        "generated_at": datetime.now().isoformat(),
        "evidence": evidence,
    }

    with open("results/evidence_manifest.json", "w") as f:
        json.dump(full_manifest, f, indent=2)

    # Generate markdown version
    complete = sum(1 for v in evidence.values() if v.get("status") == "✅")
    total = len(evidence)

    md_lines = [
        "# 📋 Evidence Manifest — AIC Hackathon Submission",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Complete**: {complete}/{total} items verified\n",
        "| Item | Status | Key Detail |",
        "|------|--------|-----------|",
    ]

    for key, val in evidence.items():
        status = val.get("status", "❌")
        detail = val.get("path", "")
        md_lines.append(f"| {key.replace('_', ' ').title()} | {status} | {detail} |")

    with open("results/evidence_manifest.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n✅ Evidence Manifest Generated")
    print(f"   Complete: {complete}/{total}")
    for key, val in evidence.items():
        print(f"   {val.get('status', '❌')} {key}")

    if complete < total:
        missing = [k for k, v in evidence.items() if v.get("status") != "✅"]
        print(f"\n⚠️  Still missing: {missing}")
        print("   Complete training and benchmark before final submission.")


if __name__ == "__main__":
    generate_manifest()
