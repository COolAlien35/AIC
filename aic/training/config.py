# aic/training/config.py
"""
Training configuration for the AIC orchestrator agent.
All hyperparameters in one place.
"""
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """All training hyperparameters."""

    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"  # Production model for hackathon
    use_peft: bool = True          # LoRA for memory efficiency
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    load_in_4bit: bool = True      # Required for T4 VRAM budget

    # PPO hyperparameters
    learning_rate: float = 1e-4
    ppo_epochs: int = 4
    mini_batch_size: int = 4
    batch_size: int = 16
    gradient_accumulation_steps: int = 4

    # Training loop
    num_episodes: int = 100
    checkpoint_interval: int = 25
    output_dir: str = "checkpoints"
    log_dir: str = "logs"
    trajectories_dir: str = "dashboard/assets"
    artifacts_dir: str = "artifacts"

    # Environment
    base_seed: int = 42
    fault_mode: str = "cascading_failure"
    use_llm_agents: bool = False  # Use rule-based during training for speed

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.3
    do_sample: bool = True

    # Prompting / sequence lengths
    max_prompt_length: int = 1024
    max_completion_length: int = 512

    # SFT warm start
    sft_dataset_path: str = "artifacts/sft/orchestrator_sft.jsonl"
    sft_num_episodes: int = 120       # 20 per scenario × 6 scenarios = 120
    sft_epochs: int = 1
    sft_batch_size: int = 2
    sft_learning_rate: float = 2e-5
    sft_output_dir: str = "checkpoints/sft"
    sft_grad_accumulation: int = 4
    use_peft_for_sft: bool = True

    # GRPO / RLVR
    grpo_dataset_path: str = "artifacts/grpo/prompts.jsonl"
    grpo_output_dir: str = "checkpoints/grpo"
    grpo_num_generations: int = 4
    grpo_max_steps: int = 150         # Enough for measurable learning
    grpo_per_device_train_batch_size: int = 1
    grpo_gradient_accumulation_steps: int = 8   # Effective batch = 8

    # Efficiency / export
    use_unsloth: bool = True
    export_dir: str = "exports"

    # Serving
    env_server_host: str = "0.0.0.0"
    env_server_port: int = 8000

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_advancement_threshold: float = -200.0
    curriculum_rolling_window: int = 10
    curriculum_min_episodes_per_tier: int = 5
    curriculum_log_path: str = "logs/curriculum.jsonl"

    # Reward audit
    use_reward_audit: bool = True     # MUST be True — it's a prize feature
    audit_log_dir: str = "logs/audit"
    audit_max_wall_clock: float = 120.0
    audit_max_steps: int = 50
    audit_severity_clamp_threshold: float = 0.5

    # Process-aware feedback
    enable_process_feedback: bool = True
