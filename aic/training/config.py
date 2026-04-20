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
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"  # Small, trainable on CPU/single GPU
    use_peft: bool = True          # LoRA for memory efficiency
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05

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

    # Environment
    base_seed: int = 42
    fault_mode: str = "cascading_failure"
    use_llm_agents: bool = False  # Use rule-based during training for speed

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.3
    do_sample: bool = True
