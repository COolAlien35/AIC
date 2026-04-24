# Design Document: RL Training Pipeline Improvements

## Overview

The RL Training Pipeline Improvements feature transforms the AIC (Adaptive Incident Choreographer) training infrastructure from a prototype into a production-ready, provable reinforcement learning system. This comprehensive redesign addresses critical gaps across 7 phases: emergency code fixes, SFT data generation, supervised fine-tuning, GRPO reinforcement learning, statistical benchmarking, evidence generation, and model deployment. The system trains a Qwen 0.5B orchestrator agent to coordinate 6 specialist agents under adversarial conditions, with formal verification that training produces measurable improvement over baselines.

The design implements a complete training-to-deployment pipeline with: (1) diverse SFT dataset generation across 6 fault scenarios with adversarial override examples, (2) curriculum-based GRPO training with reward hacking protection, (3) statistical benchmarking with t-tests and effect size analysis, (4) automated evidence manifest generation, and (5) model export with Gradio demo integration. The architecture ensures reproducibility, statistical rigor, and hackathon-grade presentation quality.

## Architecture

```mermaid
graph TB
    subgraph "Phase 0: Emergency Fixes"
        A1[Model Config Fix] --> A2[SFT Data Generator Rewrite]
        A2 --> A3[Benchmark Script Enhancement]
        A3 --> A4[GRPO Logging Callback]
        A4 --> A5[Verification Suite]
    end
    
    subgraph "Phase 1-3: Training Pipeline"
        B1[SFT Data Generation<br/>600+ examples, 6 scenarios] --> B2[Supervised Fine-Tuning<br/>Qwen 0.5B + LoRA]
        B2 --> B3[GRPO Training<br/>150 steps, reward optimization]
        B3 --> B4[Checkpoint Management<br/>Auto-save every 25 steps]
    end
    
    subgraph "Phase 4: Statistical Proof"
        C1[Benchmark Runner<br/>30 episodes × 3 policies] --> C2[Statistical Tests<br/>t-test + Cohen's d]
        C2 --> C3[Per-Scenario Analysis<br/>Success rate breakdown]
    end
    
    subgraph "Phase 5-6: Evidence & Export"
        D1[Reward Curve Plots] --> D2[Evidence Manifest Generator]
        D2 --> D3[Model Export<br/>LoRA merge + validation]
        D3 --> D4[Gradio Demo Integration]
    end
    
    A5 --> B1
    B4 --> C1
    C3 --> D1
    D4 --> E[Submission Package]
    
    style A1 fill:#ff6b6b
    style B3 fill:#4ecdc4
    style C2 fill:#ffe66d
    style D3 fill:#95e1d3
