# Onboarding Guide for Colab Agent

## Project Overview
Colab Agent is an autonomous AI agent designed for fine-tuning HuggingFace models. It handles planning, code generation, execution in Google Colab/local, and self-improvement loops.

## Core Architecture
- **Agent Orchestrator (`agent/core.py`)**: Central control loop.
- **Colab/Local Executor (`colab/executor.py`)**: Environment-aware code execution.
- **Storage Layer (`storage/`)**: Asynchronous, tenant-isolated data stores.
- **Model Orchestration (`models/`)**: Manages model/dataset loading and PEFT configurations.

## Key Concepts
- **Simulation Mode**: Run locally for testing without live GPUs.
- **Runtime Switching**: Auto-upgrades GPU tiers on OOM detection.
- **Plugin System**: Modular hooks for observability, error recovery, and data ingestion.
- **Audit Logging**: HMAC-signed, immutable security trails.

## Getting Started
1. Set up `.env` with HF/GitHub tokens and API keys.
2. Run `python cli.py init` to setup the database.
3. Use `python cli.py run "goal..."` for the agent.

## Complexity Hotspots
- `agent/core.py`: The orchestrator loop handles high-concurrency state management.
- `colab/executor.py`: Bridge between local controller and remote execution state.
