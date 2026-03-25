# PSI Risk Digital Twin

A minimalistic digital twin lab for the psi-risk-dt framework.

## Structure
- `src/` — Core Python modules for digital twin, entropy, and risk signal
- `scripts/` — Utility scripts for dataset generation and orchestration
- `data/` — Baseline and attack datasets
- `docker/` — Docker Compose and Dockerfile for environment setup
- `results/` — Output logs and figures

## Quick Start
1. Make sure Docker Engine is running
2. Build and start containers:
   ```bash
   ./scripts/run_all.sh
   ```

## Requirements
- Docker

## Main Features
- Attack scenarios dataset generators