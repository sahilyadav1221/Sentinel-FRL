# Quick Start Guide

This guide will help you get started with the Privacy-Preserving Federated RL project.

## Installation

1. **Set up Python environment** (Python 3.8+ required):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

## Running Your First Experiment

### 1. Train Federated PPO with Differential Privacy

```powershell
python experiments/train_federated.py --config config/experiment_configs/fed_ppo_dp.yaml
```

### 2. Train Baseline (Centralized RL)

```powershell
python experiments/train_federated.py --config config/experiment_configs/baseline.yaml
```

### 3. Monitor Training

Training logs and checkpoints will be saved to:
- Logs: `results/`
- Checkpoints: `checkpoints/`

## Project Structure

```
fars/
├── src/                    # Source code
│   ├── environment/        # Traffic simulation and gym environment
│   ├── agents/             # RL agents (PPO, A2C)
│   ├── federated/          # Federated learning (server, client)
│   ├── privacy/            # Differential privacy (DP-SGD)
│   ├── evaluation/         # Metrics and evaluation
│   └── utils/              # Utilities
├── experiments/            # Training scripts
├── config/                 # Configuration files
└── notebooks/              # Jupyter notebooks for analysis
```

## Key Concepts

### Environment
- **State**: Flow statistics, queue metrics, recent actions
- **Action**: 5 discrete levels (no_action, light_throttle, medium_throttle, heavy_throttle, block)
- **Reward**: -benign_loss - attack_pass - latency_penalty

### Federated Learning
- Multiple sites (edge gateways) with different traffic patterns
- Local training at each site
- Periodic aggregation at central server
- FedAvg or weighted averaging

### Differential Privacy
- DP-SGD: Gradient clipping + Gaussian noise
- Privacy budget tracking (ε, δ)
- Trade-off between privacy and utility

## Configuration

Edit `config/default_config.yaml` to customize:
- Number of sites and traffic patterns
- RL hyperparameters (learning rate, discount factor, etc.)
- Privacy parameters (noise multiplier, target ε)
- Training settings (rounds, episodes, batch size)

## Next Steps

1. **Explore notebooks**: Open `notebooks/` to see data exploration and analysis
2. **Modify config**: Adjust hyperparameters in `config/`
3. **Add custom traffic patterns**: Edit `src/environment/traffic_simulator.py`
4. **Visualize results**: Use utilities in `src/utils/visualization.py`

## Troubleshooting

### Import Errors
Make sure you're in the project root directory and have activated the virtual environment.

### CUDA Errors
If you don't have a GPU, set `device: "cpu"` in the config file.

### Out of Memory
Reduce batch size or buffer size in the agent configuration.

## Support

For questions or issues:
1. Check the README.md
2. Review configuration files
3. Open an issue on GitHub

Happy training! 🚀
