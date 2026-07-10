# Privacy-Preserving Federated RL for Anomaly Response

## Overview
This project implements a **Privacy-Preserving Federated Reinforcement Learning** system for network anomaly detection and response at edge gateways. The system learns rate-limiting and blocking policies using federated learning with differential privacy guarantees.

## Problem Statement
Edge gateways across different sites need to learn optimal rate-limit/block policies to defend against network anomalies while:
- Preserving privacy through Differential Privacy (DP)
- Learning collaboratively via Federated Learning
- Minimizing benign traffic loss
- Blocking attack traffic
- Maintaining low latency

## Architecture

### Environment
- **State (S)**: Flow statistics, queue metrics, recent action history
- **Action (A)**: Throttle/block levels (discrete or continuous)
- **Reward (R)**: −benign_loss − attack_pass − latency_penalty

### Algorithms
- **Federated RL**: Fed-PPO, Fed-A2C
- **Privacy**: DP-SGD with gradient clipping and noise injection
- **Baselines**: Local heuristics, centralized RL

### Metrics
- F1-Score, ROC-AUC
- FPR @ low-FAR
- Privacy budget (ε) vs. performance
- Rounds to convergence
- Site transfer performance

## Project Structure
```
fars/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── default_config.yaml
│   ├── experiment_configs/
│   │   ├── fed_ppo_dp.yaml
│   │   ├── fed_a2c_dp.yaml
│   │   └── baseline.yaml
├── src/
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── traffic_simulator.py
│   │   ├── anomaly_generator.py
│   │   └── gateway_env.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── ppo_agent.py
│   │   ├── a2c_agent.py
│   │   ├── baseline_agents.py
│   │   └── networks.py
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── client.py
│   │   └── aggregation.py
│   ├── privacy/
│   │   ├── __init__.py
│   │   ├── dp_sgd.py
│   │   └── privacy_accountant.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── visualization.py
├── experiments/
│   ├── train_federated.py
│   ├── train_centralized.py
│   ├── evaluate.py
│   └── analyze_results.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_environment_testing.ipynb
│   ├── 03_training_analysis.ipynb
│   └── 04_visualization.ipynb
├── data/
│   └── .gitkeep
├── results/
│   └── .gitkeep
└── tests/
    ├── __init__.py
    ├── test_environment.py
    ├── test_agents.py
    └── test_federated.py
```

## Installation

```bash
# Clone the repository
cd fars

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train Federated RL with DP
```bash
python experiments/train_federated.py --config config/experiment_configs/fed_ppo_dp.yaml
```

### 2. Train Baseline (Centralized)
```bash
python experiments/train_centralized.py --config config/experiment_configs/baseline.yaml
```

### 3. Evaluate and Compare
```bash
python experiments/evaluate.py --results_dir results/
```

### 4. Generate Analysis
```bash
python experiments/analyze_results.py --results_dir results/ --output_dir analysis/
```

## Key Features

### 1. Federated Learning
- Multiple edge gateway sites with heterogeneous traffic
- Privacy-preserving gradient aggregation
- Asynchronous updates support
- Site-specific adaptation

### 2. Differential Privacy
- DP-SGD with Gaussian noise
- Gradient clipping
- Privacy budget tracking (ε, δ)
- Composition analysis

### 3. Reinforcement Learning
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- Custom reward shaping
- Experience replay for sample efficiency

### 4. Comprehensive Evaluation
- F1-Score, Precision, Recall
- ROC-AUC analysis
- FPR at different thresholds
- Utility-privacy trade-off curves
- Convergence analysis
- Cross-site transfer learning

## Deliverables

1. **Utility-Privacy Curves**: Trade-off between model performance and privacy budget (ε)
2. **Site Transfer Analysis**: Performance when deploying models across different sites
3. **System Flowchart**: Visual representation of the federated learning pipeline
4. **Convergence Analysis**: Rounds to convergence vs. baseline
5. **Performance Metrics**: Comprehensive evaluation across all metrics

## Configuration

Edit `config/default_config.yaml` to customize:
- Number of edge sites
- Traffic patterns per site
- RL hyperparameters (learning rate, discount factor, etc.)
- DP parameters (noise multiplier, clipping norm)
- Training settings (rounds, episodes, batch size)

## Research Questions

1. How does differential privacy impact convergence and performance?
2. What is the optimal privacy-utility trade-off for this task?
3. How well do models transfer across sites with different traffic patterns?
4. How does federated learning compare to centralized and local approaches?

## Citation

```bibtex
@misc{federated_rl_anomaly_2025,
  title={Privacy-Preserving Federated Reinforcement Learning for Network Anomaly Response},
  author={Your Name},
  year={2025}
}
```

## License
MIT License

## Contact
For questions or issues, please open an issue on GitHub.
