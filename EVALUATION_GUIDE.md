# Model Evaluation Guide

Complete guide for testing and evaluating your trained federated RL models.

---

## Quick Start

After training completes (20 rounds), evaluate your model:

```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml
```

**Expected Output:**
- JSON report with detailed metrics
- Visualization plots (PNG)
- Text summary
- All saved to `results/evaluation/`

---

## What Gets Evaluated

### 1. **Performance Metrics**
- **Average Reward**: Overall performance score
- **Episode Length**: How long episodes run
- **Standard Deviation**: Consistency of performance

### 2. **Anomaly Detection Metrics**
- **F1 Score**: Harmonic mean of precision and recall (0-1, higher is better)
- **Precision**: True positive / (True positive + False positive)
- **Recall**: True positive / (True positive + False negative)
- **ROC-AUC**: Area under ROC curve (0-1, >0.8 is good)

### 3. **Action Analysis**
- **Action Distribution**: How often each action is taken
  - 0: No Action
  - 1: Light Throttle
  - 2: Medium Throttle
  - 3: Heavy Throttle
  - 4: Block

### 4. **Traffic Pattern Testing**
Evaluates on 5 different scenarios:
- `normal_light`: Light normal traffic
- `normal_moderate`: Moderate normal traffic
- `normal_heavy`: Heavy normal traffic
- `mixed_light_attack`: Light traffic with attacks
- `mixed_heavy_attack`: Heavy traffic with attacks

---

## Command Options

### Basic Evaluation
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml
```

### Extended Evaluation (more episodes for better statistics)
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --num-episodes 100
```

### Evaluate Specific Checkpoint
```powershell
# Test model from round 10
python experiments/evaluate_model.py --model checkpoints/checkpoint_round_10.pt --config config/experiment_configs/fed_ppo_dp.yaml

# Test model from round 15
python experiments/evaluate_model.py --model checkpoints/checkpoint_round_15.pt --config config/experiment_configs/fed_ppo_dp.yaml
```

### Custom Output Directory
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --output-dir results/my_evaluation
```

### Evaluate on Specific Patterns Only
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --patterns normal_moderate mixed_heavy_attack
```

### Force CPU (if GPU causes issues)
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --device cpu
```

---

## Understanding the Output

### 1. Console Output

```
======================================================================
FEDERATED RL MODEL EVALUATION
======================================================================
Model: checkpoints/final_model.pt
Config: config/experiment_configs/fed_ppo_dp.yaml
Device: cuda
Episodes per pattern: 50
======================================================================

✓ Model loaded successfully (PPO agent)

Evaluating on different traffic patterns...
----------------------------------------------------------------------

Pattern: normal_light
  Running episodes... ✓
  Reward: -45.23 ± 12.34
  F1 Score: 0.8756
  Precision: 0.8912
  Recall: 0.8604

[... more patterns ...]

======================================================================
GENERATING EVALUATION REPORT
======================================================================
✓ JSON report saved to results/evaluation/evaluation_report_20251103_160530.json
✓ Plots saved to results/evaluation/evaluation_plots_20251103_160530.png
✓ Summary saved to results/evaluation/evaluation_summary_20251103_160530.txt

======================================================================
EVALUATION COMPLETE!
======================================================================
```

### 2. JSON Report (`evaluation_report_*.json`)

Complete machine-readable results:

```json
{
  "model_path": "checkpoints/final_model.pt",
  "timestamp": "2025-11-03T16:05:30",
  "pattern_results": {
    "normal_light": {
      "avg_reward": -45.23,
      "std_reward": 12.34,
      "f1_score": 0.8756,
      "precision": 0.8912,
      "recall": 0.8604,
      "roc_auc": 0.9123,
      "action_distribution": {
        "0": 1234,
        "1": 456,
        "2": 789,
        "3": 234,
        "4": 567
      }
    }
  }
}
```

### 3. Visualizations (`evaluation_plots_*.png`)

Four-panel plot showing:
- **Top-left**: Bar chart of average rewards by pattern
- **Top-right**: F1/Precision/Recall comparison
- **Bottom-left**: Pie chart of action distribution
- **Bottom-right**: ROC-AUC scores by pattern

### 4. Text Summary (`evaluation_summary_*.txt`)

Human-readable summary:

```
======================================================================
EVALUATION REPORT - final_model
Generated: 2025-11-03 16:05:30
======================================================================

OVERALL STATISTICS
----------------------------------------------------------------------
Average Reward (all patterns): -52.34 ± 15.67
Average F1 Score: 0.8534
Average Precision: 0.8712
Average Recall: 0.8367

PER-PATTERN RESULTS
----------------------------------------------------------------------

NORMAL_LIGHT
  Reward: -45.23 ± 12.34
  F1 Score: 0.8756
  Precision: 0.8912
  Recall: 0.8604
  ROC-AUC: 0.9123
  Episode Length: 998.5

[... more patterns ...]
```

---

## Interpreting Results

### Good Performance Indicators
✅ **Rewards closer to 0** (less negative is better)
✅ **F1 Score > 0.80** (strong detection)
✅ **ROC-AUC > 0.85** (excellent discrimination)
✅ **Balanced Precision/Recall** (both > 0.75)
✅ **Appropriate action distribution** (not all no-action or all-block)

### Warning Signs
⚠️ **Very negative rewards** (< -200): Model not learning effectively
⚠️ **F1 Score < 0.60**: Poor detection capability
⚠️ **ROC-AUC < 0.70**: Weak discrimination between normal/attack
⚠️ **All actions = 0**: Model not taking mitigation actions
⚠️ **All actions = 4**: Model blocking everything (too aggressive)

---

## Comparing Models

### Compare Different Checkpoints

```powershell
# Evaluate round 5
python experiments/evaluate_model.py --model checkpoints/checkpoint_round_5.pt --config config/experiment_configs/fed_ppo_dp.yaml --output-dir results/eval_round5

# Evaluate round 10
python experiments/evaluate_model.py --model checkpoints/checkpoint_round_10.pt --config config/experiment_configs/fed_ppo_dp.yaml --output-dir results/eval_round10

# Evaluate final model
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --output-dir results/eval_final
```

Then compare the JSON reports or plots side-by-side.

### Compare Fed-PPO vs Fed-A2C

```powershell
# Evaluate Fed-PPO model
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --output-dir results/eval_ppo

# Train and evaluate Fed-A2C (if you trained it)
python experiments/evaluate_model.py --model checkpoints_a2c/final_model.pt --config config/experiment_configs/fed_a2c_dp.yaml --output-dir results/eval_a2c
```

---

## Advanced: Custom Testing

### Test on Single Episode (Debug Mode)

Create a simple test script:

```python
# test_single_episode.py
import torch
import yaml
from src.environment.gateway_env import GatewayAnomalyEnv
from src.agents.ppo_agent import PPOAgent

# Load config and model
with open('config/experiment_configs/fed_ppo_dp.yaml') as f:
    config = yaml.safe_load(f)

checkpoint = torch.load('checkpoints/final_model.pt', map_location='cpu')

agent = PPOAgent(state_dim=15, action_dim=5, config=config['agent'], device='cpu')
agent.network.load_state_dict(checkpoint['global_model'])
agent.network.eval()

# Create environment
env = GatewayAnomalyEnv(pattern='mixed_heavy_attack', episode_length=100)
state, _ = env.reset()

# Run one episode
total_reward = 0
for step in range(100):
    action, _ = agent.select_action(state, deterministic=True)
    next_state, reward, done, truncated, info = env.step(action)
    
    print(f"Step {step}: Action={action}, Reward={reward:.2f}, Attack={info.get('is_attack', False)}")
    
    total_reward += reward
    state = next_state
    
    if done or truncated:
        break

print(f"\nTotal Reward: {total_reward:.2f}")
```

Run:
```powershell
python test_single_episode.py
```

---

## Troubleshooting

### Issue: "RuntimeError: CUDA out of memory"
**Solution**: Use CPU for evaluation
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --device cpu
```

### Issue: "KeyError: 'global_model'"
**Solution**: Check if checkpoint has 'model_state_dict' instead. The script handles both, but if it fails, verify checkpoint contents:
```python
import torch
checkpoint = torch.load('checkpoints/final_model.pt', map_location='cpu')
print(checkpoint.keys())
```

### Issue: Plots not generating
**Solution**: Install matplotlib and seaborn
```powershell
pip install matplotlib seaborn
```

### Issue: Very low F1 scores
**Possible causes**:
1. Model didn't train long enough (20 rounds may be too few)
2. Hyperparameters need tuning
3. Privacy noise too high (try lower noise_multiplier)

---

## Next Steps After Evaluation

### If Performance is Good (F1 > 0.80)
✅ Deploy model for real-time inference
✅ Test on new traffic patterns
✅ Fine-tune on specific attack types
✅ Document results and create production pipeline

### If Performance is Poor (F1 < 0.60)
1. **Train longer**: Increase `communication_rounds` from 20 to 100+
2. **Adjust hyperparameters**: 
   - Lower `noise_multiplier` (but decreases privacy)
   - Increase `learning_rate`
   - Larger `batch_size`
3. **Change architecture**: Increase network size in config
4. **Try different algorithm**: Switch from PPO to A2C or vice versa

### If Mixed Results (some patterns good, others bad)
1. Increase training on weak patterns (weighted sampling)
2. Use pattern-specific fine-tuning
3. Ensemble models for different patterns

---

## Summary

**Quick evaluation command:**
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml
```

**Look for:**
- Results in `results/evaluation/`
- F1 Score > 0.75 (good)
- Rewards closer to 0 (better)
- Balanced action distribution

**Compare models:**
- Evaluate different checkpoints (round 5, 10, 15, 20, final)
- Compare algorithms (PPO vs A2C)
- Check learning curves over time

---

## Questions?

Check these files for more details:
- `TRAINING_GUIDE.md` - How to train models
- `QUICKSTART.md` - Quick setup guide
- `PROJECT_SUMMARY.md` - Overall project overview
