# 🎓 Complete Training Guide: Step-by-Step

## 📌 Table of Contents
1. [Understanding Data Flow](#understanding-data-flow)
2. [Dataset Options & Recommendations](#dataset-options)
3. [Training Process (Step-by-Step)](#training-process)
4. [Testing & Evaluation](#testing-evaluation)
5. [Expected Outputs](#expected-outputs)
6. [Training Time Estimates](#training-time)
7. [Fine-tuning Guide](#fine-tuning)
8. [Troubleshooting](#troubleshooting)

---

## 🔄 Understanding Data Flow

### **IMPORTANT: This Project Uses Simulated Data, NOT Real Datasets!**

Your current implementation **generates synthetic traffic data** during training. Here's why:

1. **Reinforcement Learning Requirement**: RL agents need to interact with an environment in real-time
2. **Federated Learning Simulation**: Different sites need different traffic patterns
3. **Privacy Testing**: You need controlled conditions to test differential privacy

### Current Flow:
```
[Traffic Simulator] → [Gateway Environment] → [RL Agent] → [Actions] → [Rewards]
        ↓                      ↓                    ↓
  Synthetic Traffic      State Vector         Policy Updates
  (Normal + Attacks)     (15 dimensions)      (Fed Learning)
```

---

## 📊 Dataset Options & Recommendations

### **Option 1: Continue with Synthetic Data (RECOMMENDED FOR NOW)**

**Pros:**
- ✅ Already implemented and working
- ✅ Controlled experiments
- ✅ Easy to test different scenarios
- ✅ No preprocessing needed
- ✅ Perfect for research and proof-of-concept

**Cons:**
- ❌ Not real-world data
- ❌ May not capture all attack patterns

**Current Traffic Patterns:**
```python
# In src/environment/traffic_simulator.py
- NORMAL_LIGHT: Low traffic, no attacks
- NORMAL_MODERATE: Medium traffic, no attacks  
- NORMAL_HEAVY: High traffic, no attacks
- MIXED_LIGHT_ATTACK: Low traffic + 30% attack probability
- MIXED_HEAVY_ATTACK: High traffic + 50% attack probability

# Attack Types:
- DoS (Denial of Service)
- DDoS (Distributed DoS)
- Port Scan
- SYN Flood
```

---

### **Option 2: Use Real Network Traffic Datasets (ADVANCED)**

If you want to use real data later, here are the best options:

#### **Best Datasets from Kaggle:**

1. **🏆 CICIDS2017 Dataset** (RECOMMENDED)
   - **Link**: [Kaggle CICIDS2017](https://www.kaggle.com/datasets/cicdataset/cicids2017)
   - **Size**: ~2GB
   - **Contains**: Normal traffic + 8 attack types (DoS, DDoS, Port Scan, etc.)
   - **Format**: CSV with 78 features per flow
   - **Use Case**: Perfect for anomaly detection
   
   **Attacks included:**
   - DoS/DDoS
   - Port Scan
   - Brute Force
   - Web Attacks
   - Infiltration
   - Botnet

2. **UNSW-NB15 Dataset**
   - **Link**: [Kaggle UNSW-NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
   - **Size**: ~100MB
   - **Contains**: 9 attack categories
   - **Format**: CSV with 49 features

3. **NSL-KDD Dataset**
   - **Link**: [Kaggle NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd)
   - **Size**: ~30MB
   - **Contains**: Classic intrusion detection dataset
   - **Format**: CSV with 41 features

4. **CTU-13 Dataset**
   - Botnet traffic captures
   - Good for DDoS detection

---

## 🚀 Training Process (Step-by-Step)

### **Phase 1: Setup (5-10 minutes)**

#### Step 1: Install Dependencies
```powershell
# Navigate to project directory
cd C:\Users\sahil\OneDrive\Desktop\fedrl

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

#### Step 2: Verify Installation
```powershell
# Run test script (quick validation)
python test_setup.py
```

**Expected Output:**
```
✓ Environment test passed
✓ Agent test passed
✓ Federated test passed
All tests passed! ✓
```

---

### **Phase 2: Understanding the Training Flow**

```
1. INITIALIZATION
   ├─ Create 5 edge gateway sites (different traffic patterns)
   ├─ Initialize PPO/A2C agent for each site
   ├─ Initialize federated server
   └─ Initialize privacy accountant (DP tracking)

2. TRAINING LOOP (200 rounds)
   For each communication round:
   │
   ├─ LOCAL TRAINING (at each site)
   │  ├─ Collect experiences (episodes)
   │  │  ├─ Traffic Simulator generates flows
   │  │  ├─ Anomaly Generator injects attacks (30% probability)
   │  │  ├─ Agent observes state (flow stats, queue, actions)
   │  │  ├─ Agent takes action (throttle/block level)
   │  │  ├─ Environment computes reward
   │  │  └─ Store transition in buffer
   │  │
   │  ├─ Update local model
   │  │  ├─ Compute gradients from collected experiences
   │  │  ├─ Apply DP-SGD (clip gradients + add noise)
   │  │  └─ Update actor-critic networks
   │  │
   │  └─ Send gradients to server
   │
   ├─ GLOBAL AGGREGATION (at server)
   │  ├─ Receive gradients from all sites
   │  ├─ Aggregate using FedAvg (weighted average)
   │  ├─ Update global model
   │  └─ Track privacy budget (ε accumulation)
   │
   ├─ DISTRIBUTION
   │  └─ Send updated global model back to all sites
   │
   └─ EVALUATION (every 5 rounds)
      ├─ Test on validation episodes
      ├─ Compute metrics (F1, ROC-AUC, FPR)
      ├─ Log performance
      └─ Save checkpoint

3. FINAL EVALUATION
   ├─ Test on all sites
   ├─ Generate performance metrics
   ├─ Create visualizations
   └─ Save final model
```

---

### **Phase 3: Running Training**

#### **Quick Test (10 minutes)**
```powershell
# Test with reduced rounds (just to verify everything works)
python experiments/train_federated.py --config config/experiment_configs/fed_ppo_dp.yaml
```

Edit config first for quick test:
```yaml
# In config/experiment_configs/fed_ppo_dp.yaml
federated:
  communication_rounds: 10  # Change from 200 to 10
  local_epochs: 2          # Change from 5 to 2
```

#### **Full Training Run**

**Experiment 1: Baseline (Centralized RL)**
```powershell
python experiments/train_federated.py --config config/experiment_configs/baseline.yaml
```

**Experiment 2: Federated PPO with DP**
```powershell
python experiments/train_federated.py --config config/experiment_configs/fed_ppo_dp.yaml
```

**Experiment 3: Federated A2C with DP**
```powershell
python experiments/train_federated.py --config config/experiment_configs/fed_a2c_dp.yaml
```

---

### **Phase 4: Monitoring Training**

During training, you'll see output like:
```
INFO - Round 1/200
INFO - Site 0: Episodes=5, Avg Reward=-45.2, F1=0.65
INFO - Site 1: Episodes=5, Avg Reward=-52.1, F1=0.62
INFO - Site 2: Episodes=5, Avg Reward=-38.5, F1=0.68
INFO - Site 3: Episodes=5, Avg Reward=-41.2, F1=0.66
INFO - Site 4: Episodes=5, Avg Reward=-48.9, F1=0.64
INFO - Aggregated model. Privacy budget: ε=0.15, δ=1e-5
INFO - Checkpoint saved: checkpoints/round_001.pt

...

INFO - Round 200/200
INFO - Site 0: Episodes=5, Avg Reward=-12.5, F1=0.84
INFO - Site 1: Episodes=5, Avg Reward=-15.2, F1=0.82
INFO - Site 2: Episodes=5, Avg Reward=-10.8, F1=0.86
INFO - Site 3: Episodes=5, Avg Reward=-13.1, F1=0.83
INFO - Site 4: Episodes=5, Avg Reward=-14.5, F1=0.84
INFO - Final Privacy budget: ε=9.8, δ=1e-5
INFO - Training complete!
```

**Key Metrics to Watch:**
- **Reward**: Should increase (become less negative) over time
- **F1 Score**: Should improve (closer to 1.0)
- **Privacy Budget (ε)**: Should stay under 10.0
- **Loss**: Should decrease

---

## ⏱️ Training Time Estimates

### **With Current Configuration:**

| Configuration | Rounds | Episodes/Round | Time per Round | Total Time |
|--------------|--------|----------------|----------------|------------|
| **Quick Test** | 10 | 5 | ~30 sec | ~5 min |
| **Small Experiment** | 50 | 5 | ~30 sec | ~25 min |
| **Default (CPU)** | 200 | 5 | ~30 sec | **~100 min (1.5 hrs)** |
| **Full (CPU)** | 500 | 10 | ~45 sec | **~375 min (6 hrs)** |
| **With GPU** | 200 | 5 | ~10 sec | **~35 min** |

**Factors Affecting Training Time:**
- CPU vs GPU (3-5x speedup with GPU)
- Number of sites (5 sites is manageable)
- Episode length (1000 steps default)
- Network size (256→128→64 default)
- Number of local epochs (5-10)

### **Recommendations:**

**For Testing/Development:**
```yaml
communication_rounds: 50
local_epochs: 3
episode_length: 500
```
**Time: ~15-20 minutes**

**For Paper/Production Results:**
```yaml
communication_rounds: 200-500
local_epochs: 5-10
episode_length: 1000
```
**Time: 1.5-6 hours (CPU), 30min-2hrs (GPU)**

---

## 🧪 Testing & Evaluation

### **Phase 5: After Training - Evaluation**

Training automatically saves checkpoints in `checkpoints/` and results in `results/`.

#### **1. Load and Test Trained Model**

Create `experiments/evaluate_model.py`:
```python
import torch
import yaml
from src.environment.gateway_env import GatewayAnomalyEnv
from src.agents.networks import create_network
from src.evaluation.metrics import compute_metrics

# Load config
with open('config/experiment_configs/fed_ppo_dp.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load trained model
checkpoint = torch.load('checkpoints/final_model.pt')
model = create_network(
    state_dim=15,
    action_dim=5,
    config=config['agent']['network']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test on new traffic
env = GatewayAnomalyEnv(pattern='mixed_heavy_attack')
state, _ = env.reset()

total_reward = 0
correct_blocks = 0
false_positives = 0

for step in range(1000):
    # Get action from model
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = model(state_tensor)
        action = torch.argmax(action_probs, dim=1).item()
    
    # Take action in environment
    next_state, reward, done, truncated, info = env.step(action)
    
    total_reward += reward
    if info['is_attack'] and action >= 3:  # Heavy throttle or block
        correct_blocks += 1
    elif not info['is_attack'] and action >= 3:
        false_positives += 1
    
    state = next_state
    if done or truncated:
        break

print(f"Total Reward: {total_reward:.2f}")
print(f"Correct Blocks: {correct_blocks}")
print(f"False Positives: {false_positives}")
```

Run evaluation:
```powershell
python experiments/evaluate_model.py
```

---

#### **2. Comprehensive Metrics Analysis**

The training script already computes these, but you can analyze saved results:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training logs
results = pd.read_csv('results/training_log.csv')

# Plot learning curves
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(results['round'], results['avg_reward'])
plt.xlabel('Communication Round')
plt.ylabel('Average Reward')
plt.title('Training Progress')

plt.subplot(132)
plt.plot(results['round'], results['f1_score'])
plt.xlabel('Communication Round')
plt.ylabel('F1 Score')
plt.title('Detection Performance')

plt.subplot(133)
plt.plot(results['round'], results['privacy_epsilon'])
plt.xlabel('Communication Round')
plt.ylabel('Privacy Budget (ε)')
plt.title('Privacy Consumption')

plt.tight_layout()
plt.savefig('results/training_analysis.png')
```

---

#### **3. Cross-Site Transfer Testing**

Test how well model trained on some sites performs on others:

```python
# Train on sites 0, 1, 2
train_sites = [0, 1, 2]

# Test on sites 3, 4 (unseen)
test_sites = [3, 4]

for test_site in test_sites:
    env = GatewayAnomalyEnv(pattern=f'site_{test_site}_pattern')
    # ... evaluate model
    print(f"Site {test_site} Transfer F1: {f1_score:.3f}")
```

---

## 📈 Expected Outputs

### **1. Training Outputs**

**Saved Files:**
```
results/
├── training_log.csv                  # Round-by-round metrics
├── fed_ppo_dp_20251103_143025.log   # Detailed logs
├── metrics_summary.json              # Final metrics
└── plots/
    ├── reward_curve.png
    ├── f1_curve.png
    ├── privacy_budget.png
    └── convergence.png

checkpoints/
├── round_010.pt                      # Checkpoints every 10 rounds
├── round_020.pt
├── ...
├── round_200.pt
└── final_model.pt                    # Best model
```

---

### **2. Performance Metrics**

**Expected Final Performance (after 200 rounds):**

| Metric | Centralized (No Privacy) | Fed-PPO (ε=10) | Fed-A2C (ε=10) |
|--------|--------------------------|----------------|----------------|
| **F1 Score** | 0.82 - 0.88 | 0.78 - 0.85 | 0.76 - 0.83 |
| **Precision** | 0.80 - 0.87 | 0.75 - 0.83 | 0.74 - 0.82 |
| **Recall** | 0.83 - 0.90 | 0.79 - 0.87 | 0.77 - 0.85 |
| **ROC-AUC** | 0.88 - 0.93 | 0.84 - 0.90 | 0.82 - 0.88 |
| **FPR** | 0.04 - 0.08 | 0.06 - 0.10 | 0.07 - 0.12 |
| **TPR** | 0.85 - 0.92 | 0.81 - 0.88 | 0.79 - 0.86 |
| **Avg Reward** | -8 to -12 | -12 to -18 | -14 to -20 |
| **Convergence** | ~150 rounds | ~200 rounds | ~220 rounds |

**Interpretation:**
- **F1 Score 0.80+**: Good detection capability
- **ROC-AUC 0.85+**: Strong classifier
- **FPR < 0.10**: Low false alarm rate (good!)
- **ε ≤ 10**: Acceptable privacy guarantee

---

### **3. Visualizations Generated**

**A. Training Curves**
```
Reward vs Rounds: Shows learning progress
F1 Score vs Rounds: Shows detection improvement
Loss vs Rounds: Shows model convergence
```

**B. Utility-Privacy Trade-off**
```
F1 Score vs ε (privacy budget)
Shows: Higher privacy (lower ε) = Lower performance
```

**C. Confusion Matrix**
```
              Predicted
              Normal  Attack
Actual Normal   TN      FP
       Attack   FN      TP
```

**D. ROC Curve**
```
TPR vs FPR curve
AUC = 0.85+ indicates good performance
```

**E. Site-wise Performance**
```
Bar chart showing F1 score per site
Helps identify which traffic patterns are harder
```

---

## 🎛️ Fine-Tuning Guide

### **Common Adjustments**

#### **1. Improve Detection Performance**

If F1 score is low (<0.70):

```yaml
# Increase model capacity
agent:
  network:
    hidden_layers: [512, 256, 128]  # Larger network

# More training
federated:
  communication_rounds: 300
  local_epochs: 10

# Better exploration
agent:
  ppo:
    entropy_coef: 0.02  # Increase from 0.01
```

---

#### **2. Reduce False Positives**

If FPR is too high (>0.15):

```yaml
# Adjust reward function
environment:
  reward:
    benign_loss_weight: -2.0  # Increase penalty (from -1.0)
    attack_pass_weight: -2.0
    correct_action_bonus: 0.2  # Reward correct non-action
```

---

#### **3. Stronger Privacy Guarantees**

If you need ε < 5:

```yaml
privacy:
  dp_sgd:
    noise_multiplier: 2.0  # Increase noise (from 1.1)
    target_epsilon: 5.0    # Stricter target

# May need more rounds to converge
federated:
  communication_rounds: 400
```

---

#### **4. Faster Convergence**

If training is too slow:

```yaml
agent:
  ppo:
    learning_rate: 5.0e-4  # Increase (from 3e-4)
    batch_size: 128        # Larger batches

federated:
  local_epochs: 8         # More local training
```

---

#### **5. Better Generalization**

For cross-site transfer:

```yaml
# More diverse training data
federated:
  num_sites: 8  # Increase diversity

# Regularization
agent:
  network:
    dropout: 0.1  # Add dropout
```

---

## 🐛 Checking for Anomalies/Issues

### **Common Issues & Solutions**

#### **Issue 1: Model Not Learning (Reward not improving)**

**Symptoms:**
- Reward stays constant or decreases
- F1 score stuck around 0.5-0.6

**Solutions:**
```yaml
# 1. Check reward scaling
environment:
  reward:
    benign_loss_weight: -1.0
    attack_pass_weight: -3.0  # Make attacks more important

# 2. Reduce learning rate
agent:
  ppo:
    learning_rate: 1.0e-4  # Smaller steps

# 3. Verify gradient clipping
agent:
  ppo:
    max_grad_norm: 1.0  # Increase if gradients vanish
```

---

#### **Issue 2: High Variance in Performance**

**Symptoms:**
- F1 score fluctuates wildly
- Different sites have very different performance

**Solutions:**
```yaml
# 1. Normalize observations
environment:
  normalize_observations: true

# 2. Use more stable aggregation
federated:
  aggregation_method: "adaptive"

# 3. Increase batch size
agent:
  ppo:
    batch_size: 128
```

---

#### **Issue 3: Privacy Budget Exceeded**

**Symptoms:**
- ε reaches target before training completes

**Solutions:**
```yaml
# 1. Increase target
privacy:
  dp_sgd:
    target_epsilon: 15.0

# 2. Reduce rounds
federated:
  communication_rounds: 150

# 3. Larger batches (better privacy-utility trade-off)
agent:
  ppo:
    batch_size: 128
```

---

#### **Issue 4: Out of Memory**

**Solutions:**
```yaml
# Reduce memory usage
agent:
  ppo:
    buffer_size: 1024  # Reduce from 2048
    batch_size: 32     # Reduce from 64

environment:
  episode_length: 500  # Reduce from 1000
```

---

## 🔍 Verification Checklist

After training, verify:

- [ ] **Training completed**: All 200 rounds finished
- [ ] **Model saved**: `checkpoints/final_model.pt` exists
- [ ] **Logs generated**: `results/*.log` files present
- [ ] **Metrics reasonable**: F1 > 0.70, ROC-AUC > 0.80
- [ ] **Privacy budget met**: ε ≤ target (10.0)
- [ ] **Convergence**: Reward curve plateaus
- [ ] **No NaN values**: Check logs for NaN in loss/reward
- [ ] **Cross-site test**: Model works on different traffic patterns

---

## 📝 Summary: Complete Workflow

```
STEP 1: SETUP (5-10 min)
  └─ Install dependencies
  └─ Run test_setup.py

STEP 2: QUICK TEST (5-10 min)
  └─ Run 10-round training to verify

STEP 3: BASELINE TRAINING (1-2 hrs)
  └─ Train centralized baseline

STEP 4: FEDERATED TRAINING (1.5-2 hrs each)
  └─ Train Fed-PPO with DP
  └─ Train Fed-A2C with DP

STEP 5: EVALUATION (30 min)
  └─ Compute all metrics
  └─ Generate visualizations
  └─ Test cross-site transfer

STEP 6: ANALYSIS (1 hr)
  └─ Compare baselines vs federated
  └─ Utility-privacy curves
  └─ Identify best configuration

STEP 7: FINE-TUNING (optional, 2-4 hrs)
  └─ Adjust hyperparameters
  └─ Re-train with better config

TOTAL TIME: 6-10 hours for complete experiments
```

---

## 🎯 Next Steps

**For immediate start:**
```powershell
# 1. Quick test (10 min)
python test_setup.py

# 2. Short training run (10 min)
#    Edit config first: set communication_rounds: 10
python experiments/train_federated.py --config config/experiment_configs/fed_ppo_dp.yaml

# 3. Monitor output and verify it works

# 4. If successful, run full training (2 hrs)
#    Restore config: set communication_rounds: 200
python experiments/train_federated.py --config config/experiment_configs/fed_ppo_dp.yaml
```

**Questions to answer with experiments:**
1. How does privacy (ε) affect performance?
2. Which algorithm is better: Fed-PPO or Fed-A2C?
3. How well does the model transfer across sites?
4. What's the optimal privacy-utility trade-off?

---

## 📚 Additional Resources

- **Understanding RL**: [Spinning Up in Deep RL](https://spinningup.openai.com/)
- **Differential Privacy**: [Programming Differential Privacy Book](https://programming-dp.com/)
- **Federated Learning**: [Federated Learning Comic](https://federated.withgoogle.com/)

---

**Ready to start training? Let me know if you have any questions!** 🚀
