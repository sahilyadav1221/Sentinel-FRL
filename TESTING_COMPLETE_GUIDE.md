# 🎯 TESTING YOUR TRAINED MODEL - COMPLETE GUIDE

After successfully training your model for 20 rounds, here's how to test and evaluate it completely.

---

## ✅ What You Have Now

Your training produced these files:
- `checkpoints/checkpoint_round_5.pt` - Model after 5 rounds
- `checkpoints/checkpoint_round_10.pt` - Model after 10 rounds  
- `checkpoints/checkpoint_round_15.pt` - Model after 15 rounds
- `checkpoints/checkpoint_round_20.pt` - Model after 20 rounds
- `checkpoints/final_model.pt` - **Final trained model** ⭐
- `logs/fed_ppo_dp_epsilon10_*.log` - Training logs

---

## 🚀 Quick Start: Run Evaluation in 30 Seconds

### Method 1: Double-click the batch file (EASIEST)
```
1. Open File Explorer
2. Navigate to: C:\Users\sahil\OneDrive\Desktop\fedrl
3. Double-click: run_evaluation.bat
4. Wait for evaluation to complete (~2-3 minutes)
5. Check results/evaluation/ folder for outputs
```

### Method 2: PowerShell command
```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1

# Run evaluation
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml
```

---

## 📊 What Gets Generated (All Outputs)

After running evaluation, you'll get **3 files** in `results/evaluation/`:

### 1. **evaluation_report_TIMESTAMP.json** 📄
Complete machine-readable metrics:
- Average rewards per traffic pattern
- F1 scores, precision, recall
- ROC-AUC scores
- Action distributions
- Episode statistics

**Use for:** Further analysis, scripting, data processing

### 2. **evaluation_plots_TIMESTAMP.png** 📈
Four-panel visualization:
- **Top-left:** Bar chart showing average rewards across patterns
- **Top-right:** Detection metrics (F1, Precision, Recall) comparison
- **Bottom-left:** Pie chart of action distribution
- **Bottom-right:** ROC-AUC scores by pattern (color-coded)

**Use for:** Presentations, reports, visual analysis

### 3. **evaluation_summary_TIMESTAMP.txt** 📝
Human-readable text summary:
- Overall statistics
- Per-pattern detailed results
- Easy to read and share

**Use for:** Quick review, documentation, reporting

---

## 🎯 Understanding Your Results

### What Good Performance Looks Like ✅

#### Rewards
- **Good:** -50 to 0 (closer to 0 is better)
- **Acceptable:** -100 to -50
- **Poor:** Below -150

#### Detection Metrics
- **F1 Score:** > 0.75 is good, > 0.85 is excellent
- **Precision:** > 0.80 (few false positives)
- **Recall:** > 0.75 (catches most attacks)
- **ROC-AUC:** > 0.85 is excellent

#### Action Distribution (should be balanced)
- **Good:** Mix of all 5 actions used appropriately
- **Bad:** All action=0 (never mitigates) or all action=4 (blocks everything)

### Example Good Results
```
OVERALL STATISTICS
Average Reward (all patterns): -45.23 ± 12.34
Average F1 Score: 0.8534
Average Precision: 0.8712
Average Recall: 0.8367

NORMAL_LIGHT
  Reward: -38.45 ± 10.23
  F1 Score: 0.8756
  Precision: 0.8912
  Recall: 0.8604
  ROC-AUC: 0.9123
```

### Example Poor Results (needs more training)
```
OVERALL STATISTICS
Average Reward (all patterns): -187.56 ± 45.67
Average F1 Score: 0.4234
Average Precision: 0.5012
Average Recall: 0.3867

NORMAL_LIGHT
  Reward: -195.23 ± 56.78
  F1 Score: 0.3456
  Precision: 0.4123
  Recall: 0.2904
  ROC-AUC: 0.5789
```

---

## 🔬 Different Ways to Test

### 1. Quick Test (10 episodes per pattern) ⚡
**Time:** ~2-3 minutes
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --num-episodes 10
```

### 2. Standard Test (50 episodes - default) 📊
**Time:** ~5-10 minutes
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml
```

### 3. Comprehensive Test (100 episodes) 🎯
**Time:** ~10-15 minutes
**Best statistical reliability**
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --num-episodes 100
```

### 4. Test Specific Checkpoint (e.g., Round 10) 🔍
```powershell
python experiments/evaluate_model.py --model checkpoints/checkpoint_round_10.pt --config config/experiment_configs/fed_ppo_dp.yaml
```

### 5. Test on Specific Patterns Only 🎨
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --patterns normal_moderate mixed_heavy_attack
```

### 6. Custom Output Directory 📁
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --output-dir results/my_custom_eval
```

---

## 🔄 Comparing Model Progress Over Time

Want to see how your model improved during training?

```powershell
# Evaluate model after 5 rounds
python experiments/evaluate_model.py --model checkpoints/checkpoint_round_5.pt --config config/experiment_configs/fed_ppo_dp.yaml --output-dir results/eval_round5

# Evaluate model after 10 rounds
python experiments/evaluate_model.py --model checkpoints/checkpoint_round_10.pt --config config/experiment_configs/fed_ppo_dp.yaml --output-dir results/eval_round10

# Evaluate model after 15 rounds
python experiments/evaluate_model.py --model checkpoints/checkpoint_round_15.pt --config config/experiment_configs/fed_ppo_dp.yaml --output-dir results/eval_round15

# Evaluate final model (20 rounds)
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --output-dir results/eval_final
```

Then compare the results to see improvement trends!

---

## 📈 Expected Console Output

When you run evaluation, you'll see:

```
======================================================================
FEDERATED RL MODEL EVALUATION
======================================================================
Model: checkpoints/final_model.pt
Config: config/experiment_configs/fed_ppo_dp.yaml
Device: cuda
Episodes per pattern: 50
======================================================================

Loading model from checkpoints/final_model.pt...
✓ Model loaded successfully (PPO agent)

Evaluating on different traffic patterns...
----------------------------------------------------------------------

Pattern: normal_light
  Running episodes... ✓
  Reward: -45.23 ± 12.34
  F1 Score: 0.8756
  Precision: 0.8912
  Recall: 0.8604

Pattern: normal_moderate
  Running episodes... ✓
  Reward: -52.67 ± 15.89
  F1 Score: 0.8534
  Precision: 0.8723
  Recall: 0.8351

Pattern: normal_heavy
  Running episodes... ✓
  Reward: -61.45 ± 18.23
  F1 Score: 0.8312
  Precision: 0.8501
  Recall: 0.8131

Pattern: mixed_light_attack
  Running episodes... ✓
  Reward: -58.34 ± 14.56
  F1 Score: 0.8645
  Precision: 0.8834
  Recall: 0.8461

Pattern: mixed_heavy_attack
  Running episodes... ✓
  Reward: -72.89 ± 21.34
  F1 Score: 0.8423
  Precision: 0.8612
  Recall: 0.8241

======================================================================
GENERATING EVALUATION REPORT
======================================================================
✓ JSON report saved to results/evaluation/evaluation_report_20251103_162345.json
✓ Plots saved to results/evaluation/evaluation_plots_20251103_162345.png
✓ Summary saved to results/evaluation/evaluation_summary_20251103_162345.txt

======================================================================
EVALUATION COMPLETE!
======================================================================
Results saved to: results\evaluation
  - JSON Report: evaluation_report_20251103_162345.json
  - Visualizations: evaluation_plots_20251103_162345.png
  - Text Summary: evaluation_summary_20251103_162345.txt
======================================================================
```

---

## 🛠️ Troubleshooting

### Problem: "No module named 'torch'"
**Solution:** Make sure virtual environment is activated
```powershell
.\venv\Scripts\Activate.ps1
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml
```

### Problem: "CUDA out of memory"
**Solution:** Use CPU instead
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --device cpu
```

### Problem: Plots not generating
**Solution:** Install missing packages
```powershell
pip install matplotlib seaborn
```

### Problem: Very poor results (F1 < 0.5)
**Likely causes:**
1. **20 rounds is too few** - Try training for 100+ rounds
2. **High privacy noise** - Reduce `noise_multiplier` in config
3. **Wrong hyperparameters** - Try adjusting learning rate

---

## 🎓 What Do the Traffic Patterns Mean?

| Pattern | Description | Expected Challenge |
|---------|-------------|-------------------|
| `normal_light` | Low traffic, no attacks | Easy - should get high scores |
| `normal_moderate` | Medium traffic, no attacks | Medium - baseline performance |
| `normal_heavy` | High traffic, no attacks | Harder - must handle congestion |
| `mixed_light_attack` | Low traffic + attacks | Medium - clear attack signals |
| `mixed_heavy_attack` | High traffic + attacks | Hardest - attacks hidden in traffic |

---

## 📦 Complete Workflow Summary

### Step 1: Train Model (Already Done ✅)
```powershell
python experiments/train_federated.py --config config/experiment_configs/fed_ppo_dp.yaml
```

### Step 2: Quick Evaluation (Do This Now! ⭐)
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --num-episodes 10
```

### Step 3: Review Results
Open `results/evaluation/` and check:
- 📊 Look at the plots PNG file
- 📄 Read the summary TXT file
- 🔬 Analyze JSON if needed

### Step 4: Full Evaluation (If Quick Test Looks Good)
```powershell
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --num-episodes 50
```

### Step 5: Compare Checkpoints (Optional but Recommended)
Evaluate rounds 5, 10, 15, 20 to see learning progress

---

## 🎯 Decision Tree: What To Do Next

```
After Evaluation:
├─ F1 Score > 0.80? 
│  ├─ YES → ✅ Great! Model is ready
│  │         - Document results
│  │         - Test on real data (if available)
│  │         - Consider deployment
│  └─ NO → Training needs improvement
│           ├─ F1 between 0.60-0.80?
│           │  └─ Train longer (100 rounds)
│           └─ F1 below 0.60?
│              └─ Check hyperparameters
│                 - Reduce noise_multiplier
│                 - Increase learning_rate
│                 - Try different architecture
```

---

## 📋 Quick Reference Commands

```powershell
# Standard evaluation (recommended first run)
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml

# Quick test (faster)
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --num-episodes 10

# Compare checkpoint from round 10
python experiments/evaluate_model.py --model checkpoints/checkpoint_round_10.pt --config config/experiment_configs/fed_ppo_dp.yaml --output-dir results/eval_round10

# Force CPU usage
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --device cpu
```

---

## 📚 Related Documentation

- `EVALUATION_GUIDE.md` - Detailed evaluation documentation
- `TRAINING_GUIDE.md` - How to train models
- `QUICKSTART.md` - Quick setup guide
- `PROJECT_SUMMARY.md` - Project overview

---

## ✨ Summary

**To test your trained model RIGHT NOW:**

1. Open PowerShell in `C:\Users\sahil\OneDrive\Desktop\fedrl`
2. Run: `.\venv\Scripts\Activate.ps1`
3. Run: `python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml`
4. Wait 5-10 minutes
5. Check `results/evaluation/` for 3 output files (JSON, PNG, TXT)
6. Open the PNG to see visualizations
7. Read the TXT for quick summary

**You'll get:**
✅ Performance metrics (rewards, F1, precision, recall)
✅ Visualizations (4-panel plot)
✅ Detailed report (JSON)
✅ Text summary (easy to read)

**Good performance:**
- F1 Score > 0.75
- Rewards > -100
- Balanced action distribution

---

**Questions? Check `EVALUATION_GUIDE.md` for more details!**
