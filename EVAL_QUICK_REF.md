# 🚀 QUICK EVALUATION CHEAT SHEET

## Run Evaluation NOW (Copy-Paste)

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Run evaluation
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml
```

---

## Outputs (Check these folders)

📁 `results/evaluation/`
- `evaluation_report_*.json` - Full metrics data
- `evaluation_plots_*.png` - **LOOK AT THIS FIRST** 📊
- `evaluation_summary_*.txt` - Quick text summary

---

## What Good Results Look Like ✅

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| **Reward** | > -50 | -50 to -100 | < -150 |
| **F1 Score** | > 0.80 | 0.65 - 0.80 | < 0.65 |
| **ROC-AUC** | > 0.85 | 0.70 - 0.85 | < 0.70 |
| **Precision** | > 0.80 | 0.65 - 0.80 | < 0.65 |
| **Recall** | > 0.75 | 0.60 - 0.75 | < 0.60 |

---

## Common Commands

```powershell
# Quick test (2 min)
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --num-episodes 10

# Standard test (5-10 min)
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml

# Compare round 10 vs final
python experiments/evaluate_model.py --model checkpoints/checkpoint_round_10.pt --config config/experiment_configs/fed_ppo_dp.yaml --output-dir results/eval_round10

# Force CPU (if GPU issues)
python experiments/evaluate_model.py --model checkpoints/final_model.pt --config config/experiment_configs/fed_ppo_dp.yaml --device cpu
```

---

## Batch File (Easiest Method)

**Just double-click:** `run_evaluation.bat`

That's it! 🎯

---

## Files Created

✅ `experiments/evaluate_model.py` - Main evaluation script
✅ `EVALUATION_GUIDE.md` - Detailed guide
✅ `TESTING_COMPLETE_GUIDE.md` - Complete testing walkthrough
✅ `run_evaluation.bat` - One-click evaluation

---

## Need Help?

Read: `TESTING_COMPLETE_GUIDE.md` for full details
