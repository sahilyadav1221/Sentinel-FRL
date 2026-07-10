@echo off
REM Quick evaluation test script
REM Activates venv and runs evaluation

echo ============================================================
echo Federated RL Model Evaluation
echo ============================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Virtual environment activated
echo Running evaluation on final_model.pt...
echo.

REM Run evaluation (10 episodes for quick test)
python experiments\evaluate_model.py --model checkpoints\final_model.pt --config config\experiment_configs\fed_ppo_dp.yaml --num-episodes 10

echo.
echo ============================================================
echo Evaluation complete! Check results\evaluation\ folder
echo ============================================================
pause
