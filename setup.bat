@echo off
REM Setup script for Privacy-Preserving Federated RL Project

echo ================================================
echo Privacy-Preserving Federated RL Setup
echo ================================================
echo.

REM Check Python version
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
python -m venv venv

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ================================================
echo Setup Complete!
echo ================================================
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate
echo.
echo To test your setup, run:
echo   python test_setup.py
echo.
echo To start training, run:
echo   python experiments/train_federated.py --config config/experiment_configs/fed_ppo_dp.yaml
echo.
pause
