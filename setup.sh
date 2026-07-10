#!/bin/bash
# Setup script for Privacy-Preserving Federated RL Project (Linux/Mac)

echo "================================================"
echo "Privacy-Preserving Federated RL Setup"
echo "================================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found! Please install Python 3.8 or higher."
    exit 1
fi

python3 --version

echo ""
echo "Creating virtual environment..."
python3 -m venv venv

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test your setup, run:"
echo "  python test_setup.py"
echo ""
echo "To start training, run:"
echo "  python experiments/train_federated.py --config config/experiment_configs/fed_ppo_dp.yaml"
echo ""
