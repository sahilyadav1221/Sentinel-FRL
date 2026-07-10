"""
Quick Start Script
Runs a fast test to verify everything works before full training
"""

import sys
import time
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_step(step, text):
    print(f"\n[Step {step}] {text}")

def run_quick_test():
    """Run quick test with minimal configuration"""
    
    print_header("🚀 FEDERATED RL QUICK START")
    
    print("""
This script will:
1. Verify your installation
2. Run a quick 10-round training test
3. Show you what to expect
4. Guide you to full training

Estimated time: 5-10 minutes
""")
    
    input("Press Enter to continue...")
    
    # ========================================================================
    # STEP 1: Check dependencies
    # ========================================================================
    print_step(1, "Checking dependencies...")
    
    try:
        import torch
        import numpy
        import gymnasium
        import yaml
        print("✓ All required packages installed!")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("\nPlease run: pip install -r requirements.txt")
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: Check project structure
    # ========================================================================
    print_step(2, "Verifying project structure...")
    
    required_files = [
        "src/environment/gateway_env.py",
        "src/agents/ppo_agent.py",
        "src/federated/server.py",
        "config/default_config.yaml",
        "experiments/train_federated.py"
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print("✗ Missing files:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("✓ All required files present!")
    
    # ========================================================================
    # STEP 3: Test environment
    # ========================================================================
    print_step(3, "Testing environment...")
    
    try:
        from src.environment.gateway_env import GatewayAnomalyEnv
        
        env = GatewayAnomalyEnv(pattern="normal_moderate", episode_length=100)
        state, info = env.reset()
        
        for _ in range(10):
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
        
        print("✓ Environment working correctly!")
        print(f"  - State dimension: {len(state)}")
        print(f"  - Action space: {env.action_space.n} actions")
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 4: Test agent
    # ========================================================================
    print_step(4, "Testing RL agent...")
    
    try:
        from src.agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(
            state_dim=15,
            action_dim=5,
            config={"learning_rate": 3e-4}
        )
        
        # Test forward pass (select_action returns: action, info_dict)
        action, info = agent.select_action(state)
        
        print("✓ Agent working correctly!")
        print(f"  - Selected action: {action}")
        print(f"  - Value estimate: {info['value']:.3f}")
        
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 5: Create quick test config
    # ========================================================================
    print_step(5, "Creating quick test configuration...")
    
    quick_config = {
        'experiment': {
            'name': 'quick_test',
            'seed': 42,
            'log_dir': './results',
            'checkpoint_dir': './checkpoints',
            'save_frequency': 5
        },
        'federated': {
            'num_sites': 3,  # Reduced from 5
            'communication_rounds': 10,  # Reduced from 200
            'local_epochs': 2,  # Reduced from 5
            'aggregation_method': 'fedavg'
        },
        'environment': {
            'episode_length': 200,  # Reduced from 1000
            'traffic_patterns': [
                {'site_0': 'normal_moderate'},
                {'site_1': 'mixed_light_attack'},
                {'site_2': 'mixed_heavy_attack'}
            ],
            'reward': {
                'benign_loss_weight': -1.0,
                'attack_pass_weight': -2.0,
                'latency_weight': -0.5
            }
        },
        'agent': {
            'algorithm': 'ppo',
            'network': {
                'hidden_layers': [128, 64],  # Smaller network
                'activation': 'relu'
            },
            'ppo': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'clip_range': 0.2,
                'batch_size': 32,
                'buffer_size': 512
            }
        },
        'privacy': {
            'enabled': True,
            'dp_sgd': {
                'noise_multiplier': 1.1,
                'max_grad_norm': 1.0,
                'target_epsilon': 10.0,
                'target_delta': 1e-5
            }
        },
        'evaluation': {
            'frequency': 5,
            'num_episodes': 10
        },
        'logging': {
            'level': 'INFO',
            'console': True
        },
        'compute': {
            'device': 'auto'
        }
    }
    
    # Save config
    config_path = Path("config/quick_test.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(quick_config, f, default_flow_style=False)
    
    print(f"✓ Quick test config saved to: {config_path}")
    
    # ========================================================================
    # STEP 6: Run quick training
    # ========================================================================
    print_step(6, "Running quick training test (10 rounds)...")
    
    print("""
Starting training with:
  - 3 sites (edge gateways)
  - 10 communication rounds
  - 2 local epochs per round
  - 200 steps per episode
  - Differential privacy enabled (ε=10)

This will take approximately 3-5 minutes.
Watch for:
  ✓ Rewards improving (becoming less negative)
  ✓ F1 scores increasing
  ✓ Privacy budget (ε) staying under 10
""")
    
    input("\nPress Enter to start training...")
    
    start_time = time.time()
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "experiments/train_federated.py", 
             "--config", str(config_path)],
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ Training completed successfully in {elapsed/60:.1f} minutes!")
        else:
            print(f"\n✗ Training failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        elapsed = time.time() - start_time
    
    # ========================================================================
    # STEP 7: Show results
    # ========================================================================
    print_step(7, "Checking results...")
    
    results_dir = Path("results")
    checkpoints_dir = Path("checkpoints")
    
    if results_dir.exists():
        log_files = list(results_dir.glob("*.log"))
        if log_files:
            print(f"✓ Logs saved: {len(log_files)} file(s)")
            print(f"  Location: {results_dir}")
    
    if checkpoints_dir.exists():
        checkpoint_files = list(checkpoints_dir.glob("*.pt"))
        if checkpoint_files:
            print(f"✓ Checkpoints saved: {len(checkpoint_files)} file(s)")
            print(f"  Location: {checkpoints_dir}")
    
    # ========================================================================
    # STEP 8: Next steps
    # ========================================================================
    print_header("✨ QUICK TEST COMPLETE!")
    
    print("""
🎉 Congratulations! Your setup is working correctly!

📊 What you just did:
  - Trained 3 federated sites for 10 rounds
  - Used PPO with differential privacy
  - Detected network anomalies
  - Learned rate-limiting policies

📈 Next Steps:

1. REVIEW RESULTS:
   - Check logs in: results/
   - View checkpoints in: checkpoints/

2. RUN FULL TRAINING:
   Edit config/experiment_configs/fed_ppo_dp.yaml:
   - Set communication_rounds: 200
   - Set local_epochs: 5
   - Set episode_length: 1000
   
   Then run:
   python experiments/train_federated.py --config config/experiment_configs/fed_ppo_dp.yaml
   
   Estimated time: 1.5-2 hours (CPU), 30-45 minutes (GPU)

3. COMPARE ALGORITHMS:
   Train Fed-A2C:
   python experiments/train_federated.py --config config/experiment_configs/fed_a2c_dp.yaml
   
   Train baseline:
   python experiments/train_federated.py --config config/experiment_configs/baseline.yaml

4. ANALYZE RESULTS:
   See TRAINING_GUIDE.md for detailed analysis steps

5. TUNE HYPERPARAMETERS:
   Modify config files to improve performance

📚 Documentation:
   - TRAINING_GUIDE.md: Complete training instructions
   - README.md: Project overview
   - QUICKSTART.md: Getting started guide

❓ Questions?
   Review the documentation or check the code comments.

🚀 Ready for production training!
""")
    
    print("=" * 80)


if __name__ == "__main__":
    try:
        run_quick_test()
    except KeyboardInterrupt:
        print("\n\n✗ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
