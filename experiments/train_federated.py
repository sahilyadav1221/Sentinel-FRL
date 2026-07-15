"""
Main Training Script for Federated RL
Trains Fed-PPO or Fed-A2C with differential privacy
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
import logging
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List

# Project imports
from src.environment.gateway_env import GatewayAnomalyEnv
from src.federated.server import FederatedServer
from src.federated.client import FederatedClient
from src.agents.networks import create_network
from src.privacy.privacy_accountant import PrivacyAccountant
from src.evaluation.metrics import compute_metrics, MetricsAggregator


def setup_logging(log_dir: Path, experiment_name: str):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("FederatedTraining")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_clients(config: Dict, device: str, network_config: Dict = None) -> List[FederatedClient]:
    """
    Create federated clients for each site
    
    Args:
        config: Full configuration dictionary
        device: Device to run on
        network_config: Network architecture config (if None, uses config['agent']['network'])
    """
    clients = []
    
    num_sites = config['federated']['num_sites']
    traffic_patterns = config['environment'].get('traffic_patterns', [])
    
    # Use provided network_config or fall back to config
    if network_config is None:
        network_config = config['agent'].get('network', {})
    
    for site_id in range(num_sites):
        # Get traffic pattern for this site
        if site_id < len(traffic_patterns):
            pattern_config = traffic_patterns[site_id]
            pattern = list(pattern_config.values())[0]
        else:
            pattern = "normal_moderate"
        
        # Create environment
        env = GatewayAnomalyEnv(
            pattern=pattern,
            episode_length=config['environment']['episode_length'],
            seed=config['experiment']['seed'] + site_id,
        )
        
        # Prepare agent config with the correct network architecture
        agent_config = config['agent'].get(config['agent']['algorithm'], {}).copy()
        agent_config['network'] = network_config  # Override network config
        
        # Create client
        client = FederatedClient(
            client_id=f"site_{site_id}",
            env=env,
            agent_type=config['agent']['algorithm'],
            agent_config=agent_config,
            dp_config=config.get('privacy', None),
            device=device,
        )
        
        clients.append(client)
        logging.info(f"Created client {client.client_id} with pattern {pattern}")
    
    return clients


def create_clients_with_params(config: Dict, device: str, initial_params: Dict, network_config: Dict) -> List[FederatedClient]:
    """Create federated clients and initialize with global model parameters"""
    clients = create_clients(config, device, network_config)
    
    # Set all clients to use the same parameters as global model
    for client in clients:
        client.set_parameters(initial_params)
    
    return clients


def train_federated(config: Dict, logger: logging.Logger):
    """Main federated training loop"""
    
    # Setup
    device = config['compute']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Set seeds
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create a temporary environment to get state/action dimensions
    temp_env = GatewayAnomalyEnv(
        pattern="normal_moderate",
        episode_length=config['environment']['episode_length'],
        seed=seed,
    )
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    
    # Get network config from main config
    network_config = config['agent'].get('network', {})
    
    # Create global model first (with config architecture)
    global_model = create_network(
        "actor_critic",
        state_dim,
        action_dim,
        network_config
    ).to(device)
    
    # Get the initial parameters from global model
    initial_params = global_model.state_dict().copy()
    
    # Now create clients with the same architecture
    clients = create_clients_with_params(config, device, initial_params, network_config)
    
    # Setup privacy accountant
    privacy_accountant = None
    if config.get('privacy', {}).get('enabled', False):
        privacy_config = config.get('privacy', {})
        accounting_config = privacy_config.get('accounting', {})
        
        privacy_accountant = PrivacyAccountant(
            method=accounting_config.get('method', 'rdp'),  # Default to 'rdp'
            target_epsilon=privacy_config['dp_sgd']['target_epsilon'],
            target_delta=privacy_config['dp_sgd']['target_delta'],
        )
        logger.info("Privacy tracking enabled")
    
    # Create federated server
    server = FederatedServer(
        global_model=global_model,
        aggregation_method=config['federated']['aggregation_method'],
        privacy_accountant=privacy_accountant,
        config=config['federated'],
    )
    
    # Training loop
    num_rounds = config['federated']['communication_rounds']
    local_epochs = config['federated']['local_epochs']
    sampling_rate = config['federated'].get('site_sampling_rate', 1.0)  # Default to 1.0 (all sites)
    
    logger.info(f"Starting federated training for {num_rounds} rounds")
    logger.info(f"{'='*50}")
    
    for round_num in range(num_rounds):
        logger.info(f"Round {round_num + 1}/{num_rounds}")
        logger.info(f"{'='*50}")
        
        # Select clients for this round
        all_client_ids = [c.client_id for c in clients]
        selected_ids = server.select_clients(
            all_client_ids,
            sampling_rate=sampling_rate,
            min_clients=config['federated'].get('min_clients', 1),  # Default to 1
        )
        
        selected_clients = [c for c in clients if c.client_id in selected_ids]
        
        # Distribute global model to selected clients
        global_params = server.get_global_parameters()
        for client in selected_clients:
            client.set_parameters(global_params)
        
        # Local training at each client
        client_parameters = {}
        client_stats = {}
        
        for client in selected_clients:
            try:
                logger.info(f"Training client {client.client_id}...")
                
                stats = client.train_local_epochs(
                    num_epochs=local_epochs,
                    episodes_per_epoch=5,
                )
                
                client_parameters[client.client_id] = client.get_parameters()
                client_stats[client.client_id] = stats
                
                logger.info(
                    f"  Avg Reward: {stats.get('avg_reward', 0):.2f}, "
                    f"Avg Length: {stats.get('avg_episode_length', 0):.1f}"
                )
                
                # Track privacy if enabled
                if privacy_accountant is not None:
                    privacy_spent = client.get_privacy_spent()
                    if privacy_spent:
                        epsilon_val = float(privacy_spent.get('epsilon', 0)) if privacy_spent.get('epsilon') else 0
                        delta_val = float(privacy_spent.get('delta', 0)) if privacy_spent.get('delta') else 0
                        logger.info(
                            f"  Privacy: epsilon={epsilon_val:.2f}, "
                            f"delta={delta_val:.2e}"
                        )
            except Exception as e:
                logger.error(f"Error training {client.client_id}: {e}", exc_info=True)
                # Continue with other clients
                continue
        
        # Aggregate and update global model
        if client_parameters:  # Only aggregate if we have client parameters
            server.update_global_model(client_parameters)
        else:
            logger.warning("No client parameters to aggregate!")
        
        # Evaluation
        if (round_num + 1) % config['evaluation']['frequency'] == 0:
            logger.info("\nEvaluating...")
            eval_results = evaluate_clients(clients, config['evaluation']['num_episodes'])
            
            # Compute average metrics
            avg_reward = np.mean([r['avg_reward'] for r in eval_results.values()])
            logger.info(f"Average evaluation reward: {avg_reward:.2f}")
            
            # Log classification metrics if available
            f1_scores = [r.get('f1_score') for r in eval_results.values() if r.get('f1_score') is not None]
            if f1_scores:
                avg_f1 = np.mean(f1_scores)
                logger.info(f"Average F1 Score: {avg_f1:.4f}")
            
            precision_scores = [r.get('precision') for r in eval_results.values() if r.get('precision') is not None]
            if precision_scores:
                avg_precision = np.mean(precision_scores)
                logger.info(f"Average Precision: {avg_precision:.4f}")
            
            recall_scores = [r.get('recall') for r in eval_results.values() if r.get('recall') is not None]
            if recall_scores:
                avg_recall = np.mean(recall_scores)
                logger.info(f"Average Recall: {avg_recall:.4f}")
            
            fpr_scores = [r.get('fpr') for r in eval_results.values() if r.get('fpr') is not None]
            if fpr_scores:
                avg_fpr = np.mean(fpr_scores)
                logger.info(f"Average FPR: {avg_fpr:.4f}")
            
            # Log privacy status
            if privacy_accountant:
                privacy_report = privacy_accountant.get_privacy_report()
                logger.info(
                    f"Global privacy: epsilon={privacy_report['global']['epsilon']:.2f}"
                )
        
        # Save checkpoint
        if (round_num + 1) % config['experiment']['save_frequency'] == 0:
            checkpoint_dir = Path(config['experiment']['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"checkpoint_round_{round_num+1}.pt"
            server.save_checkpoint(str(checkpoint_path))
    
    logger.info("\nTraining completed!")
    
    # Final evaluation
    logger.info("\nFinal evaluation...")
    final_results = evaluate_clients(clients, num_episodes=50)
    
    for client_id, results in final_results.items():
        metrics_str = f"{client_id}: Reward={results['avg_reward']:.2f} ± {results['std_reward']:.2f}"
        
        if 'f1_score' in results and results['f1_score'] is not None:
            metrics_str += f", F1={results['f1_score']:.4f}"
        if 'precision' in results and results['precision'] is not None:
            metrics_str += f", Precision={results['precision']:.4f}"
        if 'recall' in results and results['recall'] is not None:
            metrics_str += f", Recall={results['recall']:.4f}"
        if 'fpr' in results and results['fpr'] is not None:
            metrics_str += f", FPR={results['fpr']:.4f}"
        
        logger.info(metrics_str)
    
    # Save final model
    final_path = Path(config['experiment']['checkpoint_dir']) / "final_model.pt"
    server.save_checkpoint(str(final_path))
    logger.info(f"Final model saved to {final_path}")
    
    return server, clients


def evaluate_clients(clients: List[FederatedClient], num_episodes: int) -> Dict:
    """Evaluate all clients"""
    results = {}
    
    for client in clients:
        eval_result = client.evaluate(num_episodes=num_episodes)
        results[client.client_id] = eval_result
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Federated RL Training for Anomaly Response"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override experiment name if provided
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    
    # Setup logging
    log_dir = Path(config['experiment']['log_dir'])
    logger = setup_logging(log_dir, config['experiment']['name'])
    
    logger.info("="*60)
    logger.info("Privacy-Preserving Federated RL for Anomaly Response")
    logger.info("="*60)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Configuration: {args.config}")
    
    # Run training
    try:
        server, clients = train_federated(config, logger)
        logger.info("\nExperiment completed successfully!")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
