import os
import json # Still needed if you have EBM params in a dict format
import argparse
# import importlib.util # No longer needed for aig_config.py
import torch
from torch_geometric.loader import DenseDataLoader
import warnings
import os.path as osp

# --- Assume ggraph and your AIG dataset loader are importable ---
try:
    # Import the loader class (make sure the filename is correct)
    from data.aig_dataset import AIGDatasetLoader
    # Import the models
    from GraphDF import GraphDF
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'ggraph' is installed and your AIGDatasetLoader class "
          "(e.g., in ggraph/dataset/aig_dataset_loader.py) is accessible.")
    exit()
# --- End Imports ---


# --- Hardcoded Configuration Dictionary ---
# Based on the previous aig_graphdf_config_json
conf = {
    "data_name": "aig", # Used for processed data path structure
    # "data" block is no longer needed for dataset instantiation
    # but might be kept for reference or if other parts use it.
    # "data": {
    #     "num_max_node_in_csv": "64", # Not directly used by loader
    #     "atom_list_in_csv": "[0, 1, 2, 3]" # Not directly used by loader
    # },
    "model": { # Model configuration IS still needed
        "max_size": 64,
        "edge_unroll": 12,
        "node_dim": 4,  # MUST match AIGDataset processing (CONST0, PI, AND, PO)
        "bond_dim": 3,  # MUST match AIGDataset processing (REG, INV, NO-EDGE)
        "num_flow_layer": 12,
        "num_rgcn_layer": 3,
        "nhid": 128,
        "nout": 128,
        "use_gpu": True # This will be updated based on device availability
    },
    # EBM specific model parameters (add if training EBM)
    # "model_ebm": {
    #     "hidden": 128
    # },
    "lr": 0.001,
    "weight_decay": 0,
    "batch_size": 32,
    "max_epochs": 50, # Increased example value
    "save_interval": 5, # Increased example value
    "save_dir": "GraphDF/rand_gen_aig_ckpts", # Default save directory
    # EBM specific training parameters (add if training EBM)
    # "train_ebm": {
    #     "c": 0.01,
    #     "ld_step": 60,
    #     "ld_noise": 0.005,
    #     "ld_step_size": 10,
    #     "clamp": True,
    #     "alpha": 0.1
    # },
    # Generation parameters (not used during training)
    # "num_min_node_gen": 5,
    # "num_max_node_gen": 64,
    # "temperature_gen": [0.3, 0.3],
    # "atom_list_gen": [0, 1, 2, 3]
}
# --- End Hardcoded Configuration ---


def main(args):
    # --- Setup Device ---
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU.")
    # Update config dict based on actual device availability
    conf['model']['use_gpu'] = (device.type == 'cuda')
    # --- End Device Setup ---


    # --- Instantiate Dataset using AIGDatasetLoader ---
    print("Instantiating AIGDatasetLoader...")
    try:
        # Root should point to the directory containing the 'aig/processed/train' structure
        dataset = AIGDatasetLoader(
            root=args.data_root,
            name=conf.get('data_name', 'aig'), # Get name from conf or default
            dataset_type="train" # Load the training split
        )
        print(f"Number of training graphs loaded: {len(dataset)}")
        if len(dataset) == 0:
            print(f"Error: Training dataset is empty. "
                  f"Please check if '{osp.join(args.data_root, conf.get('data_name', 'aig'), 'processed', 'train', 'data.pt')}' "
                  "exists and contains data.")
            exit()
    except FileNotFoundError as e:
         print(f"Error: {e}")
         print("Ensure --data_root points to the directory containing the 'aig/processed/train' structure "
               "created by the processing script.")
         exit()
    except Exception as e:
        print(f"Error instantiating AIGDatasetLoader: {e}")
        exit()
    # --- End Dataset ---

    # --- Create DataLoader ---
    # DenseDataLoader requires 'num_atom' attribute in the Data object,
    # which _process_graph should have added.
    loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)
    print(f"Created DataLoader with batch size {conf['batch_size']}.")
    # --- End DataLoader ---

    # --- Instantiate Model ---
    print(f"Instantiating model: {args.model}")
    if args.model == 'GraphDF':
        runner = GraphDF()
        # get_model is called within train_rand_gen for GraphDF
    # elif args.model == 'GraphEBM':
        # GraphEBM needs parameters during __init__
        # try:
        #     # Check if EBM specific config exists
        #     if 'model_ebm' not in conf:
        #          raise KeyError("Missing 'model_ebm' block in configuration for GraphEBM model.")
        #
        #     model_params = conf['model_ebm']
        #     runner = GraphEBM(
        #         n_atom=conf['model']['max_size'],
        #         n_atom_type=conf['model']['node_dim'],
        #         n_edge_type=conf['model']['bond_dim'],
        #         hidden=model_params['hidden'],
        #         device=device
        #     )
        # except KeyError as e:
        #     print(f"Error: Missing required parameter {e} in config for GraphEBM initialization.")
        #     exit()
    else:
        print(f"Error: Unknown model type '{args.model}'. Choose 'GraphDF' or 'GraphEBM'.")
        exit()
    # --- End Model Instantiation ---

    # --- Start Training ---
    # Ensure the save directory exists
    save_dir = conf.get('save_dir', 'aig_training_ckpts') # Use default if not in conf
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model checkpoints will be saved in: {save_dir}")

    print(f"Starting training for {args.model}...")
    if args.model == 'GraphDF':
        runner.train_rand_gen(
            loader=loader,
            lr=conf['lr'],
            wd=conf['weight_decay'],
            max_epochs=conf['max_epochs'],
            model_conf_dict=conf['model'], # Pass the 'model' sub-dictionary
            save_interval=conf['save_interval'],
            save_dir=save_dir # Use variable
        )
    # elif args.model == 'GraphEBM':
    #     try:
    #         # Check if EBM specific training config exists
    #         if 'train_ebm' not in conf:
    #             raise KeyError("Missing 'train_ebm' block in configuration for GraphEBM training.")
    #
    #         train_params = conf['train_ebm']
    #         runner.train_rand_gen(
    #             loader=loader,
    #             lr=conf['lr'],
    #             wd=conf['weight_decay'],
    #             max_epochs=conf['max_epochs'],
    #             c=train_params['c'],
    #             ld_step=train_params['ld_step'],
    #             ld_noise=train_params['ld_noise'],
    #             ld_step_size=train_params['ld_step_size'],
    #             clamp=train_params['clamp'],
    #             alpha=train_params['alpha'],
    #             save_interval=conf['save_interval'],
    #             save_dir=save_dir # Use variable
    #         )
    #     except KeyError as e:
    #         print(f"Error: Missing required parameter {e} in config for GraphEBM training.")
    #         exit()
    #
    print("Training finished.")
    # --- End Training ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GraphDF or GraphEBM on pre-processed AIG dataset.")
    parser.add_argument('--model', type=str, default='GraphDF', choices=['GraphDF', 'GraphEBM'],
                        help='Model to train (GraphDF or GraphEBM)')
    # Removed --config_file and --aig_config_path arguments
    parser.add_argument('--data_root', default="./data/",
                        help="Root directory containing the pre-processed AIG structure (e.g., 'aig/processed/train/data.pt').")
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device to use for training.')

    args = parser.parse_args()
    main(args)
