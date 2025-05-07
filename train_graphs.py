import os
import json  # Still needed if you have EBM params in a dict format
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
    from GraphDF import GraphDF  # Assuming GraphDF.py or a GraphDF package exists
    from GraphAF import GraphAF  # From the GraphAF directory/package
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'ggraph' is installed, your AIGDatasetLoader class is accessible, "
          "and GraphDF/GraphAF models are correctly placed and importable.")
    exit()
# --- End Imports ---


# --- Hardcoded Configuration Dictionary ---
# This configuration will be used for the selected model.
# Parameters specific to one model (e.g., deq_coeff for GraphAF)
# will be ignored by other models if they don't use them.
conf = {
    "data_name": "aig",  # Used for processed data path structure
    "model": {
        # Common parameters
        "max_size": 64,  # Max nodes for AIG dataset
        "edge_unroll": 12,
        "node_dim": 4,  # MUST match AIGDataset processing (e.g., CONST0, PI, AND, PO for AIG)
        "bond_dim": 3,  # MUST match AIGDataset processing (e.g., REG, INV, NO-EDGE for AIG)
        # For GraphAF, this usually includes a virtual/no-edge type.
        # If AIG data provides 3 explicit edge types (including a "no-edge" category), this is fine.
        # GraphAF's RGCN uses edge_dim = bond_dim - 1 for actual bond weights.
        "num_flow_layer": 12,
        "num_rgcn_layer": 3,
        "nhid": 128,
        "nout": 128,
        "use_gpu": True,  # This will be updated based on device availability

        # GraphAF specific parameters (from your example GraphAF config)
        # These will be in model_conf_dict passed to GraphAF
        "deq_coeff": 0.9,
        "st_type": "exp",
        "use_df": False  # Assuming this is 'use_discrete_flow' or similar for GraphAF's GraphFlowModel
    },
    # EBM specific model parameters (add if training EBM)
    # "model_ebm": {
    #     "hidden": 128
    # },
    "lr": 0.001,
    "weight_decay": 0,
    "batch_size": 128,  # User's current setting
    "max_epochs": 50,
    "save_interval": 5,
    "save_dir": "GraphDF/rand_gen_aig_ckpts",  # This will be dynamically overwritten
    # EBM specific training parameters (add if training EBM)
    # "train_ebm": {
    #     "c": 0.01,
    #     "ld_step": 60,
    #     "ld_noise": 0.005,
    #     "ld_step_size": 10,
    #     "clamp": True,
    #     "alpha": 0.1
    # },
    # Generation parameters (not used directly during training by train_rand_gen)
    # "num_min_node_gen": 5,
    # "num_max_node_gen": 64,
    # "temperature_gen": [0.3, 0.3], # Or a single float like 0.75 for GraphAF
    # "atom_list_gen": [0, 1, 2, 3] # For AIG, node types
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
            name=conf.get('data_name', 'aig'),  # Get name from conf or default
            dataset_type="train"  # Load the training split
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
    # which _process_graph in AIGDatasetLoader should have added.
    loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)
    print(f"Created DataLoader with batch size {conf['batch_size']}.")
    # --- End DataLoader ---

    # --- Instantiate Model ---
    print(f"Instantiating model: {args.model}")
    runner = None
    if args.model == 'GraphDF':
        runner = GraphDF()
        # get_model is called within train_rand_gen for GraphDF if that's its internal structure
    elif args.model == 'GraphAF':
        runner = GraphAF()
        # GraphAF().get_model is called within its train_rand_gen method
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
        print(f"Error: Unknown model type '{args.model}'. Choose from the available options.")
        exit()

    if runner is None:
        print(f"Failed to instantiate model runner for {args.model}")
        exit()
    # --- End Model Instantiation ---

    # --- Start Training ---
    # Dynamically set save_dir based on the model name
    save_dir = f"{args.model}/rand_gen_{conf.get('data_name', 'default_data')}_ckpts"
    conf['save_dir'] = save_dir  # Update conf in case any part of the runner uses it directly

    os.makedirs(save_dir, exist_ok=True)
    print(f"Model checkpoints will be saved in: {save_dir}")

    print(f"Starting training for {args.model}...")
    if args.model == 'GraphDF' or args.model == 'GraphAF':
        # Both GraphDF and GraphAF (from your provided structure) should have this method
        runner.train_rand_gen(
            loader=loader,
            lr=conf['lr'],
            wd=conf['weight_decay'],
            max_epochs=conf['max_epochs'],
            model_conf_dict=conf['model'],  # Pass the 'model' sub-dictionary
            save_interval=conf['save_interval'],
            save_dir=save_dir
        )
    # elif args.model == 'GraphEBM': # Keep EBM logic commented if not being used
    #     try:
    #         # Check if EBM specific training config exists
    #         if 'train_ebm' not in conf:
    #             raise KeyError("Missing 'train_ebm' block in configuration for GraphEBM training.")
    #
    #         train_params = conf['train_ebm']
    #         runner.train_rand_gen( # Assuming GraphEBM also has a compatible train_rand_gen
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
    #             save_dir=save_dir
    #         )
    #     except KeyError as e:
    #         print(f"Error: Missing required parameter {e} in config for GraphEBM training.")
    #         exit()
    #     except AttributeError:
    #         print(f"Error: {args.model} runner does not have a compatible train_rand_gen method for EBM.")
    #         exit()

    print("Training finished.")
    # --- End Training ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train graph generation models on pre-processed AIG dataset.")
    parser.add_argument('--model', type=str, default='GraphDF',
                        choices=['GraphDF', 'GraphAF', 'GraphEBM'],  # Added GraphAF
                        help='Model to train (GraphDF, GraphAF, or GraphEBM)')
    # Removed --config_file and --aig_config_path arguments
    parser.add_argument('--data_root', default="./data/",
                        help="Root directory containing the pre-processed AIG structure (e.g., 'data/aig/processed/train/data.pt').")
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device to use for training.')

    args = parser.parse_args()
    main(args)
