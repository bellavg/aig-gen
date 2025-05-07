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
    from GraphEBM import GraphEBM  # For GraphEBM
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'ggraph' (including dig.ggraph) is installed, your AIGDatasetLoader class is accessible, "
          "and GraphDF/GraphAF models are correctly placed and importable.")
    exit()
# --- End Imports ---


# --- Hardcoded Configuration Dictionary ---
# This configuration will be used for the selected model.
conf = {
    "data_name": "aig",  # Used for processed data path structure
    "model": {
        # Common parameters potentially used by all models
        "max_size": 64,  # Max nodes for AIG dataset (GraphEBM: n_atom)
        "node_dim": 4,  # Node feature dimension (AIG: CONST0, PI, AND, PO) (GraphEBM: n_atom_type)
        "bond_dim": 3,  # Edge feature dimension (AIG: REG, INV, NO-EDGE) (GraphEBM: n_edge_type)
        "use_gpu": True,  # This will be updated based on device availability

        # GraphDF/GraphAF specific parameters
        "edge_unroll": 12,  # GraphAF/GraphDF
        "num_flow_layer": 12,  # GraphAF/GraphDF
        "num_rgcn_layer": 3,  # GraphAF/GraphDF
        "nhid": 128,  # GraphAF/GraphDF (hidden size for RGCN/STNet)
        "nout": 128,  # GraphAF/GraphDF (output size for RGCN)
        "deq_coeff": 0.9,  # GraphAF
        "st_type": "exp",  # GraphAF
        "use_df": False  # GraphAF
    },
    "model_ebm": {  # GraphEBM specific model parameters
        "hidden": 64  # Hidden dimension for GraphEBM's internal networks
    },
    "lr": 0.001,
    "weight_decay": 0,
    "batch_size": 128,
    "max_epochs": 50,
    "save_interval": 5,
    "save_dir": "GraphDF/rand_gen_aig_ckpts",  # This will be dynamically overwritten

    "train_ebm": {  # GraphEBM specific training parameters from your example
        "c": 0.0,  # Coefficient for energy term (0 for unconditional)
        "ld_step": 150,  # Langevin dynamics steps
        "ld_noise": 0.005,  # Langevin dynamics noise level
        "ld_step_size": 30,  # Langevin dynamics step size
        "clamp": True,  # Whether to clamp generated values
        "alpha": 1.0  # Weight for reconstruction loss (if applicable)
    },
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
        dataset = AIGDatasetLoader(
            root=args.data_root,
            name=conf.get('data_name', 'aig'),
            dataset_type="train"
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
    loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)
    print(f"Created DataLoader with batch size {conf['batch_size']}.")
    # --- End DataLoader ---

    # --- Instantiate Model ---
    print(f"Instantiating model: {args.model}")
    runner = None
    if args.model == 'GraphDF':
        runner = GraphDF()
    elif args.model == 'GraphAF':
        runner = GraphAF()
    elif args.model == 'GraphEBM':
        try:
            runner = GraphEBM(
                n_atom=conf['model']['max_size'],
                n_atom_type=conf['model']['node_dim'],
                n_edge_type=conf['model']['bond_dim'],
                hidden=conf['model_ebm']['hidden'],
                device=device  # Pass the torch device object
            )
        except KeyError as e:
            print(f"Error: Missing required parameter {e} in config for GraphEBM initialization.")
            exit()
        except Exception as e:
            print(f"Error instantiating GraphEBM: {e}")
            exit()
    else:
        print(f"Error: Unknown model type '{args.model}'. Choose from the available options.")
        exit()

    if runner is None:  # Should not happen if model is in choices, but as a safeguard
        print(f"Failed to instantiate model runner for {args.model}")
        exit()
    # --- End Model Instantiation ---

    # --- Start Training ---
    save_dir = f"{args.model}/rand_gen_{conf.get('data_name', 'default_data')}_ckpts"
    conf['save_dir'] = save_dir

    os.makedirs(save_dir, exist_ok=True)
    print(f"Model checkpoints will be saved in: {save_dir}")

    print(f"Starting training for {args.model}...")
    if args.model == 'GraphDF' or args.model == 'GraphAF':
        # GraphDF and GraphAF use model_conf_dict for internal model setup
        runner.train_rand_gen(
            loader=loader,
            lr=conf['lr'],
            wd=conf['weight_decay'],
            max_epochs=conf['max_epochs'],
            model_conf_dict=conf['model'],
            save_interval=conf['save_interval'],
            save_dir=save_dir
        )
    elif args.model == 'GraphEBM':
        try:
            train_ebm_params = conf['train_ebm']
            runner.train_rand_gen(
                dataloader=loader,  # GraphEBM's train_rand_gen expects 'dataloader'
                lr=conf['lr'],
                wd=conf['weight_decay'],
                max_epochs=conf['max_epochs'],
                c=train_ebm_params['c'],
                ld_step=train_ebm_params['ld_step'],
                ld_noise=train_ebm_params['ld_noise'],
                ld_step_size=train_ebm_params['ld_step_size'],
                clamp=train_ebm_params['clamp'],
                alpha=train_ebm_params['alpha'],
                save_interval=conf['save_interval'],
                save_dir=save_dir
            )
        except KeyError as e:
            print(f"Error: Missing required parameter {e} in 'train_ebm' config for GraphEBM training.")
            exit()
        except AttributeError:
            print(f"Error: {args.model} runner does not have a compatible train_rand_gen method or "
                  "there's an issue with its parameters.")
            exit()
        except Exception as e:
            print(f"An unexpected error occurred during GraphEBM training: {e}")
            exit()

    print("Training finished.")
    # --- End Training ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train graph generation models on pre-processed AIG dataset.")
    parser.add_argument('--model', type=str, default='GraphDF',
                        choices=['GraphDF', 'GraphAF', 'GraphEBM'],
                        help='Model to train (GraphDF, GraphAF, or GraphEBM)')
    parser.add_argument('--data_root', default="./data/",
                        help="Root directory containing the pre-processed AIG structure (e.g., 'data/aig/processed/train/data.pt').")
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device to use for training.')

    args = parser.parse_args()
    main(args)
