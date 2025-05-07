import os
import json
import argparse
import torch
from torch_geometric.loader import DenseDataLoader
import warnings
import os.path as osp

# --- Assume ggraph and your AIG dataset loader are importable ---
try:
    # Import the loader class (make sure the filename is correct)
    from data.aig_dataset import AIGDatasetLoader # Assuming this is in data/aig_dataset.py
    # Import the models
    from GraphDF import GraphDF
    from GraphAF import GraphAF
    from GraphEBM import GraphEBM
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'ggraph' (including dig.ggraph) is installed, your AIGDatasetLoader class is accessible, "
          "and GraphDF/GraphAF/GraphEBM models are correctly placed and importable.")
    exit()
# --- End Imports ---


# --- Base Configuration Dictionary ---
# Some parts of this will be overridden by command-line arguments
base_conf = {
    "data_name": "aig",
    "model": {
        "max_size": 64,
        "node_dim": 4,
        "bond_dim": 3,
        "use_gpu": True, # This will be updated based on device availability and args

        # GraphDF/GraphAF specific parameters (defaults, can be overridden by args)
        "edge_unroll": 12,
        "num_flow_layer": 12,
        "num_rgcn_layer": 3,
        "nhid": 128,
        "nout": 128,
        "deq_coeff": 0.9,
        "st_type": "exp",
        "use_df": False
    },
    "model_ebm": {
        "hidden": 64 # Default, can be overridden by args
    },
    # General training parameters (defaults, can be overridden by args)
    "lr": 0.001,
    "weight_decay": 0,
    "batch_size": 128,
    "max_epochs": 50,
    "save_interval": 5,
    "save_dir": "GraphDF/rand_gen_aig_ckpts", # This will be dynamically overwritten

    "train_ebm": { # GraphEBM specific training parameters (defaults, can be overridden by args)
        "c": 0.0,
        "ld_step": 150,
        "ld_noise": 0.005,
        "ld_step_size": 30,
        "clamp": True,
        "alpha": 1.0
    }
}
# --- End Base Configuration ---


def main(args):
    # --- Create a working copy of the configuration ---
    conf = base_conf.copy() # Start with base defaults
    conf['model'] = base_conf['model'].copy()
    conf['model_ebm'] = base_conf['model_ebm'].copy()
    conf['train_ebm'] = base_conf['train_ebm'].copy()


    # --- Update config with command-line arguments ---
    # General training params
    conf['lr'] = args.lr
    conf['weight_decay'] = args.weight_decay
    conf['batch_size'] = args.batch_size
    conf['max_epochs'] = args.max_epochs
    conf['save_interval'] = args.save_interval

    # GraphAF/GraphDF specific model params
    conf['model']['edge_unroll'] = args.edge_unroll
    conf['model']['num_flow_layer'] = args.num_flow_layer
    conf['model']['num_rgcn_layer'] = args.num_rgcn_layer
    conf['model']['nhid'] = args.gaf_nhid
    conf['model']['nout'] = args.gaf_nout
    conf['model']['deq_coeff'] = args.deq_coeff
    conf['model']['st_type'] = args.st_type

    # GraphEBM specific model params
    conf['model_ebm']['hidden'] = args.ebm_hidden

    # GraphEBM specific training params
    conf['train_ebm']['c'] = args.ebm_c
    conf['train_ebm']['ld_step'] = args.ebm_ld_step
    conf['train_ebm']['ld_noise'] = args.ebm_ld_noise
    conf['train_ebm']['ld_step_size'] = args.ebm_ld_step_size
    conf['train_ebm']['alpha'] = args.ebm_alpha
    # --- End Config Update ---

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
    conf['model']['use_gpu'] = (device.type == 'cuda')
    # --- End Device Setup ---

    # --- Instantiate Dataset using AIGDatasetLoader ---
    print("Instantiating AIGDatasetLoader...")
    try:
        dataset = AIGDatasetLoader(
            root=args.data_root,
            name=conf.get('data_name', 'aig'), # data_name is still from base_conf
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
    # Uses batch_size from args (via updated conf)
    loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)
    print(f"Created DataLoader with batch size {conf['batch_size']}.")
    # --- End DataLoader ---

    # --- Instantiate Model ---
    print(f"Instantiating model: {args.model_type}") # Changed from args.model to args.model_type
    runner = None
    if args.model_type == 'GraphDF':
        runner = GraphDF()
    elif args.model_type == 'GraphAF':
        runner = GraphAF()
    elif args.model_type == 'GraphEBM':
        try:
            runner = GraphEBM(
                n_atom=conf['model']['max_size'],       # max_size is still from base_conf
                n_atom_type=conf['model']['node_dim'],  # node_dim is still from base_conf
                n_edge_type=conf['model']['bond_dim'],  # bond_dim is still from base_conf
                hidden=conf['model_ebm']['hidden'],     # hidden is from args (via updated conf)
                device=device
            )
        except KeyError as e:
            print(f"Error: Missing required parameter {e} in config for GraphEBM initialization.")
            exit()
        except Exception as e:
            print(f"Error instantiating GraphEBM: {e}")
            exit()
    else:
        print(f"Error: Unknown model type '{args.model_type}'. Choose from GraphDF, GraphAF, GraphEBM.")
        exit()

    if runner is None:
        print(f"Failed to instantiate model runner for {args.model_type}")
        exit()
    # --- End Model Instantiation ---

    # --- Start Training ---
    # save_dir is dynamically set based on model_type and data_name
    save_dir = f"{args.model_type}/rand_gen_{conf.get('data_name', 'default_data')}_ckpts"
    conf['save_dir'] = save_dir # Update conf with the dynamic save_dir

    os.makedirs(save_dir, exist_ok=True)
    print(f"Model checkpoints will be saved in: {save_dir}")

    print(f"Starting training for {args.model_type}...")
    if args.model_type == 'GraphDF' or args.model_type == 'GraphAF':
        runner.train_rand_gen(
            loader=loader,
            lr=conf['lr'],
            wd=conf['weight_decay'],
            max_epochs=conf['max_epochs'],
            model_conf_dict=conf['model'], # Pass the 'model' sub-dictionary
            save_interval=conf['save_interval'],
            save_dir=conf['save_dir']
        )
    elif args.model_type == 'GraphEBM':
        try:
            # train_ebm_params now directly comes from the updated conf
            runner.train_rand_gen(
                loader=loader,
                lr=conf['lr'],
                wd=conf['weight_decay'],
                max_epochs=conf['max_epochs'],
                c=conf['train_ebm']['c'],
                ld_step=conf['train_ebm']['ld_step'],
                ld_noise=conf['train_ebm']['ld_noise'],
                ld_step_size=conf['train_ebm']['ld_step_size'],
                clamp=conf['train_ebm']['clamp'], # clamp is still from base_conf['train_ebm']
                alpha=conf['train_ebm']['alpha'],
                save_interval=conf['save_interval'],
                save_dir=conf['save_dir']
            )
        except KeyError as e:
            print(f"Error: Missing required parameter {e} in 'train_ebm' config for GraphEBM training.")
            exit()
        except AttributeError:
            print(f"Error: {args.model_type} runner does not have a compatible train_rand_gen method or "
                  "there's an issue with its parameters.")
            exit()
        except Exception as e:
            print(f"An unexpected error occurred during GraphEBM training: {e}")
            exit()

    print("Training finished.")
    # --- End Training ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train graph generation models on pre-processed AIG dataset.")

    # --- Essential Arguments ---
    parser.add_argument('--model_type', type=str, default=base_conf.get('model_type', 'GraphAF'), # Added a default model_type
                        choices=['GraphDF', 'GraphAF', 'GraphEBM'],
                        help='Model to train (GraphDF, GraphAF, or GraphEBM)')
    parser.add_argument('--data_root', default="./data/",
                        help="Root directory containing the pre-processed AIG structure (e.g., 'data/aig/processed/train/data.pt').")
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device to use for training.')

    # --- General Training Hyperparameters ---
    parser.add_argument('--lr', type=float, default=base_conf['lr'],
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=base_conf['weight_decay'],
                        help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=base_conf['batch_size'],
                        help='Batch size for training.')
    parser.add_argument('--max_epochs', type=int, default=base_conf['max_epochs'],
                        help='Maximum number of training epochs.')
    parser.add_argument('--save_interval', type=int, default=base_conf['save_interval'],
                        help='Epoch interval for saving model checkpoints.')

    # --- GraphAF/GraphDF Specific Model Hyperparameters ---
    parser.add_argument('--edge_unroll', type=int, default=base_conf['model']['edge_unroll'],
                        help='Edge unroll factor for GraphAF/GraphDF.')
    parser.add_argument('--num_flow_layer', type=int, default=base_conf['model']['num_flow_layer'],
                        help='Number of flow layers in GraphAF/GraphDF.')
    parser.add_argument('--num_rgcn_layer', type=int, default=base_conf['model']['num_rgcn_layer'],
                        help='Number of RGCN layers in GraphAF/GraphDF.')
    parser.add_argument('--gaf_nhid', type=int, default=base_conf['model']['nhid'],
                        help='Hidden size for RGCN/STNet in GraphAF/GraphDF.')
    parser.add_argument('--gaf_nout', type=int, default=base_conf['model']['nout'],
                        help='Output size for RGCN in GraphAF/GraphDF.')
    parser.add_argument('--deq_coeff', type=float, default=base_conf['model']['deq_coeff'],
                        help='Dequantization coefficient for GraphAF.')
    parser.add_argument('--st_type', type=str, default=base_conf['model']['st_type'], choices=['exp', 'sigmoid', 'softplus'],
                        help='Type of ST network for GraphAF (exp, sigmoid, softplus).')

    # --- GraphEBM Specific Model Hyperparameters ---
    parser.add_argument('--ebm_hidden', type=int, default=base_conf['model_ebm']['hidden'],
                        help='Hidden dimension for GraphEBM\'s internal networks.')

    # --- GraphEBM Specific Training Hyperparameters ---
    parser.add_argument('--ebm_c', type=float, default=base_conf['train_ebm']['c'],
                        help='Dequantization coefficient for GraphEBM training.')
    parser.add_argument('--ebm_ld_step', type=int, default=base_conf['train_ebm']['ld_step'],
                        help='Number of Langevin dynamics steps for GraphEBM.')
    parser.add_argument('--ebm_ld_noise', type=float, default=base_conf['train_ebm']['ld_noise'],
                        help='Noise level for Langevin dynamics in GraphEBM.')
    parser.add_argument('--ebm_ld_step_size', type=float, default=base_conf['train_ebm']['ld_step_size'],
                        help='Step size for Langevin dynamics in GraphEBM.')
    parser.add_argument('--ebm_alpha', type=float, default=base_conf['train_ebm']['alpha'],
                        help='Weight for the energy regularizer term in GraphEBM loss.')

    args = parser.parse_args()
    main(args)
