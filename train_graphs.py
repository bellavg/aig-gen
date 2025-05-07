import os
import json
import argparse
import torch
from torch_geometric.loader import DenseDataLoader
import warnings
import os.path as osp

# --- Assume ggraph and your AIG dataset loader are importable ---
try:
    # Adjust this path if your AIGDatasetLoader is located elsewhere
    # For example, if it's in a 'data_handling' directory: from data_handling.aig_dataset import AIGDatasetLoader
    from data.aig_dataset import AIGDatasetLoader
    from GraphDF import GraphDF # Assuming these are in the current dir or PYTHONPATH
    from GraphAF import GraphAF
    # Ensure this is the correct import path for your GraphEBM
    from GraphEBM import GraphEBM
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'ggraph' (including dig.ggraph) is installed, your AIGDatasetLoader class is accessible, "
          "and GraphDF/GraphAF/GraphEBM models are correctly placed and importable.")
    exit(1)
# --- End Imports ---


# --- Base Configuration Dictionary ---
base_conf = {
    "data_name": "aig", # Or your specific dataset name if it differs
    "model": { # For GraphAF/DF and common EBM params
        "max_size": 64,    # Corresponds to n_atom for EBM, adjust as per your AIG dataset
        "node_dim": 4,     # Corresponds to n_atom_type for EBM (e.g., CONST0, PI, AND, PO for AIGs)
        "bond_dim": 3,     # Corresponds to n_edge_type for EBM (e.g., REG, INV, NO_EDGE for AIGs)
        "use_gpu": True,

        # GraphDF/GraphAF specific parameters
        "edge_unroll": 12,
        "num_flow_layer": 12,
        "num_rgcn_layer": 3,
        "nhid": 128,
        "nout": 128,
        "deq_coeff": 0.9,
        "st_type": "exp", # or "sigmoid" or "softplus"
        "use_df": False # This might be specific to GraphAF/DF, set True for GraphDF if it uses it
    },
    "model_ebm": { # GraphEBM specific model parameters
        "hidden": 64,
        "depth": 2,
        "swish_act": True,
        "add_self": False, # Or True, depending on your EBM architecture
        "dropout": 0.0,
        "n_power_iterations": 1 # For spectral normalization if used
    },
    "lr": 0.001,
    "weight_decay": 0,
    "batch_size": 128, # Adjust based on your GPU memory and dataset
    "max_epochs": 50,
    "save_interval": 5,
    "grad_clip_value": None, # Default for gradient clipping of model parameters (e.g., 1.0 or 5.0)
    # "save_dir": "GraphDF/rand_gen_aig_ckpts", # Default, will be overridden if --save_dir is provided
                                                # or constructed based on model_type if --save_dir is not given.

    "train_ebm": { # GraphEBM specific training parameters
        "c": 0.0, # Controls strength of energy regularization or other EBM specific terms
        "ld_step": 150, # Number of Langevin dynamics steps for sampling
        "ld_noise": 0.005, # Noise level for Langevin dynamics
        "ld_step_size": 30, # Step size for Langevin dynamics
        "clamp_lgd_grad": True, # Whether to clamp gradients during Langevin dynamics
        "alpha": 1.0 # Weight for reconstruction loss or other EBM terms
    }
}
# --- End Base Configuration ---


def main(args):
    conf = base_conf.copy()
    # Deep copy nested dictionaries to avoid modifying base_conf inadvertently
    conf['model'] = base_conf['model'].copy()
    conf['model_ebm'] = base_conf['model_ebm'].copy()
    conf['train_ebm'] = base_conf['train_ebm'].copy()

    # General training params from command line arguments
    conf['lr'] = args.lr
    conf['weight_decay'] = args.weight_decay
    conf['batch_size'] = args.batch_size
    conf['max_epochs'] = args.max_epochs
    conf['save_interval'] = args.save_interval
    conf['grad_clip_value'] = args.grad_clip_value

    # GraphAF/GraphDF specific model params from command line arguments
    conf['model']['edge_unroll'] = args.edge_unroll
    conf['model']['num_flow_layer'] = args.num_flow_layer
    conf['model']['num_rgcn_layer'] = args.num_rgcn_layer
    conf['model']['nhid'] = args.gaf_nhid
    conf['model']['nout'] = args.gaf_nout
    conf['model']['deq_coeff'] = args.deq_coeff
    conf['model']['st_type'] = args.st_type
    # If GraphDF needs a specific flag (like use_df for Discrete Flow):
    if args.model_type == 'GraphDF':
        conf['model']['use_df'] = True # Example: Set to True if GraphDF implies discrete flow

    # GraphEBM specific model params from command line arguments
    conf['model_ebm']['hidden'] = args.ebm_hidden
    conf['model_ebm']['depth'] = args.ebm_depth
    conf['model_ebm']['swish_act'] = args.ebm_swish_act
    conf['model_ebm']['add_self'] = args.ebm_add_self
    conf['model_ebm']['dropout'] = args.ebm_dropout
    conf['model_ebm']['n_power_iterations'] = args.ebm_n_power_iterations

    # GraphEBM specific training params from command line arguments
    conf['train_ebm']['c'] = args.ebm_c
    conf['train_ebm']['ld_step'] = args.ebm_ld_step
    conf['train_ebm']['ld_noise'] = args.ebm_ld_noise
    conf['train_ebm']['ld_step_size'] = args.ebm_ld_step_size
    conf['train_ebm']['alpha'] = args.ebm_alpha
    conf['train_ebm']['clamp_lgd_grad'] = args.ebm_clamp_lgd_grad

    # Determine device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda')
        # It's good practice to also set torch.cuda.set_device(0) or based on an arg if multiple GPUs
        # However, DataParallel handles multi-GPU usage if enabled in the model.
        print(f"Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A - CUDA not fully initialized?'}")
    else:
        device = torch.device('cpu')
        print("Using CPU.")
    conf['model']['use_gpu'] = (device.type == 'cuda') # Update conf for model instantiation

    # --- Dataset Loading ---
    print("Instantiating AIGDatasetLoader...")
    try:
        # Ensure your AIGDatasetLoader is correctly implemented to return PyG Data objects
        # with 'x' (node features) and 'adj' (adjacency representation suitable for your model)
        dataset = AIGDatasetLoader(
            root=args.data_root, # e.g., "./data/"
            name=conf.get('data_name', 'aig'), # e.g., "aig"
            dataset_type="train" # Or "valid", "test" as needed
        )
        print(f"Number of training graphs loaded: {len(dataset)}")
        if len(dataset) == 0:
            # Provide a more specific path for the user to check
            expected_data_path = osp.join(args.data_root, conf.get('data_name', 'aig'), 'processed', 'train', 'data.pt') # Example for PyG
            print(f"Error: Training dataset is empty. Check path: '{expected_data_path}' or similar, "
                  "and ensure your AIGDatasetLoader processes data correctly.")
            exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure --data_root ('{args.data_root}') points to the correct directory "
              f"containing your '{conf.get('data_name', 'aig')}' dataset.")
        exit(1)
    except Exception as e:
        print(f"Error instantiating AIGDatasetLoader: {e}")
        exit(1)

    # DataLoader
    # For DenseDataLoader, your dataset items should be dense.
    # If your AIGs are sparse, consider DataLoader from torch_geometric.data
    loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)
    print(f"Created DataLoader with batch size {conf['batch_size']}.")

    # --- Model Instantiation ---
    print(f"Instantiating model: {args.model_type}")
    runner = None # This will be an instance of GraphDF, GraphAF, or GraphEBM
    # These runner classes should internally instantiate their respective PyTorch models (like GraphFlowModel for GraphDF)
    if args.model_type == 'GraphDF':
        runner = GraphDF() # GraphDF internally creates GraphFlowModel
        # The runner's get_model or train_rand_gen method will handle passing model_conf_dict
    elif args.model_type == 'GraphAF':
        runner = GraphAF() # GraphAF internally creates its model
    elif args.model_type == 'GraphEBM':
        # GraphEBM runner might take parameters directly or via a config
        # Pass necessary parameters from conf['model'] and conf['model_ebm']
        try:
            runner = GraphEBM( # Assuming GraphEBM class takes these directly
                n_atom=conf['model']['max_size'],
                n_atom_type=conf['model']['node_dim'],
                n_edge_type=conf['model']['bond_dim'],
                hidden=conf['model_ebm']['hidden'],
                depth=conf['model_ebm']['depth'],
                swish_act=conf['model_ebm']['swish_act'],
                add_self=conf['model_ebm']['add_self'],
                dropout=conf['model_ebm']['dropout'],
                n_power_iterations=conf['model_ebm']['n_power_iterations'],
                device=device # Pass the determined device
            )
        except KeyError as e:
            print(f"Error: Missing required parameter {e} in config for GraphEBM initialization.")
            exit(1)
        except Exception as e:
            print(f"Error instantiating GraphEBM: {e}")
            exit(1)
    else:
        print(f"Error: Unknown model type '{args.model_type}'.")
        exit(1)

    if runner is None:
        print(f"Failed to instantiate model runner for {args.model_type}")
        exit(1)

    # --- Determine Save Directory ---
    if args.save_dir:
        # Use the command-line argument if provided
        save_dir = args.save_dir
        conf['save_dir'] = save_dir # Update conf for consistency if used elsewhere
        print(f"Using provided save directory from --save_dir: {save_dir}")
    else:
        # Fallback to internal construction if --save_dir is not provided
        # This was the original behavior. You can customize this.
        # For example, use a sub-directory within a base 'outputs' folder.
        default_save_dir_base = "outputs" # Example base
        model_specific_path = f"{args.model_type}/rand_gen_{conf.get('data_name', 'aig')}_ckpts"
        save_dir = osp.join(default_save_dir_base, model_specific_path)
        conf['save_dir'] = save_dir # Update conf
        print(f"Constructed save directory (since --save_dir was not provided): {save_dir}")

    os.makedirs(conf['save_dir'], exist_ok=True) # Use conf['save_dir'] which is now correctly set
    print(f"Model checkpoints will be saved in: {conf['save_dir']}")


    # --- Training ---
    print(f"Starting training for {args.model_type}...")
    # The runner's train_rand_gen method should handle:
    # 1. Instantiating its internal PyTorch model (e.g., GraphFlowModel for GraphDF)
    # 2. Moving the model to the correct device
    # 3. Setting up optimizer
    # 4. The training loop
    if args.model_type == 'GraphDF' or args.model_type == 'GraphAF':
        # These models might take the model_conf_dict to instantiate their internal nn.Module
        runner.train_rand_gen(
            loader=loader,
            lr=conf['lr'],
            wd=conf['weight_decay'],
            max_epochs=conf['max_epochs'],
            model_conf_dict=conf['model'], # Pass the relevant model config
            save_interval=conf['save_interval'],
            save_dir=conf['save_dir'] # Pass the determined save_dir
            # Add grad_clip_value if these runners support it
        )
    elif args.model_type == 'GraphEBM':
        # GraphEBM's train_rand_gen might take EBM-specific training params
        try:
            runner.train_rand_gen(
                loader=loader,
                lr=conf['lr'],
                wd=conf['weight_decay'],
                max_epochs=conf['max_epochs'],
                c=conf['train_ebm']['c'],
                ld_step=conf['train_ebm']['ld_step'],
                ld_noise=conf['train_ebm']['ld_noise'],
                ld_step_size=conf['train_ebm']['ld_step_size'],
                clamp_lgd_grad=conf['train_ebm']['clamp_lgd_grad'],
                alpha=conf['train_ebm']['alpha'],
                save_interval=conf['save_interval'],
                save_dir=conf['save_dir'], # Pass the determined save_dir
                grad_clip_value=conf['grad_clip_value'] # Pass grad_clip_value here
            )
        except KeyError as e:
            print(f"Error: Missing required parameter {e} for GraphEBM training in conf['train_ebm'].")
            exit(1)
        except AttributeError as e: # If runner doesn't have train_rand_gen or expects different args
            print(f"Error: {args.model_type} runner method issue: {e}")
            exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during GraphEBM training: {e}")
            exit(1)

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train graph generation models on pre-processed AIG dataset.")

    # --- General Arguments ---
    parser.add_argument('--model_type', type=str, default='GraphDF', # Changed default to GraphDF for testing
                        choices=['GraphDF', 'GraphAF', 'GraphEBM'],
                        help='Model to train.')
    parser.add_argument('--data_root', default="./data/", # Ensure this points to where "aig" (or your data_name) folder is
                        help="Root directory for data (e.g., contains 'aig/processed/train/data.pt').")
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device for training (e.g., cuda, cpu).')
    # MODIFIED: Added --save_dir argument
    parser.add_argument('--save_dir', type=str, default=None, # Default is None, so internal logic takes over
                        help='Directory to save model checkpoints. Overrides internal construction if provided.')

    # --- General Training Hyperparameters ---
    parser.add_argument('--lr', type=float, default=base_conf['lr'])
    parser.add_argument('--weight_decay', type=float, default=base_conf['weight_decay'])
    parser.add_argument('--batch_size', type=int, default=base_conf['batch_size'])
    parser.add_argument('--max_epochs', type=int, default=base_conf['max_epochs'])
    parser.add_argument('--save_interval', type=int, default=base_conf['save_interval'])
    parser.add_argument('--grad_clip_value', type=float, default=base_conf['grad_clip_value'],
                        help='Value for gradient norm clipping for the main optimizer. None or 0 to disable.')

    # --- GraphAF/GraphDF Specific Model Hyperparameters ---
    # These are passed via model_conf_dict to the runner
    parser.add_argument('--edge_unroll', type=int, default=base_conf['model']['edge_unroll'])
    parser.add_argument('--num_flow_layer', type=int, default=base_conf['model']['num_flow_layer'])
    parser.add_argument('--num_rgcn_layer', type=int, default=base_conf['model']['num_rgcn_layer'])
    parser.add_argument('--gaf_nhid', type=int, default=base_conf['model']['nhid'], help="Hidden dim for GAF/GDF's RGCN/ST-nets.")
    parser.add_argument('--gaf_nout', type=int, default=base_conf['model']['nout'], help="Output dim for GAF/GDF's RGCN (embedding size).")
    parser.add_argument('--deq_coeff', type=float, default=base_conf['model']['deq_coeff'], help="Dequantization coefficient if used.")
    parser.add_argument('--st_type', type=str, default=base_conf['model']['st_type'],
                        choices=['exp', 'sigmoid', 'softplus'], help="Type of ST network for GAF/GDF.")

    # --- GraphEBM Model Hyperparameters ---
    # These are passed to the GraphEBM runner/constructor
    parser.add_argument('--ebm_hidden', type=int, default=base_conf['model_ebm']['hidden'])
    parser.add_argument('--ebm_depth', type=int, default=base_conf['model_ebm']['depth'])
    # For boolean flags, store_true/store_false or BooleanOptionalAction (Python 3.9+)
    # Using BooleanOptionalAction for broader compatibility if Python version allows.
    # If using older Python, you might need separate --ebm_swish_act and --no-ebm_swish_act.
    parser.add_argument('--ebm_swish_act', action=argparse.BooleanOptionalAction, default=base_conf['model_ebm']['swish_act'])
    parser.add_argument('--ebm_add_self', action=argparse.BooleanOptionalAction, default=base_conf['model_ebm']['add_self'])
    parser.add_argument('--ebm_dropout', type=float, default=base_conf['model_ebm']['dropout'])
    parser.add_argument('--ebm_n_power_iterations', type=int, default=base_conf['model_ebm']['n_power_iterations'])

    # --- GraphEBM Training Hyperparameters ---
    # These are passed to the GraphEBM runner's train_rand_gen method
    parser.add_argument('--ebm_c', type=float, default=base_conf['train_ebm']['c'])
    parser.add_argument('--ebm_ld_step', type=int, default=base_conf['train_ebm']['ld_step'])
    parser.add_argument('--ebm_ld_noise', type=float, default=base_conf['train_ebm']['ld_noise'])
    parser.add_argument('--ebm_ld_step_size', type=float, default=base_conf['train_ebm']['ld_step_size'])
    parser.add_argument('--ebm_alpha', type=float, default=base_conf['train_ebm']['alpha'])
    parser.add_argument('--ebm_clamp_lgd_grad', action=argparse.BooleanOptionalAction,
                        default=base_conf['train_ebm']['clamp_lgd_grad'])

    args = parser.parse_args()
    main(args)
