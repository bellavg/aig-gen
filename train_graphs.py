#!/usr/bin/env python3
# train_graphs.py - Modified to use loader-only dataset
import os
import argparse
import torch
from torch_geometric.loader import DenseDataLoader
import warnings
import os.path as osp
import traceback  # For error logging

# --- Dataset and Model Imports ---
try:
    # Use the new loader-only dataset class
    # Make sure to name your file/class accordingly
    from data.aig_dataset import AIGPreprocessedDatasetLoader  # MODIFIED
    from GraphDF import GraphDF
    from GraphAF import GraphAF
    from GraphEBM import GraphEBM

    try:
        from GraphDF import aig_config

        print("Imported aig_config from GraphDF package.")
    except ImportError:
        try:
            import data.aig_config as aig_config

            print("Imported aig_config from data directory.")
        except ImportError:
            try:
                import G2PT.configs.aig as aig_config

                print("Imported aig_config from G2PT.configs.")
            except ImportError:
                print("CRITICAL ERROR: Cannot find aig_config.py. Please ensure it's accessible.")
                exit(1)

except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure dataset class, models (GraphDF, GraphAF, GraphEBM), and aig_config.py are accessible.")
    exit(1)
# --- End Imports ---

# --- Base Configuration Dictionary (Defaults) ---
base_conf = {
    "data_name": "aig_ds",  # This will be overridden by args.dataset_name
    "model": {
        "max_size": 64, "node_dim": 4, "bond_dim": 3, "use_gpu": True,
        "edge_unroll": 15, "num_flow_layer": 12, "num_rgcn_layer": 3,
        "nhid": 128, "nout": 128, "deq_coeff": 0.9, "st_type": "exp", "use_df": False
    },
    "model_ebm": {
        "hidden": 64, "depth": 2, "swish_act": True, "add_self": False,
        "dropout": 0.0, "n_power_iterations": 1
    },
    "lr": 0.0005, "weight_decay": 1e-5, "batch_size": 64, "max_epochs": 50,
    "save_interval": 5, "grad_clip_value": 1.0,
    "train_ebm": {
        "c": 0.0, "ld_step": 150, "ld_noise": 0.005, "ld_step_size": 30,
        "clamp_lgd_grad": True, "alpha": 1.0
    }
}


# --- End Base Configuration ---

# REMOVED: allocate_raw_files function is no longer needed

def main(args):
    conf = base_conf.copy()
    conf['model'] = base_conf['model'].copy()
    conf['model_ebm'] = base_conf['model_ebm'].copy()
    conf['train_ebm'] = base_conf['train_ebm'].copy()

    # --- Update Configuration from Arguments ---
    conf['lr'] = getattr(args, 'lr', conf['lr'])
    conf['weight_decay'] = getattr(args, 'weight_decay', conf['weight_decay'])
    conf['batch_size'] = getattr(args, 'batch_size', conf['batch_size'])
    conf['max_epochs'] = getattr(args, 'max_epochs', conf['max_epochs'])
    conf['save_interval'] = getattr(args, 'save_interval', conf['save_interval'])
    conf['grad_clip_value'] = getattr(args, 'grad_clip_value', conf['grad_clip_value'])
    conf['model']['edge_unroll'] = getattr(args, 'edge_unroll', conf['model']['edge_unroll'])
    conf['model']['num_flow_layer'] = getattr(args, 'num_flow_layer', conf['model']['num_flow_layer'])
    conf['model']['num_rgcn_layer'] = getattr(args, 'num_rgcn_layer', conf['model']['num_rgcn_layer'])
    conf['model']['nhid'] = getattr(args, 'gaf_nhid', conf['model']['nhid'])
    conf['model']['nout'] = getattr(args, 'gaf_nout', conf['model']['nout'])
    conf['model']['deq_coeff'] = getattr(args, 'deq_coeff', conf['model']['deq_coeff'])
    conf['model']['st_type'] = getattr(args, 'st_type', conf['model']['st_type'])
    conf['model']['node_dim'] = getattr(aig_config, 'NUM_NODE_FEATURES', 4)
    conf['model']['bond_dim'] = getattr(aig_config, 'NUM_EDGE_FEATURES', 2) + 1
    if args.model_type == 'GraphDF': conf['model']['use_df'] = True
    conf['model_ebm']['hidden'] = getattr(args, 'ebm_hidden', conf['model_ebm']['hidden'])
    conf['model_ebm']['depth'] = getattr(args, 'ebm_depth', conf['model_ebm']['depth'])
    conf['model_ebm']['swish_act'] = getattr(args, 'ebm_swish_act', conf['model_ebm']['swish_act'])
    conf['model_ebm']['add_self'] = getattr(args, 'ebm_add_self', conf['model_ebm']['add_self'])
    conf['model_ebm']['dropout'] = getattr(args, 'ebm_dropout', conf['model_ebm']['dropout'])
    conf['model_ebm']['n_power_iterations'] = getattr(args, 'ebm_n_power_iterations',
                                                      conf['model_ebm']['n_power_iterations'])
    conf['train_ebm']['c'] = getattr(args, 'ebm_c', conf['train_ebm']['c'])
    conf['train_ebm']['ld_step'] = getattr(args, 'ebm_ld_step', conf['train_ebm']['ld_step'])
    conf['train_ebm']['ld_noise'] = getattr(args, 'ebm_ld_noise', conf['train_ebm']['ld_noise'])
    conf['train_ebm']['ld_step_size'] = getattr(args, 'ebm_ld_step_size', conf['train_ebm']['ld_step_size'])
    conf['train_ebm']['alpha'] = getattr(args, 'ebm_alpha', conf['train_ebm']['alpha'])
    conf['train_ebm']['clamp_lgd_grad'] = getattr(args, 'ebm_clamp_lgd_grad', conf['train_ebm']['clamp_lgd_grad'])
    conf['data_name'] = args.dataset_name

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda');
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu');
        print("Using CPU.")
    conf['model']['use_gpu'] = (device.type == 'cuda')

    # --- Direct torch.load Test (add this section) ---
    print(f"\n--- Direct torch.load Test (before AIGPreprocessedDatasetLoader) ---")
    # Path as confirmed by your investigation script and error messages:
    direct_path_to_pt_file = "/gpfs/home6/igardner1/aig-gen/data/aigs_pyg/aig/processed/train_processed_data.pt"

    print(f"Attempting to load: {direct_path_to_pt_file}")

    if osp.exists(direct_path_to_pt_file):
        print(f"File exists at '{direct_path_to_pt_file}'. Attempting torch.load...")
        try:
            # InMemoryDataset saves a tuple: (data, slices)
            # weights_only=False is important if it contains Tensors and not just state_dict
            loaded_data_tuple = torch.load(direct_path_to_pt_file, map_location='cpu', weights_only=False)
            print(f"Successfully loaded with torch.load!")
            if isinstance(loaded_data_tuple, tuple) and len(loaded_data_tuple) == 2:
                print(f"Loaded data is a tuple of length 2 (expected for InMemoryDataset: data, slices).")
                # You can add more checks here if you know the structure, e.g.:
                # print(f"Type of first element (data): {type(loaded_data_tuple[0])}")
                # print(f"Type of second element (slices): {type(loaded_data_tuple[1])}")
            else:
                print(f"Warning: Loaded data is not a tuple of length 2. Type: {type(loaded_data_tuple)}")
            # To prevent holding large data in memory if not needed for this test, you can delete it
            del loaded_data_tuple
        except Exception as e_torch_load:
            print(f"!!! torch.load FAILED for '{direct_path_to_pt_file}' !!!")
            print(f"Error type: {type(e_torch_load).__name__}")
            print(f"Error message: {e_torch_load}")
            print(f"This suggests the .pt file might be corrupted or not in the expected format for torch.load.")
            # Optionally, re-raise the error if you want the script to stop here on failure
            # raise # Uncomment to stop if torch.load fails
    else:
        print(
            f"File NOT FOUND at '{direct_path_to_pt_file}' according to osp.exists. This contradicts the investigation script!")
    print(f"--- End Direct torch.load Test ---\n")
    # --- End Direct torch.load Test ---

    # --- Dataset Loading (Training Only for this script example) ---
    print(f"\nLoading pre-processed dataset from root: {args.data_root}, name: {args.dataset_name}")
    try:
        print("Instantiating AIGPreprocessedDatasetLoader for Training...")
        # MODIFIED: Call to dataset constructor with fewer arguments
        train_dataset = AIGPreprocessedDatasetLoader(
            root=args.data_root,
            dataset_name=args.dataset_name,
            split="train"
            # num_augmentations is no longer relevant for this loader
        )
        print(f"Total training samples available: {len(train_dataset)}")
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty after loading. Check paths and file content.")

    except FileNotFoundError as fnf_e:
        print(f"Error: Pre-processed training file not found. {fnf_e}")
        print("Please ensure dataset .pt files exist and paths are correct:")
        print(f"  Expected location structure: {args.data_root}/{args.dataset_name}/processed/train_processed_data.pt")
        exit(1)
    except RuntimeError as rt_e:
        print(f"Error loading training dataset (RuntimeError): {rt_e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading training dataset: {e}")
        traceback.print_exc()
        exit(1)

    train_loader = DenseDataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=True, drop_last=True)
    print(f"Created Training DataLoader with batch size {conf['batch_size']}.")

    print(f"Instantiating model runner: {args.model_type}")
    runner = None
    if args.model_type == 'GraphDF':
        runner = GraphDF()
    elif args.model_type == 'GraphAF':
        runner = GraphAF()
    elif args.model_type == 'GraphEBM':
        try:
            runner = GraphEBM(n_atom=conf['model']['max_size'], n_atom_type=conf['model']['node_dim'],
                              n_edge_type=conf['model']['bond_dim'], **conf['model_ebm'], device=device)
        except Exception as e:
            print(f"Error instantiating GraphEBM: {e}"); exit(1)
    else:
        print(f"Error: Unknown model type '{args.model_type}'."); exit(1)
    if runner is None: print(f"Failed to instantiate model runner for {args.model_type}"); exit(1)

    if args.save_dir:
        save_dir = args.save_dir
    else:
        default_save_dir_base = "outputs"
        model_specific_path = f"{args.model_type}/rand_gen_{args.dataset_name}_ckpts"
        save_dir = osp.join(default_save_dir_base, model_specific_path)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model checkpoints will be saved in: {save_dir}")

    print(f"\n--- Starting Training ({args.model_type} on {args.dataset_name}) ---")
    print(f"Delegating training loop to runner: {args.model_type}.train_rand_gen")
    try:
        if args.model_type == 'GraphDF' or args.model_type == 'GraphAF':
            if not hasattr(runner, 'train_rand_gen'): raise NotImplementedError(
                f"{args.model_type} runner missing 'train_rand_gen' method.")
            runner.train_rand_gen(
                loader=train_loader, lr=conf['lr'], wd=conf['weight_decay'], max_epochs=conf['max_epochs'],
                model_conf_dict=conf['model'], save_interval=conf['save_interval'], save_dir=save_dir
            )
        elif args.model_type == 'GraphEBM':
            if not hasattr(runner, 'train_rand_gen'): raise NotImplementedError(
                f"{args.model_type} runner missing 'train_rand_gen' method.")
            runner.train_rand_gen(
                loader=train_loader, lr=conf['lr'], wd=conf['weight_decay'], max_epochs=conf['max_epochs'],
                c=conf['train_ebm']['c'], ld_step=conf['train_ebm']['ld_step'], ld_noise=conf['train_ebm']['ld_noise'],
                ld_step_size=conf['train_ebm']['ld_step_size'], clamp_lgd_grad=conf['train_ebm']['clamp_lgd_grad'],
                alpha=conf['train_ebm']['alpha'], save_interval=conf['save_interval'], save_dir=save_dir,
                grad_clip_value=conf['grad_clip_value']
            )
        else:
            print(f"Model type {args.model_type} not recognized for training delegation."); exit(1)
        print("\n--- Training Process Delegated and Finished ---")
    except NotImplementedError as nie:
        print(f"\nError: Method not implemented: {nie}"); exit(1)
    except Exception as train_e:
        print(f"\nAn error occurred during training delegated to {args.model_type}.train_rand_gen:");
        print(f"Error Type: {type(train_e).__name__}");
        print(f"Error Details: {train_e}");
        print("--- Traceback ---");
        traceback.print_exc();
        print("-----------------");
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train graph generation models (delegated loop, pre-processed data).")

    # --- Essential Arguments ---
    parser.add_argument('--model_type', type=str, required=True, choices=['GraphDF', 'GraphAF', 'GraphEBM'],
                        help='Model runner class to use.')
    parser.add_argument('--data_root', required=True,
                        help="Root directory containing processed dataset folders (e.g., ./aigs_pyg). This is where 'dataset_name/processed/*.pt' will be expected.")
    parser.add_argument('--dataset_name', required=True, help="Name of the dataset subfolder (e.g., 'aig_graphs_v1').")
    parser.add_argument('--edge_unroll', type=int, required=True,
                        help="Edge unroll value (potentially used by model or data interpretation).")

    # --- Optional Overrides & Configuration ---
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for training.')
    parser.add_argument('--save_dir', type=str, default=None, help='Override default save directory.')

    # --- Training Hyperparameters ---
    parser.add_argument('--lr', type=float, help=f"Learning rate (default: {base_conf['lr']}).")
    parser.add_argument('--weight_decay', type=float, help=f"Weight decay (default: {base_conf['weight_decay']}).")
    parser.add_argument('--batch_size', type=int, help=f"Batch size (default: {base_conf['batch_size']}).")
    parser.add_argument('--max_epochs', type=int, help=f"Maximum training epochs (default: {base_conf['max_epochs']}).")
    parser.add_argument('--save_interval', type=int,
                        help=f"Save checkpoints every N epochs (default: {base_conf['save_interval']}).")
    parser.add_argument('--grad_clip_value', type=float,
                        help=f"Max norm for gradient clipping (default: {base_conf['grad_clip_value']}). 0 or None to disable.")

    # --- Model Architecture Hyperparameters ---
    parser.add_argument('--num_flow_layer', type=int,
                        help=f"Number of flow layers (default: {base_conf['model']['num_flow_layer']}).")
    parser.add_argument('--num_rgcn_layer', type=int,
                        help=f"Number of RGCN layers (default: {base_conf['model']['num_rgcn_layer']}).")
    parser.add_argument('--gaf_nhid', type=int, help=f"Hidden dim for GAF/GDF (default: {base_conf['model']['nhid']}).")
    parser.add_argument('--gaf_nout', type=int, help=f"Output dim for GAF/GDF (default: {base_conf['model']['nout']}).")
    parser.add_argument('--deq_coeff', type=float,
                        help=f"Dequantization coefficient (default: {base_conf['model']['deq_coeff']}).")
    parser.add_argument('--st_type', type=str, choices=['exp', 'sigmoid', 'softplus'],
                        help=f"ST network type (default: {base_conf['model']['st_type']}).")
    parser.add_argument('--ebm_hidden', type=int, help=f"EBM hidden dim (default: {base_conf['model_ebm']['hidden']}).")
    parser.add_argument('--ebm_depth', type=int, help=f"EBM depth (default: {base_conf['model_ebm']['depth']}).")
    parser.add_argument('--ebm_swish_act', action=argparse.BooleanOptionalAction,
                        default=base_conf['model_ebm']['swish_act'], help="Use Swish activation in EBM.")
    parser.add_argument('--ebm_add_self', action=argparse.BooleanOptionalAction,
                        default=base_conf['model_ebm']['add_self'], help="Add self-connections in EBM.")
    parser.add_argument('--ebm_dropout', type=float,
                        help=f"EBM dropout (default: {base_conf['model_ebm']['dropout']}).")
    parser.add_argument('--ebm_n_power_iterations', type=int,
                        help=f"EBM power iterations (default: {base_conf['model_ebm']['n_power_iterations']}).")

    # --- EBM Training Hyperparameters ---
    parser.add_argument('--ebm_c', type=float, help=f"EBM dequant scale (default: {base_conf['train_ebm']['c']}).")
    parser.add_argument('--ebm_ld_step', type=int,
                        help=f"EBM Langevin steps (default: {base_conf['train_ebm']['ld_step']}).")
    parser.add_argument('--ebm_ld_noise', type=float,
                        help=f"EBM Langevin noise std (default: {base_conf['train_ebm']['ld_noise']}).")
    parser.add_argument('--ebm_ld_step_size', type=float,
                        help=f"EBM Langevin step size (default: {base_conf['train_ebm']['ld_step_size']}).")
    parser.add_argument('--ebm_alpha', type=float,
                        help=f"EBM regularization weight (default: {base_conf['train_ebm']['alpha']}).")
    parser.add_argument('--ebm_clamp_lgd_grad', action=argparse.BooleanOptionalAction,
                        default=base_conf['train_ebm']['clamp_lgd_grad'], help="Clamp Langevin gradients.")

    args = parser.parse_args()
    main(args)
