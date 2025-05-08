#!/usr/bin/env python3
# train_graphs.py - Simplified to delegate training loop to runner classes
import os
import json
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DenseDataLoader
import warnings
import os.path as osp
import numpy as np
import pickle
import networkx as nx
from tqdm import tqdm
import sys
import traceback # For error logging

# --- Dataset and Model Imports ---
try:
    from data.aig_dataset import AIGProcessedAugmentedDataset
    from GraphDF import GraphDF
    from GraphAF import GraphAF
    from GraphEBM import GraphEBM

    # --- Use a single, canonical config import ---
    # Adjust this logic based on your final project structure
    try:
        # Prioritize a canonical location if defined, e.g., within the GraphDF package
        from GraphDF import aig_config
        print("Imported aig_config from GraphDF package.")
    except ImportError:
        try:
            import data.aig_config as aig_config # Check data dir
            print("Imported aig_config from data directory.")
        except ImportError:
            try:
                # Fallback for original structure if needed
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

# --- Constants ---
# Constants are now expected to be defined within the imported aig_config
# AIG_NODE_TYPE_KEYS = getattr(aig_config, 'NODE_TYPE_KEYS', ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"])
# AIG_EDGE_TYPE_KEYS = getattr(aig_config, 'EDGE_TYPE_KEYS', ["EDGE_REG", "EDGE_INV"])
# --- End Constants ---


# --- Base Configuration Dictionary (Defaults) ---
# These provide fallback values if not specified by args
base_conf = {
    "data_name": "aig_ds",
    "model": {
        "max_size": 64,
        "node_dim": 4, # Should match aig_config.NUM_NODE_FEATURES
        "bond_dim": 3, # Should match aig_config.NUM_EDGE_FEATURES + 1 (for no-edge)
        "use_gpu": True,
        "edge_unroll": 15,
        "num_flow_layer": 12,
        "num_rgcn_layer": 3,
        "nhid": 128,
        "nout": 128,
        "deq_coeff": 0.9,
        "st_type": "exp",
        "use_df": False
    },
    "model_ebm": {
        "hidden": 64, "depth": 2, "swish_act": True, "add_self": False,
        "dropout": 0.0, "n_power_iterations": 1
    },
    "lr": 0.0005,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "max_epochs": 50,
    "save_interval": 5,
    "grad_clip_value": 1.0,
    "train_ebm": {
        "c": 0.0, "ld_step": 150, "ld_noise": 0.005, "ld_step_size": 30,
        "clamp_lgd_grad": True, "alpha": 1.0
    }
}
# --- End Base Configuration ---

# --- Removed helper functions: _convert_raw_to_aig_digraph, run_validation, run_generation_and_eval ---

def main(args):
    """
    Sets up configuration, loads data, instantiates the appropriate model runner,
    and calls the runner's train_rand_gen method to perform training.
    """
    conf = base_conf.copy()
    conf['model'] = base_conf['model'].copy()
    conf['model_ebm'] = base_conf['model_ebm'].copy()
    conf['train_ebm'] = base_conf['train_ebm'].copy()

    # --- Update Configuration from Arguments ---
    # General Training Params
    conf['lr'] = getattr(args, 'lr', conf['lr'])
    conf['weight_decay'] = getattr(args, 'weight_decay', conf['weight_decay'])
    conf['batch_size'] = getattr(args, 'batch_size', conf['batch_size'])
    conf['max_epochs'] = getattr(args, 'max_epochs', conf['max_epochs'])
    conf['save_interval'] = getattr(args, 'save_interval', conf['save_interval'])
    conf['grad_clip_value'] = getattr(args, 'grad_clip_value', conf['grad_clip_value']) # Pass this to runner methods

    # Model Architecture Params (Crucial for instantiation)
    conf['model']['edge_unroll'] = getattr(args, 'edge_unroll', conf['model']['edge_unroll'])
    conf['model']['num_flow_layer'] = getattr(args, 'num_flow_layer', conf['model']['num_flow_layer'])
    conf['model']['num_rgcn_layer'] = getattr(args, 'num_rgcn_layer', conf['model']['num_rgcn_layer'])
    conf['model']['nhid'] = getattr(args, 'gaf_nhid', conf['model']['nhid'])
    conf['model']['nout'] = getattr(args, 'gaf_nout', conf['model']['nout'])
    conf['model']['deq_coeff'] = getattr(args, 'deq_coeff', conf['model']['deq_coeff'])
    conf['model']['st_type'] = getattr(args, 'st_type', conf['model']['st_type'])
    # Set node/bond dim based on config or args if needed (usually fixed for AIG)
    conf['model']['node_dim'] = getattr(aig_config, 'NUM_NODE_FEATURES', 4)
    conf['model']['bond_dim'] = getattr(aig_config, 'NUM_EDGE_FEATURES', 2) + 1 # Add 1 for no-edge/virtual

    if args.model_type == 'GraphDF': conf['model']['use_df'] = True

    # EBM Architecture Params
    conf['model_ebm']['hidden'] = getattr(args, 'ebm_hidden', conf['model_ebm']['hidden'])
    conf['model_ebm']['depth'] = getattr(args, 'ebm_depth', conf['model_ebm']['depth'])
    conf['model_ebm']['swish_act'] = getattr(args, 'ebm_swish_act', conf['model_ebm']['swish_act'])
    conf['model_ebm']['add_self'] = getattr(args, 'ebm_add_self', conf['model_ebm']['add_self'])
    conf['model_ebm']['dropout'] = getattr(args, 'ebm_dropout', conf['model_ebm']['dropout'])
    conf['model_ebm']['n_power_iterations'] = getattr(args, 'ebm_n_power_iterations', conf['model_ebm']['n_power_iterations'])

    # EBM Training Params
    conf['train_ebm']['c'] = getattr(args, 'ebm_c', conf['train_ebm']['c'])
    conf['train_ebm']['ld_step'] = getattr(args, 'ebm_ld_step', conf['train_ebm']['ld_step'])
    conf['train_ebm']['ld_noise'] = getattr(args, 'ebm_ld_noise', conf['train_ebm']['ld_noise'])
    conf['train_ebm']['ld_step_size'] = getattr(args, 'ebm_ld_step_size', conf['train_ebm']['ld_step_size'])
    conf['train_ebm']['alpha'] = getattr(args, 'ebm_alpha', conf['train_ebm']['alpha'])
    conf['train_ebm']['clamp_lgd_grad'] = getattr(args, 'ebm_clamp_lgd_grad', conf['train_ebm']['clamp_lgd_grad'])

    # Data/Save Params
    conf['data_name'] = args.dataset_name # Used for potential save dir naming

    # Determine device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    else:
        device = torch.device('cpu')
        print("Using CPU.")
    conf['model']['use_gpu'] = (device.type == 'cuda') # Inform model config about device choice

    # --- Dataset Loading (Training Only) ---
    print(f"Loading processed augmented dataset from root: {args.data_root}, name: {args.dataset_name}")
    try:
        print("Instantiating AIGProcessedAugmentedDataset for Training...")
        # Pass num_augmentations used during processing if dataset needs it
        train_dataset = AIGProcessedAugmentedDataset(root=args.data_root, dataset_name=args.dataset_name, split="train",
                                                     num_augmentations=args.num_augmentations)
        print(f"Total training samples loaded: {len(train_dataset)}")
        if len(train_dataset) == 0: raise ValueError("Training dataset is empty after loading.")
    except FileNotFoundError:
         print(f"Error: Processed training file not found at expected location under {osp.join(args.data_root, args.dataset_name, 'processed')}")
         print("Please ensure you ran the processing script first.")
         exit(1)
    except Exception as e: print(f"Error loading training dataset: {e}"); exit(1)

    # DataLoaders (Training Only)
    train_loader = DenseDataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=True, drop_last=True)
    print(f"Created Training DataLoader with batch size {conf['batch_size']}.")

    # --- Model Runner Instantiation ---
    print(f"Instantiating model runner: {args.model_type}")
    runner = None
    if args.model_type == 'GraphDF': runner = GraphDF()
    elif args.model_type == 'GraphAF': runner = GraphAF()
    elif args.model_type == 'GraphEBM':
        try:
            # Pass EBM specific architecture params and device
            runner = GraphEBM(n_atom=conf['model']['max_size'], n_atom_type=conf['model']['node_dim'],
                              n_edge_type=conf['model']['bond_dim'], **conf['model_ebm'], device=device)
        except Exception as e: print(f"Error instantiating GraphEBM: {e}"); exit(1)
    else: print(f"Error: Unknown model type '{args.model_type}'."); exit(1)
    if runner is None: print(f"Failed to instantiate model runner for {args.model_type}"); exit(1)

    # --- Determine Save Directory ---
    if args.save_dir: save_dir = args.save_dir
    else:
        default_save_dir_base = "outputs"
        # Use dataset_name in path for clarity
        model_specific_path = f"{args.model_type}/rand_gen_{args.dataset_name}_ckpts"
        save_dir = osp.join(default_save_dir_base, model_specific_path)
    # Ensure save_dir exists (runner's train method should also check, but good practice here too)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model checkpoints will be saved in: {save_dir}")

    # --- Delegate Training to Runner ---
    print(f"\n--- Starting Training ({args.model_type} on {args.dataset_name}) ---")
    print(f"Delegating training loop to runner: {args.model_type}.train_rand_gen")

    try:
        if args.model_type == 'GraphDF' or args.model_type == 'GraphAF':
            if not hasattr(runner, 'train_rand_gen'):
                raise NotImplementedError(f"{args.model_type} runner missing 'train_rand_gen' method.")
            # Call GraphDF/GraphAF train_rand_gen
            # Note: Assumes GraphDF/GraphAF's train_rand_gen handles model instantiation/loading internally via get_model
            # Pass necessary hyperparameters
            runner.train_rand_gen(
                loader=train_loader,
                lr=conf['lr'],
                wd=conf['weight_decay'],
                max_epochs=conf['max_epochs'],
                model_conf_dict=conf['model'], # Pass the full model config dict
                save_interval=conf['save_interval'],
                save_dir=save_dir
                # Add grad_clip_value if the runner's method accepts it
                # grad_clip_value=conf['grad_clip_value']
            )
        elif args.model_type == 'GraphEBM':
            if not hasattr(runner, 'train_rand_gen'):
                 raise NotImplementedError(f"{args.model_type} runner missing 'train_rand_gen' method.")
            # Call GraphEBM train_rand_gen
            # Pass necessary hyperparameters including EBM specific ones
            runner.train_rand_gen(
                loader=train_loader,
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
                save_dir=save_dir,
                grad_clip_value=conf['grad_clip_value'] # Pass grad clip value
            )
        else:
            # Should not be reached due to earlier check, but for safety
            print(f"Model type {args.model_type} not recognized for training delegation.")
            exit(1)

        print("\n--- Training Process Delegated and Finished ---")

    except NotImplementedError as nie:
        print(f"\nError: Training method not implemented in runner: {nie}")
        exit(1)
    except Exception as train_e:
        print(f"\nAn error occurred during training delegated to {args.model_type}.train_rand_gen:")
        print(f"Error Type: {type(train_e).__name__}")
        print(f"Error Details: {train_e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")
        exit(1)

    # --- Removed History Saving (handled within runner methods) ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train graph generation models (delegated loop).")

    # --- Essential Arguments ---
    parser.add_argument('--model_type', type=str, required=True, choices=['GraphDF', 'GraphAF', 'GraphEBM'], help='Model runner class to use.')
    parser.add_argument('--data_root', required=True, help="Root directory containing processed dataset folders.")
    parser.add_argument('--dataset_name', required=True, help="Name of the dataset subfolder (e.g., 'aig_ds').")
    parser.add_argument('--edge_unroll', type=int, required=True, help="Edge unroll value used during data processing/model training.")

    # --- Optional Overrides & Configuration ---
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for training.')
    parser.add_argument('--save_dir', type=str, default=None, help='Override default save directory.')
    parser.add_argument('--num_augmentations', type=int, default=5, help='Number of augmentations used for the training dataset.')

    # --- Training Hyperparameters ---
    parser.add_argument('--lr', type=float, help=f"Learning rate (default: {base_conf['lr']}).")
    parser.add_argument('--weight_decay', type=float, help=f"Weight decay (default: {base_conf['weight_decay']}).")
    parser.add_argument('--batch_size', type=int, help=f"Batch size (default: {base_conf['batch_size']}).")
    parser.add_argument('--max_epochs', type=int, help=f"Maximum training epochs (default: {base_conf['max_epochs']}).")
    parser.add_argument('--save_interval', type=int, help=f"Save checkpoints every N epochs (default: {base_conf['save_interval']}).")
    parser.add_argument('--grad_clip_value', type=float, help=f"Max norm for gradient clipping (default: {base_conf['grad_clip_value']}). 0 or None to disable.")

    # --- Model Architecture Hyperparameters (Only needed if different from base_conf defaults) ---
    parser.add_argument('--num_flow_layer', type=int, help=f"Number of flow layers (default: {base_conf['model']['num_flow_layer']}).")
    parser.add_argument('--num_rgcn_layer', type=int, help=f"Number of RGCN layers (default: {base_conf['model']['num_rgcn_layer']}).")
    parser.add_argument('--gaf_nhid', type=int, help=f"Hidden dim for GAF/GDF (default: {base_conf['model']['nhid']}).")
    parser.add_argument('--gaf_nout', type=int, help=f"Output dim for GAF/GDF (default: {base_conf['model']['nout']}).")
    parser.add_argument('--deq_coeff', type=float, help=f"Dequantization coefficient (default: {base_conf['model']['deq_coeff']}).")
    parser.add_argument('--st_type', type=str, choices=['exp', 'sigmoid', 'softplus'], help=f"ST network type (default: {base_conf['model']['st_type']}).")

    # --- EBM Architecture Hyperparameters ---
    parser.add_argument('--ebm_hidden', type=int, help=f"EBM hidden dim (default: {base_conf['model_ebm']['hidden']}).")
    parser.add_argument('--ebm_depth', type=int, help=f"EBM depth (default: {base_conf['model_ebm']['depth']}).")
    parser.add_argument('--ebm_swish_act', action=argparse.BooleanOptionalAction, default=base_conf['model_ebm']['swish_act'], help="Use Swish activation in EBM.")
    parser.add_argument('--ebm_add_self', action=argparse.BooleanOptionalAction, default=base_conf['model_ebm']['add_self'], help="Add self-connections in EBM.")
    parser.add_argument('--ebm_dropout', type=float, help=f"EBM dropout (default: {base_conf['model_ebm']['dropout']}).")
    parser.add_argument('--ebm_n_power_iterations', type=int, help=f"EBM power iterations (default: {base_conf['model_ebm']['n_power_iterations']}).")

    # --- EBM Training Hyperparameters ---
    parser.add_argument('--ebm_c', type=float, help=f"EBM dequant scale (default: {base_conf['train_ebm']['c']}).")
    parser.add_argument('--ebm_ld_step', type=int, help=f"EBM Langevin steps (default: {base_conf['train_ebm']['ld_step']}).")
    parser.add_argument('--ebm_ld_noise', type=float, help=f"EBM Langevin noise std (default: {base_conf['train_ebm']['ld_noise']}).")
    parser.add_argument('--ebm_ld_step_size', type=float, help=f"EBM Langevin step size (default: {base_conf['train_ebm']['ld_step_size']}).")
    parser.add_argument('--ebm_alpha', type=float, help=f"EBM regularization weight (default: {base_conf['train_ebm']['alpha']}).")
    parser.add_argument('--ebm_clamp_lgd_grad', action=argparse.BooleanOptionalAction, default=base_conf['train_ebm']['clamp_lgd_grad'], help="Clamp Langevin gradients.")

    args = parser.parse_args()
    main(args)

