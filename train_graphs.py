#!/usr/bin/env python3
# train_graphs.py - Fixed to pass raw file info to dataset
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
    # Use the dataset class with debugging prints
    from aig_dataset_debug import AIGProcessedAugmentedDataset # *** Assuming file is named aig_dataset_debug.py ***
    from GraphDF import GraphDF
    from GraphAF import GraphAF
    from GraphEBM import GraphEBM

    # --- Use a single, canonical config import ---
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
# (Keep base_conf as before)
base_conf = {
    "data_name": "aig_ds",
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


def allocate_raw_files(raw_dir, file_prefix, num_train, num_val, num_test):
    """Lists files in raw_dir and allocates them to splits."""
    if not osp.isdir(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    try:
        all_pkl_files = sorted([
            f for f in os.listdir(raw_dir)
            if f.startswith(file_prefix) and f.endswith(".pkl")
        ])
    except OSError as e:
        raise OSError(f"Error listing files in raw directory '{raw_dir}': {e}")

    if not all_pkl_files:
        raise FileNotFoundError(f"No PKL files found in '{raw_dir}' with prefix '{file_prefix}'.")

    print(f"Found {len(all_pkl_files)} raw PKL files matching prefix '{file_prefix}'.")

    current_idx = 0
    train_files, val_files, test_files = [], [], []
    effective_total = len(all_pkl_files)

    if num_train > 0:
        take_n = min(num_train, effective_total - current_idx)
        if take_n > 0: train_files = all_pkl_files[current_idx : current_idx + take_n]; current_idx += take_n
    if num_val > 0:
        take_n = min(num_val, effective_total - current_idx)
        if take_n > 0: val_files = all_pkl_files[current_idx : current_idx + take_n]; current_idx += take_n
    if num_test > 0:
        take_n = min(num_test, effective_total - current_idx)
        if take_n > 0: test_files = all_pkl_files[current_idx : current_idx + take_n]; current_idx += take_n

    print("Raw File Allocation:")
    print(f"  Train: {train_files if train_files else 'None'}")
    print(f"  Val:   {val_files if val_files else 'None'}")
    print(f"  Test:  {test_files if test_files else 'None'}")
    if current_idx < effective_total:
        warnings.warn(f"{effective_total - current_idx} raw PKL files remain unallocated.")

    return {'train': train_files, 'val': val_files, 'test': test_files}


def main(args):
    """
    Sets up configuration, loads data (providing raw file context),
    instantiates the appropriate model runner, and calls the runner's
    train_rand_gen method to perform training.
    """
    conf = base_conf.copy()
    conf['model'] = base_conf['model'].copy()
    conf['model_ebm'] = base_conf['model_ebm'].copy()
    conf['train_ebm'] = base_conf['train_ebm'].copy()

    # --- Update Configuration from Arguments ---
    # (Keep updates as before)
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
    conf['model_ebm']['n_power_iterations'] = getattr(args, 'ebm_n_power_iterations', conf['model_ebm']['n_power_iterations'])
    conf['train_ebm']['c'] = getattr(args, 'ebm_c', conf['train_ebm']['c'])
    conf['train_ebm']['ld_step'] = getattr(args, 'ebm_ld_step', conf['train_ebm']['ld_step'])
    conf['train_ebm']['ld_noise'] = getattr(args, 'ebm_ld_noise', conf['train_ebm']['ld_noise'])
    conf['train_ebm']['ld_step_size'] = getattr(args, 'ebm_ld_step_size', conf['train_ebm']['ld_step_size'])
    conf['train_ebm']['alpha'] = getattr(args, 'ebm_alpha', conf['train_ebm']['alpha'])
    conf['train_ebm']['clamp_lgd_grad'] = getattr(args, 'ebm_clamp_lgd_grad', conf['train_ebm']['clamp_lgd_grad'])
    conf['data_name'] = args.dataset_name

    # Determine device
    # (Keep device logic as before)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda'); print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu'); print("Using CPU.")
    conf['model']['use_gpu'] = (device.type == 'cuda')

    # --- Allocate Raw Files ---
    # This needs the raw directory path and the file allocation numbers
    try:
        split_files = allocate_raw_files(
            raw_dir=args.raw_data_dir, # Use the new argument
            file_prefix=args.raw_file_prefix, # Use the new argument
            num_train=args.num_train_files,
            num_val=args.num_val_files,
            num_test=args.num_test_files
        )
        train_pkl_files = split_files['train']
        if not train_pkl_files:
            print("Error: No raw PKL files allocated for the training split based on provided numbers.")
            exit(1)
    except (FileNotFoundError, OSError, ValueError) as alloc_e:
        print(f"Error during raw file allocation: {alloc_e}")
        exit(1)

    # --- Dataset Loading (Training Only) ---
    print(f"\nLoading processed augmented dataset from root: {args.data_root}, name: {args.dataset_name}")
    try:
        print("Instantiating AIGProcessedAugmentedDataset for Training...")
        # *** Pass raw file info to the dataset constructor ***
        train_dataset = AIGProcessedAugmentedDataset(
            root=args.data_root,
            dataset_name=args.dataset_name,
            split="train",
            raw_dir=args.raw_data_dir, # Pass raw directory
            file_prefix=args.raw_file_prefix, # Pass prefix
            pkl_file_names_for_split=train_pkl_files, # Pass allocated file list
            num_augmentations=args.num_augmentations
        )
        print(f"Total training samples available: {len(train_dataset)}")
        if len(train_dataset) == 0:
             # This might happen if the loaded file is empty or processing failed silently
             raise ValueError("Training dataset is empty after loading/initialization.")

    except FileNotFoundError as fnf_e:
         print(f"Error: Processed training file not found or dataset class could not initialize.")
         print(f"Details: {fnf_e}")
         print("Please ensure dataset processing completed successfully and paths are correct.")
         exit(1)
    except RuntimeError as rt_e:
         print(f"Error loading training dataset (RuntimeError): {rt_e}")
         print("This might indicate a corrupted file or issue with loaded data structure.")
         exit(1)
    except Exception as e:
         print(f"An unexpected error occurred loading training dataset: {e}")
         traceback.print_exc()
         exit(1)

    # DataLoaders (Training Only)
    train_loader = DenseDataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=True, drop_last=True)
    print(f"Created Training DataLoader with batch size {conf['batch_size']}.")

    # --- Model Runner Instantiation ---
    # (Keep as before)
    print(f"Instantiating model runner: {args.model_type}")
    runner = None
    if args.model_type == 'GraphDF': runner = GraphDF()
    elif args.model_type == 'GraphAF': runner = GraphAF()
    elif args.model_type == 'GraphEBM':
        try:
            runner = GraphEBM(n_atom=conf['model']['max_size'], n_atom_type=conf['model']['node_dim'],
                              n_edge_type=conf['model']['bond_dim'], **conf['model_ebm'], device=device)
        except Exception as e: print(f"Error instantiating GraphEBM: {e}"); exit(1)
    else: print(f"Error: Unknown model type '{args.model_type}'."); exit(1)
    if runner is None: print(f"Failed to instantiate model runner for {args.model_type}"); exit(1)


    # --- Determine Save Directory ---
    # (Keep as before)
    if args.save_dir: save_dir = args.save_dir
    else:
        default_save_dir_base = "outputs"
        model_specific_path = f"{args.model_type}/rand_gen_{args.dataset_name}_ckpts"
        save_dir = osp.join(default_save_dir_base, model_specific_path)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model checkpoints will be saved in: {save_dir}")

    # --- Delegate Training to Runner ---
    # (Keep as before)
    print(f"\n--- Starting Training ({args.model_type} on {args.dataset_name}) ---")
    print(f"Delegating training loop to runner: {args.model_type}.train_rand_gen")
    try:
        if args.model_type == 'GraphDF' or args.model_type == 'GraphAF':
            if not hasattr(runner, 'train_rand_gen'): raise NotImplementedError(f"{args.model_type} runner missing 'train_rand_gen' method.")
            runner.train_rand_gen(
                loader=train_loader, lr=conf['lr'], wd=conf['weight_decay'], max_epochs=conf['max_epochs'],
                model_conf_dict=conf['model'], save_interval=conf['save_interval'], save_dir=save_dir
            )
        elif args.model_type == 'GraphEBM':
            if not hasattr(runner, 'train_rand_gen'): raise NotImplementedError(f"{args.model_type} runner missing 'train_rand_gen' method.")
            runner.train_rand_gen(
                loader=train_loader, lr=conf['lr'], wd=conf['weight_decay'], max_epochs=conf['max_epochs'],
                c=conf['train_ebm']['c'], ld_step=conf['train_ebm']['ld_step'], ld_noise=conf['train_ebm']['ld_noise'],
                ld_step_size=conf['train_ebm']['ld_step_size'], clamp_lgd_grad=conf['train_ebm']['clamp_lgd_grad'],
                alpha=conf['train_ebm']['alpha'], save_interval=conf['save_interval'], save_dir=save_dir,
                grad_clip_value=conf['grad_clip_value']
            )
        else: print(f"Model type {args.model_type} not recognized for training delegation."); exit(1)
        print("\n--- Training Process Delegated and Finished ---")
    except NotImplementedError as nie: print(f"\nError: Method not implemented: {nie}"); exit(1)
    except Exception as train_e:
        print(f"\nAn error occurred during training delegated to {args.model_type}.train_rand_gen:"); print(f"Error Type: {type(train_e).__name__}"); print(f"Error Details: {train_e}"); print("--- Traceback ---"); traceback.print_exc(); print("-----------------"); exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train graph generation models (delegated loop).")

    # --- Essential Arguments ---
    parser.add_argument('--model_type', type=str, required=True, choices=['GraphDF', 'GraphAF', 'GraphEBM'], help='Model runner class to use.')
    parser.add_argument('--data_root', required=True, help="Root directory containing processed dataset folders (e.g., ./aigs_pyg).")
    parser.add_argument('--dataset_name', required=True, help="Name of the dataset subfolder (e.g., 'aig').")
    parser.add_argument('--edge_unroll', type=int, required=True, help="Edge unroll value used during data processing/model training.")
    # *** Added Arguments for Raw File Context ***
    parser.add_argument('--raw_data_dir', type=str, required=True, help="Directory containing the original raw PKL files.")
    parser.add_argument('--raw_file_prefix', type=str, default="real_aigs_part_", help="Prefix of the raw PKL files.")
    parser.add_argument('--num_train_files', type=int, required=True, help="Number of raw PKL files allocated to the training set.")
    parser.add_argument('--num_val_files', type=int, default=0, help="Number of raw PKL files allocated to the validation set (used for allocation).")
    parser.add_argument('--num_test_files', type=int, default=0, help="Number of raw PKL files allocated to the test set (used for allocation).")


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

    # --- Model Architecture Hyperparameters ---
    # (Keep these as before)
    parser.add_argument('--num_flow_layer', type=int, help=f"Number of flow layers (default: {base_conf['model']['num_flow_layer']}).")
    parser.add_argument('--num_rgcn_layer', type=int, help=f"Number of RGCN layers (default: {base_conf['model']['num_rgcn_layer']}).")
    parser.add_argument('--gaf_nhid', type=int, help=f"Hidden dim for GAF/GDF (default: {base_conf['model']['nhid']}).")
    parser.add_argument('--gaf_nout', type=int, help=f"Output dim for GAF/GDF (default: {base_conf['model']['nout']}).")
    parser.add_argument('--deq_coeff', type=float, help=f"Dequantization coefficient (default: {base_conf['model']['deq_coeff']}).")
    parser.add_argument('--st_type', type=str, choices=['exp', 'sigmoid', 'softplus'], help=f"ST network type (default: {base_conf['model']['st_type']}).")
    parser.add_argument('--ebm_hidden', type=int, help=f"EBM hidden dim (default: {base_conf['model_ebm']['hidden']}).")
    parser.add_argument('--ebm_depth', type=int, help=f"EBM depth (default: {base_conf['model_ebm']['depth']}).")
    parser.add_argument('--ebm_swish_act', action=argparse.BooleanOptionalAction, default=base_conf['model_ebm']['swish_act'], help="Use Swish activation in EBM.")
    parser.add_argument('--ebm_add_self', action=argparse.BooleanOptionalAction, default=base_conf['model_ebm']['add_self'], help="Add self-connections in EBM.")
    parser.add_argument('--ebm_dropout', type=float, help=f"EBM dropout (default: {base_conf['model_ebm']['dropout']}).")
    parser.add_argument('--ebm_n_power_iterations', type=int, help=f"EBM power iterations (default: {base_conf['model_ebm']['n_power_iterations']}).")

    # --- EBM Training Hyperparameters ---
    # (Keep these as before)
    parser.add_argument('--ebm_c', type=float, help=f"EBM dequant scale (default: {base_conf['train_ebm']['c']}).")
    parser.add_argument('--ebm_ld_step', type=int, help=f"EBM Langevin steps (default: {base_conf['train_ebm']['ld_step']}).")
    parser.add_argument('--ebm_ld_noise', type=float, help=f"EBM Langevin noise std (default: {base_conf['train_ebm']['ld_noise']}).")
    parser.add_argument('--ebm_ld_step_size', type=float, help=f"EBM Langevin step size (default: {base_conf['train_ebm']['ld_step_size']}).")
    parser.add_argument('--ebm_alpha', type=float, help=f"EBM regularization weight (default: {base_conf['train_ebm']['alpha']}).")
    parser.add_argument('--ebm_clamp_lgd_grad', action=argparse.BooleanOptionalAction, default=base_conf['train_ebm']['clamp_lgd_grad'], help="Clamp Langevin gradients.")

    args = parser.parse_args()
    main(args)

