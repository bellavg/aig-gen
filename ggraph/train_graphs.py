#!/usr/bin/env python3
# train_graphs.py - Modified to use loader-only dataset
import argparse
import os
import os.path as osp
import traceback  # For error logging
import torch
from torch_geometric.loader import DenseDataLoader
from GraphDF import GraphDF
from aig_config import *
from use_dataset import AIGPreprocessedDatasetLoader
from GraphAF import GraphAF
from GraphEBM import GraphEBM


# --- End Base Configuration ---

def main(args):
    conf = base_conf.copy()
    conf['model'] = base_conf['model'].copy()
    # conf['model_ebm'] = base_conf['model_ebm'].copy()
    # conf['train_ebm'] = base_conf['train_ebm'].copy()

    # --- Update Configuration from Arguments ---
    # General training params
    conf['lr'] = getattr(args, 'lr', conf['lr'])
    conf['weight_decay'] = getattr(args, 'weight_decay', conf['weight_decay'])
    conf['batch_size'] = getattr(args, 'batch_size', conf['batch_size'])
    conf['max_epochs'] = getattr(args, 'max_epochs', conf['max_epochs'])
    conf['save_interval'] = getattr(args, 'save_interval', conf['save_interval'])
    conf['grad_clip_value'] = getattr(args, 'grad_clip_value', conf['grad_clip_value'])


    # Model-specific params (GraphAF/GraphDF)
    conf['model']['edge_unroll'] = getattr(args, 'edge_unroll', conf['model']['edge_unroll'])
    conf['model']['num_flow_layer'] = getattr(args, 'num_flow_layer', conf['model']['num_flow_layer'])
    conf['model']['num_rgcn_layer'] = getattr(args, 'num_rgcn_layer', conf['model']['num_rgcn_layer'])
    conf['model']['nhid'] = getattr(args, 'gaf_nhid', conf['model']['nhid'])
    conf['model']['nout'] = getattr(args, 'gaf_nout', conf['model']['nout'])


    # --- Device Setup ---
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


    # --- Dataset Loading ---
    print(f"\nLoading pre-processed dataset from root: {args.data_root}")
    train_dataset = AIGPreprocessedDatasetLoader(
        root=args.data_root,
        split="train"
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after loading. Check paths and file content.")


    # --- Model Instantiation ---
    print(f"Instantiating model runner: {args.model}")
    runner = None
    if args.model == 'GraphDF':
        runner = GraphDF()  # Assuming GraphDF() takes no args or uses conf internally
    elif args.model == 'GraphAF':
        runner = GraphAF()  # Assuming GraphAF() takes no args or uses conf internally
    elif args.model == 'GraphEBM':
        runner = GraphEBM(n_atom=MAX_NODE_COUNT, n_atom_type=NUM_NODE_FEATURES,
                          n_edge_type=NUM_EXPLICIT_EDGE_TYPES, hidden=base_conf['model']['hidden'], device=device)
        conf['lr'] = base_conf['ebm_lr']
        conf['batch_size'] = base_conf['ebm_bs']

    else:
        print(f"Error: Unknown model type '{args.model}'.");
        exit(1)

    if runner is None:
        print(f"Failed to instantiate model runner for {args.model}");
        exit(1)

    train_loader = DenseDataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=True, drop_last=True)
    print(f"Created Training DataLoader with batch size {conf['batch_size']}.")


    default_save_dir_base = "./ggraph/checkpoints"  # Relative to where script is run
    model_specific_path = f"{args.model}"
    save_dir = osp.join(default_save_dir_base, model_specific_path)

    # Ensure save_dir is absolute if it was relative, using the CWD
    # Note: If using SBATCH, CWD is usually SLURM_SUBMIT_DIR, so this makes it absolute from there.
    save_dir = osp.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model checkpoints will be saved in: {save_dir}")

    # --- Training ---
    print(f"\n--- Starting Training ({args.model}")
    runner.train_rand_gen(
        loader=train_loader,
        lr=conf['lr'],
        wd=conf['weight_decay'],
        max_epochs=conf['max_epochs'],
        model_conf_dict=conf['model'],  # Pass model config
        save_interval=conf['save_interval'],
        save_dir=save_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train graph generation models (delegated loop, pre-processed data).")

    # --- Essential Arguments ---
    parser.add_argument('--model', type=str, default='GraphDF', choices=['GraphDF', 'GraphAF', 'GraphEBM'],
                        help='Model runner class to use.')
    parser.add_argument('--data_root', default="./ggraph/data/aigs_pyg",
                        help="Root directory containing processed dataset folders (e.g., ./aigs_pyg). This is where 'dataset_name/processed/*.pt' will be expected.")
    parser.add_argument('--edge_unroll', type=int, default=25,
                        help="Edge unroll value (potentially used by model or data interpretation).")

    # --- Optional Overrides & Configuration ---
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for training.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Override default save directory. If relative, it will be relative to the execution directory.')

    # --- Training Hyperparameters ---
    parser.add_argument('--lr', type=float, default=base_conf['lr'],
                        help=f"Learning rate (default: {base_conf['lr']}).")
    parser.add_argument('--weight_decay', type=float, default=base_conf['weight_decay'],
                        help=f"Weight decay (default: {base_conf['weight_decay']}).")
    parser.add_argument('--batch_size', type=int, default=base_conf['batch_size'],
                        help=f"Batch size (default: {base_conf['batch_size']}).")
    parser.add_argument('--max_epochs', type=int, default=base_conf['max_epochs'],
                        help=f"Maximum training epochs (default: {base_conf['max_epochs']}).")
    parser.add_argument('--save_interval', type=int, default=base_conf['save_interval'],
                        help=f"Save checkpoints every N epochs (default: {base_conf['save_interval']}).")

    # --- Model Architecture Hyperparameters ---
    # Note: edge_unroll is essential and required
    parser.add_argument('--num_flow_layer', type=int, default=base_conf['model']['num_flow_layer'],
                        help=f"Number of flow layers (default: {base_conf['model']['num_flow_layer']}).")
    parser.add_argument('--num_rgcn_layer', type=int, default=base_conf['model']['num_rgcn_layer'],
                        help=f"Number of RGCN layers (default: {base_conf['model']['num_rgcn_layer']}).")
    parser.add_argument('--gaf_nhid', type=int, default=base_conf['model']['nhid'],
                        help=f"Hidden dim for GAF/GDF (default: {base_conf['model']['nhid']}).")
    parser.add_argument('--gaf_nout', type=int, default=base_conf['model']['nout'],
                        help=f"Output dim for GAF/GDF (default: {base_conf['model']['nout']}).")
    parser.add_argument('--deq_coeff', type=float,  default=base_conf['model']['deq_coeff'],
                        help=f"Dequantization coefficient (default from base_conf: {base_conf['model']['deq_coeff']}).")


    args = parser.parse_args()
    main(args)
