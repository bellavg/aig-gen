# GraphDF/train_graphs.py

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
    # Import both the base loader and the wrapper
    from data.aig_dataset import AIGDatasetLoader, AugmentedAIGDataset # <-- MODIFIED IMPORT
    from GraphDF import GraphDF # Assuming these are in the current dir or PYTHONPATH
    from GraphAF import GraphAF
    # Ensure this is the correct import path for your GraphEBM
    from GraphEBM import GraphEBM
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'ggraph' (including dig.ggraph) is installed, your AIGDatasetLoader/AugmentedAIGDataset classes are accessible, "
          "and GraphDF/GraphAF/GraphEBM models are correctly placed and importable.")
    exit(1)
# --- End Imports ---


# --- Base Configuration Dictionary (Keep as before) ---
base_conf = {
    "data_name": "aig",
    "model": {
        "max_size": 64,
        "node_dim": 4,
        "bond_dim": 3,
        "use_gpu": True,
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
        "hidden": 64,
        "depth": 2,
        "swish_act": True,
        "add_self": False,
        "dropout": 0.0,
        "n_power_iterations": 1
    },
    "lr": 0.001,
    "weight_decay": 0,
    "batch_size": 128,
    "max_epochs": 50,
    "save_interval": 5,
    "grad_clip_value": None,
    "train_ebm": {
        "c": 0.0,
        "ld_step": 150,
        "ld_noise": 0.005,
        "ld_step_size": 30,
        "clamp_lgd_grad": True,
        "alpha": 1.0
    }
}
# --- End Base Configuration ---


def main(args):
    conf = base_conf.copy()
    # Deep copy nested dictionaries
    conf['model'] = base_conf['model'].copy()
    conf['model_ebm'] = base_conf['model_ebm'].copy()
    conf['train_ebm'] = base_conf['train_ebm'].copy()

    # Update conf from args (General)
    conf['lr'] = args.lr
    conf['weight_decay'] = args.weight_decay
    conf['batch_size'] = args.batch_size
    conf['max_epochs'] = args.max_epochs
    conf['save_interval'] = args.save_interval
    conf['grad_clip_value'] = args.grad_clip_value

    # Update conf from args (GraphAF/DF specific)
    conf['model']['edge_unroll'] = args.edge_unroll
    conf['model']['num_flow_layer'] = args.num_flow_layer
    conf['model']['num_rgcn_layer'] = args.num_rgcn_layer
    conf['model']['nhid'] = args.gaf_nhid
    conf['model']['nout'] = args.gaf_nout
    conf['model']['deq_coeff'] = args.deq_coeff
    conf['model']['st_type'] = args.st_type
    if args.model_type == 'GraphDF':
        conf['model']['use_df'] = True

    # Update conf from args (GraphEBM model specific)
    conf['model_ebm']['hidden'] = args.ebm_hidden
    conf['model_ebm']['depth'] = args.ebm_depth
    conf['model_ebm']['swish_act'] = args.ebm_swish_act
    conf['model_ebm']['add_self'] = args.ebm_add_self
    conf['model_ebm']['dropout'] = args.ebm_dropout
    conf['model_ebm']['n_power_iterations'] = args.ebm_n_power_iterations

    # Update conf from args (GraphEBM train specific)
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
        print(f"Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    else:
        device = torch.device('cpu')
        print("Using CPU.")
    conf['model']['use_gpu'] = (device.type == 'cuda')

    # --- Dataset Loading ---
    print("Instantiating AIGDatasetLoader...")
    try:
        # 1. Instantiate the base loader
        base_train_dataset = AIGDatasetLoader(
            root=args.data_root,
            name=conf.get('data_name', 'aig'),
            dataset_type="train" # Load the 'train' split
        )
        print(f"Base training dataset loaded with {len(base_train_dataset)} graphs.")

        # 2. WRAP the base dataset for augmentation using the new wrapper class
        # Use the num_augmentations argument passed from the command line
        dataset = AugmentedAIGDataset(
            base_dataset=base_train_dataset,
            num_augmentations=args.num_augmentations # Use the arg value
        )
        # --- End Wrapping ---

        print(f"Total training samples including augmentations: {len(dataset)}")
        if len(dataset) == 0:
            expected_data_path = osp.join(args.data_root, conf.get('data_name', 'aig'), 'processed', 'train', 'data.pt')
            print(f"Error: Training dataset (potentially after augmentation logic) is empty. Check path: '{expected_data_path}' or similar.")
            exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure --data_root ('{args.data_root}') points to the correct directory "
              f"containing your '{conf.get('data_name', 'aig')}/processed/train/data.pt' file.")
        exit(1)
    except Exception as e:
        print(f"Error instantiating AIGDatasetLoader or AugmentedAIGDataset: {e}")
        exit(1)

    # DataLoader - Use the wrapped 'dataset' which handles augmentation
    loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)
    print(f"Created DataLoader with batch size {conf['batch_size']} for augmented dataset.")

    # --- Model Instantiation (Remains the same) ---
    print(f"Instantiating model: {args.model_type}")
    runner = None
    if args.model_type == 'GraphDF':
        runner = GraphDF()
    elif args.model_type == 'GraphAF':
        runner = GraphAF()
    elif args.model_type == 'GraphEBM':
        try:
            runner = GraphEBM(
                n_atom=conf['model']['max_size'],
                n_atom_type=conf['model']['node_dim'],
                n_edge_type=conf['model']['bond_dim'],
                hidden=conf['model_ebm']['hidden'],
                depth=conf['model_ebm']['depth'],
                swish_act=conf['model_ebm']['swish_act'],
                add_self=conf['model_ebm']['add_self'],
                dropout=conf['model_ebm']['dropout'],
                n_power_iterations=conf['model_ebm']['n_power_iterations'],
                device=device
            )
        except KeyError as e: print(f"Error: Missing required parameter {e} for GraphEBM init."); exit(1)
        except Exception as e: print(f"Error instantiating GraphEBM: {e}"); exit(1)
    else: print(f"Error: Unknown model type '{args.model_type}'."); exit(1)
    if runner is None: print(f"Failed to instantiate model runner for {args.model_type}"); exit(1)

    # --- Determine Save Directory (Remains the same) ---
    if args.save_dir:
        save_dir = args.save_dir
        conf['save_dir'] = save_dir
        print(f"Using provided save directory from --save_dir: {save_dir}")
    else:
        default_save_dir_base = "outputs" # Example base
        model_specific_path = f"{args.model_type}/rand_gen_{conf.get('data_name', 'aig')}_ckpts"
        save_dir = osp.join(default_save_dir_base, model_specific_path)
        conf['save_dir'] = save_dir
        print(f"Constructed save directory (since --save_dir was not provided): {save_dir}")
    os.makedirs(conf['save_dir'], exist_ok=True)
    print(f"Model checkpoints will be saved in: {conf['save_dir']}")


    # --- Training (Remains the same logic, uses the augmented loader) ---
    print(f"Starting training for {args.model_type}...")
    if args.model_type == 'GraphDF' or args.model_type == 'GraphAF':
        runner.train_rand_gen(
            loader=loader, # Pass the augmented loader
            lr=conf['lr'],
            wd=conf['weight_decay'],
            max_epochs=conf['max_epochs'],
            model_conf_dict=conf['model'],
            save_interval=conf['save_interval'],
            save_dir=conf['save_dir']
        )
    elif args.model_type == 'GraphEBM':
        try:
            runner.train_rand_gen(
                loader=loader, # Pass the augmented loader
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
                save_dir=conf['save_dir'],
                grad_clip_value=conf['grad_clip_value']
            )
        except KeyError as e: print(f"Error: Missing required parameter {e} for GraphEBM training."); exit(1)
        except AttributeError as e: print(f"Error: {args.model_type} runner method issue: {e}"); exit(1)
        except Exception as e: print(f"An unexpected error occurred during GraphEBM training: {e}"); exit(1)

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train graph generation models on pre-processed AIG dataset.")

    # --- General Arguments ---
    parser.add_argument('--model_type', type=str, default='GraphDF',
                        choices=['GraphDF', 'GraphAF', 'GraphEBM'], help='Model to train.')
    parser.add_argument('--data_root', default="./data/",
                        help="Root directory for data (e.g., contains 'aig/processed/train/data.pt').")
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device for training (e.g., cuda, cpu).')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save model checkpoints. Overrides internal construction if provided.')
    # --- ADDED ARGUMENT FOR AUGMENTATIONS ---
    parser.add_argument('--num_augmentations', type=int, default=5,
                        help='Number of randomized topological sort augmentations per graph during training.')
    # --- END ADDED ARGUMENT ---

    # --- General Training Hyperparameters ---
    parser.add_argument('--lr', type=float, default=base_conf['lr'])
    parser.add_argument('--weight_decay', type=float, default=base_conf['weight_decay'])
    parser.add_argument('--batch_size', type=int, default=base_conf['batch_size'])
    parser.add_argument('--max_epochs', type=int, default=base_conf['max_epochs'])
    parser.add_argument('--save_interval', type=int, default=base_conf['save_interval'])
    parser.add_argument('--grad_clip_value', type=float, default=base_conf['grad_clip_value'],
                        help='Value for gradient norm clipping for the main optimizer. None or 0 to disable.')

    # --- GraphAF/GraphDF Specific Model Hyperparameters ---
    parser.add_argument('--edge_unroll', type=int, default=base_conf['model']['edge_unroll'])
    parser.add_argument('--num_flow_layer', type=int, default=base_conf['model']['num_flow_layer'])
    parser.add_argument('--num_rgcn_layer', type=int, default=base_conf['model']['num_rgcn_layer'])
    parser.add_argument('--gaf_nhid', type=int, default=base_conf['model']['nhid'], help="Hidden dim for GAF/GDF's RGCN/ST-nets.")
    parser.add_argument('--gaf_nout', type=int, default=base_conf['model']['nout'], help="Output dim for GAF/GDF's RGCN (embedding size).")
    parser.add_argument('--deq_coeff', type=float, default=base_conf['model']['deq_coeff'], help="Dequantization coefficient if used.")
    parser.add_argument('--st_type', type=str, default=base_conf['model']['st_type'],
                        choices=['exp', 'sigmoid', 'softplus'], help="Type of ST network for GAF/GDF.")

    # --- GraphEBM Model Hyperparameters ---
    parser.add_argument('--ebm_hidden', type=int, default=base_conf['model_ebm']['hidden'])
    parser.add_argument('--ebm_depth', type=int, default=base_conf['model_ebm']['depth'])
    parser.add_argument('--ebm_swish_act', action=argparse.BooleanOptionalAction, default=base_conf['model_ebm']['swish_act'])
    parser.add_argument('--ebm_add_self', action=argparse.BooleanOptionalAction, default=base_conf['model_ebm']['add_self'])
    parser.add_argument('--ebm_dropout', type=float, default=base_conf['model_ebm']['dropout'])
    parser.add_argument('--ebm_n_power_iterations', type=int, default=base_conf['model_ebm']['n_power_iterations'])

    # --- GraphEBM Training Hyperparameters ---
    parser.add_argument('--ebm_c', type=float, default=base_conf['train_ebm']['c'])
    parser.add_argument('--ebm_ld_step', type=int, default=base_conf['train_ebm']['ld_step'])
    parser.add_argument('--ebm_ld_noise', type=float, default=base_conf['train_ebm']['ld_noise'])
    parser.add_argument('--ebm_ld_step_size', type=float, default=base_conf['train_ebm']['ld_step_size'])
    parser.add_argument('--ebm_alpha', type=float, default=base_conf['train_ebm']['alpha'])
    parser.add_argument('--ebm_clamp_lgd_grad', action=argparse.BooleanOptionalAction,
                        default=base_conf['train_ebm']['clamp_lgd_grad'])

    args = parser.parse_args()
    main(args)