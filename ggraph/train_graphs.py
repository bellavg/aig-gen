#!/usr/bin/env python3
# train_graphs.py - Updated to use AIGPaddedInMemoryDataset and new config names
import argparse
import os
import os.path as osp
import traceback  # For error logging
import torch
from torch_geometric.loader import DenseDataLoader
import warnings  # Added for warnings

# Model and Dataset Imports
from GraphDF import GraphDF
# from GraphAF import GraphAF
# from GraphEBM import GraphEBM

# Assuming aig_padded_inmemory_dataset.py contains AIGPaddedInMemoryDataset
# and is in the Python path or same directory.
try:
    # Replace 'aig_padded_inmemory_dataset_module' with the actual file name if different.
    from aig_padded_inmemory_dataset import AIGPaddedInMemoryDataset
except ImportError:
    print("ERROR: Could not import AIGPaddedInMemoryDataset. "
          "Ensure 'aig_padded_inmemory_dataset.py' (from the artifact) is accessible.")
    exit(1)

# Configuration import from user's aig_config.py
try:
    from aig_config import *  # Imports all constants like MAX_NODE_COUNT, NUM_NODE_ATTRIBUTES, base_conf, etc.

    print("Successfully imported configuration from aig_config.py")
except ImportError:
    print("ERROR: Could not import from aig_config.py. Ensure it's in the Python path.")
    # Define placeholders if aig_config is missing, to allow script structure to be parsed,
    # but actual execution will likely fail.
    base_conf = {'lr': 0.001, 'weight_decay': 0.0, 'batch_size': 32, 'max_epochs': 100, 'save_interval': 1,
                 'grad_clip_value': 1.0,
                 'model': {'edge_unroll': 12, 'num_flow_layer': 12, 'num_rgcn_layer': 3, 'nhid': 128, 'nout': 128,
                           'hidden': 64, 'deq_coeff': 0.9},
                 'ebm_lr': 0.0001, 'ebm_bs': 64}  # Minimal placeholder
    MAX_NODE_COUNT = 64
    NUM_EXPLICIT_NODE_FEATURES = 4  # From your aig_config.py
    NUM_NODE_ATTRIBUTES = 5  # From your aig_config.py (NUM_EXPLICIT_NODE_FEATURES + 1)
    NUM_EXPLICIT_EDGE_FEATURES = 2  # From your aig_config.py
    NUM_EDGE_ATTRIBUTES = 3  # From your aig_config.py (NUM_EXPLICIT_EDGE_FEATURES + 1)
    # NUM_ADJ_CHANNELS might be NUM_EDGE_ATTRIBUTES or NUM_EXPLICIT_EDGE_FEATURES depending on model
    # For GraphEBM, it uses NUM_ADJ_CHANNELS from aig_config for its internal total edge channels.
    # Let's assume NUM_ADJ_CHANNELS is defined in aig_config.py and is what GraphEBM expects for its total.
    # If not, GraphEBM might need NUM_EDGE_ATTRIBUTES.
    if 'NUM_ADJ_CHANNELS' not in globals():  # Check if NUM_ADJ_CHANNELS was imported
        NUM_ADJ_CHANNELS = NUM_EDGE_ATTRIBUTES  # Fallback if not in aig_config
        warnings.warn(
            f"NUM_ADJ_CHANNELS not found in aig_config.py, defaulting to NUM_EDGE_ATTRIBUTES ({NUM_EDGE_ATTRIBUTES}) for GraphEBM.")
    else:
        # Ensure NUM_ADJ_CHANNELS from config is consistent with what GraphEBM might expect
        # (i.e., number of explicit edge types + virtual/no-edge if GraphEBM handles it that way)
        # GraphEBM's `n_edge_type_actual` is NUM_EXPLICIT_EDGE_FEATURES.
        # Its internal `self.model_total_edge_channels` is NUM_ADJ_CHANNELS from aig_config.
        # So, this should be fine if NUM_ADJ_CHANNELS in aig_config.py is indeed NUM_EXPLICIT_EDGE_FEATURES + 1.
        pass

    warnings.warn("Using placeholder configurations as aig_config.py was not found or fully imported.")

# wandb import
try:
    import wandb
except ImportError:
    wandb = None
    warnings.warn("wandb not installed. Wandb logging will be disabled. Run 'pip install wandb'")


def main(args):
    # Initialize conf from base_conf loaded from aig_config.py
    conf = {}
    for key, value in base_conf.items():
        if isinstance(value, dict):
            conf[key] = value.copy()
        else:
            conf[key] = value

    if 'model' not in conf:  # Ensure 'model' sub-dictionary exists
        conf['model'] = {}

    # --- Update Configuration from Arguments ---
    # General training params
    conf['lr'] = getattr(args, 'lr', conf.get('lr', 0.001))
    conf['weight_decay'] = getattr(args, 'weight_decay', conf.get('weight_decay', 0.0))
    conf['batch_size'] = getattr(args, 'batch_size', conf.get('batch_size', 32))
    conf['max_epochs'] = getattr(args, 'max_epochs', conf.get('max_epochs', 100))
    conf['save_interval'] = getattr(args, 'save_interval', conf.get('save_interval', 1))
    conf['grad_clip_value'] = getattr(args, 'grad_clip_value', conf.get('grad_clip_value', 1.0))

    # Model-specific params from base_conf['model'] and args
    model_sub_conf = conf.get('model', {})
    model_sub_conf['edge_unroll'] = getattr(args, 'edge_unroll', model_sub_conf.get('edge_unroll', 12))
    model_sub_conf['num_flow_layer'] = getattr(args, 'num_flow_layer', model_sub_conf.get('num_flow_layer', 12))
    model_sub_conf['num_rgcn_layer'] = getattr(args, 'num_rgcn_layer', model_sub_conf.get('num_rgcn_layer', 3))
    model_sub_conf['nhid'] = getattr(args, 'gaf_nhid', model_sub_conf.get('nhid', 128))  # GAF/GDF hidden
    model_sub_conf['nout'] = getattr(args, 'gaf_nout', model_sub_conf.get('nout', 128))  # GAF/GDF output
    model_sub_conf['hidden'] = getattr(args, 'ebm_hidden', model_sub_conf.get('hidden', 64))  # EBM hidden
    model_sub_conf['deq_coeff'] = getattr(args, 'deq_coeff', model_sub_conf.get('deq_coeff', 0.9))

    # Critical: Adjust node_dim and bond_dim for GraphDF/AF to match AIGPaddedInMemoryDataset output
    model_sub_conf['node_dim'] = NUM_NODE_ATTRIBUTES  # Includes padding type
    model_sub_conf['bond_dim'] = NUM_EDGE_ATTRIBUTES  # Includes no-edge type

    conf['model'] = model_sub_conf

    # --- Device Setup ---
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

    # --- Wandb Initialization ---
    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            print("Wandb is not installed, but --use_wandb was specified. Skipping wandb initialization.")
            args.use_wandb = False  # Disable wandb if not installed
        else:
            try:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name if args.wandb_run_name else args.model,
                    # Use specified run name or model type
                    entity=args.wandb_entity,  # Optional: your wandb entity (team/user)
                    config={**conf, **vars(args)}  # Log merged configurations
                )
                print(f"Wandb initialized for project '{args.wandb_project}', run '{wandb.run.name}'.")
            except Exception as e:
                print(f"Error initializing wandb: {e}. Disabling wandb.")
                args.use_wandb = False

    # --- Dataset Loading ---
    print(f"\nInitializing AIGPaddedInMemoryDataset.")
    print(f"  Dataset root (for its processed files): {args.data_root}")
    print(f"  Raw .pt input directory (unpadded data): {args.raw_pt_input_dir}")
    print(f"  Raw .pt input filename: {args.raw_pt_input_filename}")

    train_dataset = AIGPaddedInMemoryDataset(
        root=args.data_root,
        split="train",
        raw_pt_input_dir=args.raw_pt_input_dir,
        raw_pt_input_filename=args.raw_pt_input_filename,
        processed_file_prefix=args.processed_file_prefix
    )

    if len(train_dataset) == 0:
        if wandb_run: wandb.finish(exit_code=1)
        raise ValueError("Training dataset is empty after loading/processing. Check paths and input files.")

    train_loader = DenseDataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=True, drop_last=True)
    print(f"Created Training DataLoader with batch size {conf['batch_size']}.")
    if len(train_loader) > 0 and args.use_wandb and wandb_run:  # Log a sample batch shape if wandb is active
        print("Verifying sample batch data shapes for wandb logging context:")
        try:
            sample_batch = next(iter(train_loader))
            print(f"  Batch x shape: {sample_batch.x.shape} (Expected B, {MAX_NODE_COUNT}, {NUM_NODE_ATTRIBUTES})")
            print(
                f"  Batch adj shape: {sample_batch.adj.shape} (Expected B, {NUM_EDGE_ATTRIBUTES}, {MAX_NODE_COUNT}, {MAX_NODE_COUNT})")
            wandb.log({
                "sample_batch_x_shape": str(sample_batch.x.shape),
                "sample_batch_adj_shape": str(sample_batch.adj.shape)
            }, step=0)  # Log at step 0 or as an initial summary
        except Exception as e:
            print(f"Could not get sample batch for shape verification: {e}")

    # --- Model Instantiation ---
    print(f"Instantiating model runner: {args.model}")
    runner = None
    if args.model == 'GraphDF':
        runner = GraphDF()
    # elif args.model == 'GraphAF':
    #     runner = GraphAF()
    # elif args.model == 'GraphEBM':
    #     warnings.warn(
    #         "GraphEBM is selected. The AIGPaddedInMemoryDataset provides node features "
    #         f"with {NUM_NODE_ATTRIBUTES} dimensions (including padding type). "
    #         "The current GraphEBM implementation (graphebm.py) might expect {NUM_EXPLICIT_NODE_FEATURES} "
    #         "dimensions for its internal transformation '_transform_node_features_add_virtual_channel'. "
    #         "This could lead to a mismatch if GraphEBM.py is not adapted to handle pre-padded features "
    #         "or if the input to its train_rand_gen is not sliced appropriately."
    #     )
    #     runner = GraphEBM(
    #         n_atom=MAX_NODE_COUNT,
    #         n_atom_type_actual=NUM_EXPLICIT_NODE_FEATURES,
    #         n_edge_type_actual=NUM_EXPLICIT_EDGE_FEATURES,
    #         hidden=conf['model'].get('hidden', 64),
    #         device=device
    #     )
    #     conf['lr'] = base_conf.get('ebm_lr', conf['lr'])
    #     current_general_batch_size = conf['batch_size']  # Save before potential override
    #     conf['batch_size'] = base_conf.get('ebm_bs', conf['batch_size'])
    #     if conf['batch_size'] != current_general_batch_size:
    #         train_loader = DenseDataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=True, drop_last=True)
    #         print(f"Re-created DataLoader for GraphEBM with batch size {conf['batch_size']}.")
    else:
        if wandb_run: wandb.finish(exit_code=1)
        print(f"Error: Unknown model type '{args.model}'.");
        exit(1)

    if runner is None:
        if wandb_run: wandb.finish(exit_code=1)
        print(f"Failed to instantiate model runner for {args.model}");
        exit(1)

    # --- Save Directory Setup ---
    if args.save_dir:
        save_dir = args.save_dir
    else:
        default_save_dir_base = "./checkpoints_aig"
        model_specific_path = f"{args.model}"
        if wandb_run and wandb.run.name:  # Add wandb run name to path if available
            model_specific_path = f"{args.model}_{wandb.run.name}"
        save_dir = osp.join(default_save_dir_base, model_specific_path)

    save_dir = osp.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model checkpoints will be saved in: {save_dir}")
    if wandb_run: wandb.config.update({"checkpoint_save_dir": save_dir})

    # --- Training ---
    print(f"\n--- Starting Training ({args.model}) ---")
    try:
        # Pass wandb_active flag to the runner's training method
        # The runner's train_rand_gen method will need to be modified to accept and use this
        runner.train_rand_gen(
            loader=train_loader,
            lr=conf['lr'],
            wd=conf['weight_decay'],
            max_epochs=conf['max_epochs'],
            model_conf_dict=conf['model'],
            save_interval=conf['save_interval'],
            save_dir=save_dir,
            wandb_active=args.use_wandb  # Pass the flag
        )
        print(f"\n--- Training Finished ({args.model}) ---")
        if wandb_run:
            wandb.log({"training_completed": 1})  # Log successful completion
    except Exception as e:
        print(f"Error during training: {e}")
        if wandb_run:
            wandb.log({"training_error": str(e)})
            wandb.finish(exit_code=1)  # Mark run as failed
        traceback.print_exc()
        exit(1)
    finally:
        if wandb_run:
            wandb.finish()  # Ensure wandb run is finished


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AIG generation models with AIGPaddedInMemoryDataset and Wandb.")

    # --- Essential Arguments ---
    parser.add_argument('--model', type=str, default='GraphDF', choices=['GraphDF', 'GraphAF', 'GraphEBM'],
                        help='Model runner class to use.')
    parser.add_argument('--data_root', default="./data/aig_padded_processed_final",  # Changed default
                        help="Root directory for AIGPaddedInMemoryDataset to store its processed files.")
    parser.add_argument('--raw_pt_input_dir', default="./",
                        help="Directory containing the unpadded input .pt file (e.g., output of make_undirected_dataset.py).")
    parser.add_argument('--raw_pt_input_filename', default="aig_undirected.pt",
                        help="Filename of the unpadded input .pt file.")
    parser.add_argument('--processed_file_prefix', default="final_padded_aig_",  # Changed default
                        help="Prefix for the processed files created by AIGPaddedInMemoryDataset.")

    # --- Wandb Arguments ---
    parser.add_argument('--use_wandb', action='store_true', help="Enable Weights & Biases logging.")
    parser.add_argument('--wandb_project', type=str, default="ggraph", help="Wandb project name.")
    parser.add_argument('--wandb_entity', type=str, default=None, help="Wandb entity (username or team). Optional.")
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help="Custom name for the wandb run. Defaults to model type.")

    # --- Optional Overrides & Configuration ---
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='Device for training. Default: cuda if available, else cpu.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Override default save directory for checkpoints. If relative, relative to execution dir.')

    # --- Training Hyperparameters (defaults from aig_config.base_conf) ---
    parser.add_argument('--lr', type=float, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, help="Weight decay.")
    parser.add_argument('--batch_size', type=int, help="Batch size.")
    parser.add_argument('--max_epochs', type=int, help="Maximum training epochs.")
    parser.add_argument('--save_interval', type=int, help="Save checkpoints every N epochs.")
    parser.add_argument('--grad_clip_value', type=float, help="Gradient clipping value.")

    # --- Model Architecture Hyperparameters (defaults from aig_config.base_conf.model) ---
    parser.add_argument('--edge_unroll', type=int, help="Edge unroll value (GraphAF/DF).")
    parser.add_argument('--num_flow_layer', type=int, help="Number of flow layers (GraphAF/DF).")
    parser.add_argument('--num_rgcn_layer', type=int, help="Number of RGCN layers (GraphAF/DF).")
    parser.add_argument('--gaf_nhid', type=int, help="Hidden dim for GraphAF/DF RGCN.")
    parser.add_argument('--gaf_nout', type=int, help="Output dim for GraphAF/DF RGCN.")
    parser.add_argument('--ebm_hidden', type=int, help="Hidden dim for GraphEBM EnergyFunc.")
    parser.add_argument('--deq_coeff', type=float, help="Dequantization coefficient for GraphAF/DF.")

    args = parser.parse_args()

    # Set defaults for args that were not provided from CLI, using base_conf
    # This ensures base_conf is the primary source of defaults if not overridden
    # Ensure base_conf is loaded before this point (it should be by the global import)
    if 'base_conf' in globals():
        arg_defaults = {
            'lr': base_conf.get('lr'), 'weight_decay': base_conf.get('weight_decay'),
            'batch_size': base_conf.get('batch_size'), 'max_epochs': base_conf.get('max_epochs'),
            'save_interval': base_conf.get('save_interval'), 'grad_clip_value': base_conf.get('grad_clip_value'),
            'edge_unroll': base_conf.get('model', {}).get('edge_unroll'),
            'num_flow_layer': base_conf.get('model', {}).get('num_flow_layer'),
            'num_rgcn_layer': base_conf.get('model', {}).get('num_rgcn_layer'),
            'gaf_nhid': base_conf.get('model', {}).get('nhid'),
            'gaf_nout': base_conf.get('model', {}).get('nout'),
            'ebm_hidden': base_conf.get('model', {}).get('hidden'),
            'deq_coeff': base_conf.get('model', {}).get('deq_coeff')
        }
        for arg_name, default_val in arg_defaults.items():
            if getattr(args, arg_name, None) is None and default_val is not None:
                setattr(args, arg_name, default_val)

    # Special default for device, considering base_conf if args.device is not set
    if args.device is None:
        use_gpu_from_conf = base_conf.get('model', {}).get('use_gpu', True)
        args.device = 'cuda' if use_gpu_from_conf and torch.cuda.is_available() else 'cpu'

    main(args)
