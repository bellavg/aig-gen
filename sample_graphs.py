import os
import json
import argparse
import torch
import warnings
import os.path as osp
import pickle
import networkx as nx
from collections import OrderedDict  # Added for robust checkpoint loading

# --- Assume necessary model classes are importable ---
try:
    # Ensure these point to the updated classes
    from GraphDF import GraphDF
    from GraphAF import GraphAF
    # GraphEBM (Keep if used, ensure it's compatible)
    from GraphEBM import GraphEBM # Example import
except ImportError as e:
    print(f"Error importing base model classes (GraphDF, GraphAF): {e}")
    print("Please ensure GraphDF/GraphAF models are correctly placed and importable.")
    exit()
# --- End Imports ---


# --- Hardcoded Configuration Dictionary (Base for Model Config) ---
# Contains model architecture details needed for instantiation.
# Generation parameters will be taken from command-line args.
conf = {
    "data_name": "aig",  # Used mainly for context, less critical for generation
    "model": {
        # Common parameters potentially used by all models
        "max_size": 64,  # Max nodes model was trained with
        "node_dim": 4,  # AIG node types (CONST0, PI, AND, PO)
        "bond_dim": 3,  # AIG edge types (REG, INV, NO-EDGE channels for model)
        "use_gpu": True,  # Will be updated based on device arg

        # GraphDF/GraphAF specific parameters (needed for instantiation)
        "edge_unroll": 12,
        "num_flow_layer": 12,
        "num_rgcn_layer": 3,
        "nhid": 128,
        "nout": 128,
        "deq_coeff": 0.9,  # GraphAF
        "st_type": "exp",  # GraphAF (Example, ensure matches training)
        # "use_df": False      # GraphAF specific flag, might not be needed if model type implies it
    },
    "model_ebm": {  # GraphEBM specific model parameters (Example)
        "hidden": 64
    },
    # Training parameters below are NOT used for generation but kept for structure
    "lr": 0.001,
    "weight_decay": 0,
    "batch_size": 128,
    "max_epochs": 50,
    "save_interval": 5,
    "save_dir": "checkpoints/aig",  # Example save dir

    "train_ebm": {  # GraphEBM training params - some needed for generation (Example)
        "c": 0.0,
        "ld_step": 150,
        "ld_noise": 0.005,
        "ld_step_size": 30,
        "clamp": True,
        "alpha": 1.0
    },
}
# --- End Hardcoded Configuration ---

# --- Define AIG Node Types based on config ---
# Ensure this order matches your data processing and model training
# Attempt to import from config first
try:
    from G2PT.configs import aig_config  # Adjust import path if needed

    AIG_NODE_TYPE_KEYS = aig_config.NODE_TYPE_KEYS
    AIG_EDGE_TYPE_KEYS = aig_config.EDGE_TYPE_KEYS  # Needed for conversion helper
    print("Successfully imported AIG type keys from G2PT.configs.aig_config")
except ImportError:
    print("Warning: G2PT.configs.aig_config not found. Using default AIG type keys.")
    AIG_NODE_TYPE_KEYS = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO']  # Default
    AIG_EDGE_TYPE_KEYS = ['EDGE_REG', 'EDGE_INV']  # Default

if len(AIG_NODE_TYPE_KEYS) != conf['model']['node_dim']:
    raise ValueError(
        f"Mismatch between AIG_NODE_TYPE_KEYS length ({len(AIG_NODE_TYPE_KEYS)}) and conf['model']['node_dim'] ({conf['model']['node_dim']})")
if len(AIG_EDGE_TYPE_KEYS) != 2:  # Should be REG, INV
    print(f"Warning: Expected 2 AIG_EDGE_TYPE_KEYS (REG, INV), found {len(AIG_EDGE_TYPE_KEYS)}")


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
    # Update config dict based on actual device availability for model loading
    conf['model']['use_gpu'] = (device.type == 'cuda')
    # --- End Device Setup ---

    # --- Instantiate Model Runner ---
    print(f"Instantiating model runner for: {args.model}")
    runner = None
    if args.model == 'GraphDF':
        runner = GraphDF()
    elif args.model == 'GraphAF':
        runner = GraphAF()
    elif args.model == 'GraphEBM':
        # Ensure GraphEBM class and necessary config exist if using
        try:
            from GraphEBM import GraphEBM  # Example import
            runner = GraphEBM(
                n_atom=conf['model']['max_size'],
                n_atom_type=conf['model']['node_dim'],
                n_edge_type=conf['model']['bond_dim'],
                hidden=conf['model_ebm']['hidden'],
                device=device  # Pass the torch device object
            )
        except ImportError:
            print("Error: GraphEBM class not found or importable.")
            exit()
        except KeyError as e:
            print(f"Error: Missing required parameter {e} in config for GraphEBM initialization.")
            exit()
        except Exception as e:
            print(f"Error instantiating GraphEBM: {e}")
            exit()
    else:
        print(f"Error: Unknown model type '{args.model}'. Choose from 'GraphDF', 'GraphAF', 'GraphEBM'.")
        exit()

    if runner is None:
        print(f"Failed to instantiate model runner for {args.model}")
        exit()
    # --- End Model Instantiation ---

    # --- Checkpoint Loading ---
    # GraphDF/GraphAF load checkpoint within run_rand_gen via get_model
    # GraphEBM might need explicit loading here if its run_rand_gen doesn't handle it
    # (Assuming GraphEBM's run_rand_gen *does* handle checkpoint loading based on previous context)
    print(f"Checkpoint ({args.checkpoint}) will be loaded within {args.model}.run_rand_gen.")
    # --- End Checkpoint Loading ---

    print(f"Preparing to generate AIGs using {args.model}...")

    # --- Prepare Generation Arguments ---
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)  # Use exist_ok=True
        print(f"Created output directory: {output_dir}")

    # Base arguments common to GraphDF/GraphAF
    generation_args = {
        "model_conf_dict": conf['model'],  # Pass the relevant model config section
        "checkpoint_path": args.checkpoint,
        "num_samples": args.num_samples,
        "num_min_nodes": args.min_nodes,
        # Use the imported/default AIG node type strings
        # The key name must match the parameter name in run_rand_gen
        "aig_node_type_strings": AIG_NODE_TYPE_KEYS,  # CHANGED KEY NAME
        "output_pickle_path": args.output_file
    }

    # Add model-specific arguments
    if args.model == 'GraphDF':
        generation_args["temperature"] = args.temperature_df  # List [node, edge]
        # Ensure GraphDF's run_rand_gen expects 'aig_node_type_strings'
    elif args.model == 'GraphAF':
        generation_args["temperature"] = args.temperature_af  # Single float
        # Ensure GraphAF's run_rand_gen expects 'aig_node_type_strings'
        # REMOVED: "num_max_nodes_config" as it's not in the updated GraphAF.run_rand_gen signature
    elif args.model == 'GraphEBM':
        # --- Arguments specific to GraphEBM's run_rand_gen ---
        # Remove args not used by GraphEBM's run_rand_gen if necessary
        generation_args.pop("model_conf_dict", None)
        generation_args.pop("num_min_nodes", None)  # Filtering happens after generation
        generation_args.pop("aig_node_type_strings", None)  # EBM likely uses internal mapping

        # Add EBM specific args (ensure these match GraphEBM.run_rand_gen signature)
        generation_args["n_samples"] = generation_args.pop("num_samples")  # Rename if needed
        generation_args["checkpoint_path"] = args.checkpoint  # EBM run_rand_gen needs checkpoint explicitly
        generation_args["c"] = args.ebm_c
        generation_args["ld_step"] = args.ebm_ld_step
        generation_args["ld_noise"] = args.ebm_ld_noise
        generation_args["ld_step_size"] = args.ebm_ld_step_size
        generation_args["clamp"] = args.ebm_clamp
        generation_args["aig_node_types"] = AIG_NODE_TYPE_KEYS  # Pass node types if EBM needs them for conversion
        # Add any other specific args required by GraphEBM.run_rand_gen

    # --- Run Generation ---
    try:
        print(f"Calling run_rand_gen for {args.model} with args:")
        # Print args carefully, avoiding overly long dicts if needed
        # print(json.dumps(generation_args, indent=2, default=str)) # Use default=str for non-serializable items like lists

        generated_graphs = runner.run_rand_gen(**generation_args)

        print(f"\nGeneration finished for {args.model}.")
        if generated_graphs is not None and hasattr(generated_graphs, '__len__'):
            print(f"Number of graphs returned by run_rand_gen: {len(generated_graphs)}")
            # Note: The graphs should already be saved to the pickle file by run_rand_gen
        else:
            print("Generation method did not return a list of graphs or returned None.")

    except NotImplementedError as nie:
        print(f"\nError: The 'run_rand_gen' method (or an internal generation method it calls, "
              f"like 'generate_aig_discrete_raw_data' for GraphDF or 'generate_aig_raw_data' for GraphAF) "
              f"is not implemented correctly for AIGs in the {args.model} class or its internal model.")
        print(f"Specific Error: {nie}")
        print("Ensure the AIG-specific generation logic exists and method signatures match.")
    except FileNotFoundError as fnf:
        print(f"\nError: Checkpoint file not found.")
        print(f"Details: {fnf}")
        print(f"Looked for: {args.checkpoint}")
    except TypeError as te:
        print(f"\nTypeError during generation with {args.model}. This often indicates mismatched arguments.")
        print(f"Check if the arguments passed match the signature of {args.model}.run_rand_gen.")
        print(f"Arguments passed: {list(generation_args.keys())}")
        print(f"Error Details: {te}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\nAn unexpected error occurred during generation with {args.model}:")
        print(f"Error Type: {type(e)}")
        print(f"Error Details: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AIG graphs using trained models.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['GraphDF', 'GraphAF', 'GraphEBM'],
                        help='Model type to use for generation.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint (.pth or .pt file).')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of AIG samples to generate.')
    parser.add_argument('--output_file', type=str, default="generated_aigs.pkl",
                        help='Path to save the generated AIGs (list of nx.DiGraph) as a pickle file.')
    parser.add_argument('--min_nodes', type=int, default=5,
                        help='Minimum number of nodes for a generated graph to be kept (for GraphDF/GraphAF).')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device to use for generation.')

    # --- Model Specific Generation Args ---
    # GraphDF
    parser.add_argument('--temperature_df', type=float, nargs=2, default=[0.6, 0.6],  # Default for GraphDF
                        help='Temperature for discrete sampling [node_temp, edge_temp] (for GraphDF).')
    # GraphAF
    parser.add_argument('--temperature_af', type=float, default=0.75,  # Default for GraphAF
                        help='Temperature for sampling (for GraphAF).')
    # GraphEBM
    parser.add_argument('--ebm_c', type=float, default=conf['train_ebm']['c'],  # Default from training conf
                        help='Dequantization scaling factor (for GraphEBM).')
    parser.add_argument('--ebm_ld_step', type=int, default=conf['train_ebm']['ld_step'],
                        help='Langevin dynamics steps (for GraphEBM).')
    parser.add_argument('--ebm_ld_noise', type=float, default=conf['train_ebm']['ld_noise'],
                        help='Langevin dynamics noise level (for GraphEBM).')
    parser.add_argument('--ebm_ld_step_size', type=float, default=conf['train_ebm']['ld_step_size'],
                        # Allow float step size
                        help='Langevin dynamics step size (for GraphEBM).')
    parser.add_argument('--ebm_clamp', action=argparse.BooleanOptionalAction, default=conf['train_ebm']['clamp'],
                        help='Enable/disable gradient clamping during generation (for GraphEBM). Use --ebm_clamp or --no-ebm_clamp.')

    args = parser.parse_args()
    main(args)
