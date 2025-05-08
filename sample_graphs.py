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
    from GraphDF import GraphDF
    from GraphAF import GraphAF
    from GraphEBM import GraphEBM # Ensure this is the one from graphebm_main_script_aig_gen
except ImportError as e:
    print(f"Error importing base model classes (GraphDF, GraphAF, GraphEBM): {e}")
    print("Please ensure models are correctly placed and importable.")
    exit(1)
# --- End Imports ---


# --- Hardcoded Configuration Dictionary (Base for Model Config) ---
# Contains model architecture details needed for instantiation.
# Generation parameters will be taken from command-line args.
conf = {
    "data_name": "aig",
    "model": {
        "max_size": 64,
        "node_dim": 4,
        "bond_dim": 3, # For EBM, this is n_edge_type (e.g., REG, INV, VIRTUAL)
        "use_gpu": True,

        # GraphDF/GraphAF specific parameters
        "edge_unroll": 12,
        "num_flow_layer": 12,
        "num_rgcn_layer": 3,
        "nhid": 128,
        "nout": 128,
        "deq_coeff": 0.9,
        "st_type": "exp",
    },
    "model_ebm": { # GraphEBM specific model parameters (defaults if not passed via args)
        "hidden": 64,
        "depth": 2,
        "swish_act": True,
        "add_self": False,
        "dropout": 0.0,
        "n_power_iterations": 1
    },
    # Training parameters below are NOT used for generation but kept for structure/defaults
    "train_ebm": {
        "c": 0.0,
        "ld_step": 150,
        "ld_noise": 0.005,
        "ld_step_size": 30,
        "clamp_lgd_grad": True, # Note: name changed here to match EBM class
        "alpha": 1.0
    },
}
# --- End Hardcoded Configuration ---

# --- Define AIG Node Types based on config ---
from data import aig_config as aig_config
AIG_NODE_TYPE_KEYS = aig_config.NODE_TYPE_KEYS
AIG_EDGE_TYPE_KEYS = aig_config.EDGE_TYPE_KEYS
print("Successfully imported AIG type keys from G2PT.configs.aig_config")


if len(AIG_NODE_TYPE_KEYS) != conf['model']['node_dim']:
    raise ValueError(f"Mismatch: AIG_NODE_TYPE_KEYS len ({len(AIG_NODE_TYPE_KEYS)}) vs node_dim ({conf['model']['node_dim']})")
if len(AIG_EDGE_TYPE_KEYS) != 2:
    print(f"Warning: Expected 2 AIG_EDGE_TYPE_KEYS (REG, INV), found {len(AIG_EDGE_TYPE_KEYS)}")


def main(args):
    # --- Setup Device ---
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
    # --- End Device Setup ---

    # --- Instantiate Model Runner ---
    print(f"Instantiating model runner for: {args.model}")
    runner = None
    if args.model == 'GraphDF':
        runner = GraphDF()
    elif args.model == 'GraphAF':
        runner = GraphAF()
    elif args.model == 'GraphEBM':
        try:
            # Use defaults from conf, but allow override if specific EBM model args were added to parser
            runner = GraphEBM(
                n_atom=conf['model']['max_size'],
                n_atom_type=conf['model']['node_dim'],
                n_edge_type=conf['model']['bond_dim'],
                hidden=getattr(args, 'ebm_hidden', conf['model_ebm']['hidden']), # Example: allow override
                depth=getattr(args, 'ebm_depth', conf['model_ebm']['depth']),
                swish_act=getattr(args, 'ebm_swish_act', conf['model_ebm']['swish_act']),
                add_self=getattr(args, 'ebm_add_self', conf['model_ebm']['add_self']),
                dropout=getattr(args, 'ebm_dropout', conf['model_ebm']['dropout']),
                n_power_iterations=getattr(args, 'ebm_n_power_iterations', conf['model_ebm']['n_power_iterations']),
                device=device
            )
        except ImportError: print("Error: GraphEBM class not found."); exit(1)
        except KeyError as e: print(f"Error: Missing param {e} for GraphEBM init."); exit(1)
        except Exception as e: print(f"Error instantiating GraphEBM: {e}"); exit(1)
    else:
        print(f"Error: Unknown model type '{args.model}'."); exit(1)

    if runner is None: print(f"Failed to instantiate runner for {args.model}"); exit(1)
    # --- End Model Instantiation ---

    print(f"Preparing to generate AIGs using {args.model}...")
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True); print(f"Created output directory: {output_dir}")

    # --- Prepare Generation Arguments ---
    generation_args = {
        "checkpoint_path": args.checkpoint,
        "output_pickle_path": args.output_file # Common arg
    }

    # Add model-specific arguments
    if args.model == 'GraphDF':
        generation_args["model_conf_dict"] = conf['model']
        generation_args["num_samples"] = args.num_samples
        generation_args["num_min_nodes"] = args.min_nodes
        generation_args["temperature"] = args.temperature_df
        generation_args["aig_node_type_strings"] = AIG_NODE_TYPE_KEYS
    elif args.model == 'GraphAF':
        generation_args["model_conf_dict"] = conf['model']
        generation_args["num_samples"] = args.num_samples
        generation_args["num_min_nodes"] = args.min_nodes
        generation_args["temperature"] = args.temperature_af
        generation_args["aig_node_type_strings"] = AIG_NODE_TYPE_KEYS
    elif args.model == 'GraphEBM':
        # Args specific to GraphEBM's run_rand_gen
        generation_args["n_samples"] = args.num_samples # Correct key name
        generation_args["c"] = args.ebm_c
        generation_args["ld_step"] = args.ebm_ld_step
        generation_args["ld_noise"] = args.ebm_ld_noise
        generation_args["ld_step_size"] = args.ebm_ld_step_size
        generation_args["clamp_lgd_grad"] = args.ebm_clamp_lgd_grad # Correct key name
        generation_args["num_min_nodes"] = args.min_nodes # Pass min_nodes
        generation_args["aig_node_type_strings"] = AIG_NODE_TYPE_KEYS # Pass node types

    # --- Run Generation ---
    try:
        print(f"Calling run_rand_gen for {args.model}...")
        # print(f"Arguments: {generation_args}") # Uncomment for debugging

        # Ensure the runner has the method
        if not hasattr(runner, 'run_rand_gen'):
             raise NotImplementedError(f"{args.model} class does not have a 'run_rand_gen' method.")

        generated_graphs = runner.run_rand_gen(**generation_args)

        print(f"\nGeneration finished for {args.model}.")
        if generated_graphs is not None and hasattr(generated_graphs, '__len__'):
            print(f"Number of graphs returned by run_rand_gen: {len(generated_graphs)}")
            # Note: The graphs should already be saved to the pickle file by run_rand_gen
        else:
            print("Generation method did not return a list of graphs or returned None.")

    except NotImplementedError as nie: print(f"\nError: Method not implemented: {nie}")
    except FileNotFoundError as fnf: print(f"\nError: Checkpoint file not found: {fnf}"); print(f"Looked for: {args.checkpoint}")
    except TypeError as te:
        print(f"\nTypeError during generation with {args.model}. Check argument names and types.")
        print(f"Arguments passed: {list(generation_args.keys())}")
        print(f"Expected signature likely mismatching for {args.model}.run_rand_gen.")
        print(f"Error Details: {te}")
        import traceback; traceback.print_exc()
    except Exception as e:
        print(f"\nAn unexpected error occurred during generation with {args.model}: {type(e).__name__} - {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AIG graphs using trained models.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['GraphDF', 'GraphAF', 'GraphEBM'],
                        help='Model type to use for generation.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint.')
    parser.add_argument('--num_samples', type=int, default=1000, # Changed default to 1000
                        help='Number of AIG samples to generate.')
    parser.add_argument('--output_file', type=str, default="generated_aigs.pkl",
                        help='Path to save the generated AIGs pickle file.')
    parser.add_argument('--min_nodes', type=int, default=5,
                        help='Minimum number of actual nodes for generated graphs.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device for generation.')

    # --- Model Specific Generation Args ---
    # GraphDF
    parser.add_argument('--temperature_df', type=float, nargs=2, default=[0.6, 0.6],
                        help='Temperature [node, edge] (for GraphDF).')
    # GraphAF
    parser.add_argument('--temperature_af', type=float, default=0.75,
                        help='Temperature (for GraphAF).')
    # GraphEBM
    parser.add_argument('--ebm_c', type=float, default=conf['train_ebm']['c'],
                        help='Dequantization scaling factor (for GraphEBM generation).')
    parser.add_argument('--ebm_ld_step', type=int, default=conf['train_ebm']['ld_step'],
                        help='Langevin dynamics steps (for GraphEBM generation).')
    parser.add_argument('--ebm_ld_noise', type=float, default=conf['train_ebm']['ld_noise'],
                        help='Langevin dynamics noise level (for GraphEBM generation).')
    parser.add_argument('--ebm_ld_step_size', type=float, default=conf['train_ebm']['ld_step_size'],
                        help='Langevin dynamics step size (for GraphEBM generation).')
    # Corrected argument name and help text to match GraphEBM.py
    parser.add_argument('--ebm_clamp_lgd_grad', action=argparse.BooleanOptionalAction,
                        default=conf['train_ebm']['clamp_lgd_grad'],
                        help='Enable/disable LGD gradient clamping (for GraphEBM generation).')
    # Add parser arguments for EBM model structure if needed for instantiation override
    parser.add_argument('--ebm_hidden', type=int, default=conf['model_ebm']['hidden'])
    parser.add_argument('--ebm_depth', type=int, default=conf['model_ebm']['depth'])
    parser.add_argument('--ebm_swish_act', action=argparse.BooleanOptionalAction, default=conf['model_ebm']['swish_act'])
    parser.add_argument('--ebm_add_self', action=argparse.BooleanOptionalAction, default=conf['model_ebm']['add_self'])
    parser.add_argument('--ebm_dropout', type=float, default=conf['model_ebm']['dropout'])
    parser.add_argument('--ebm_n_power_iterations', type=int, default=conf['model_ebm']['n_power_iterations'])


    args = parser.parse_args()
    main(args)
