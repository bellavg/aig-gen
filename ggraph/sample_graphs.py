# sample_graphs.py
import os
import sys
import argparse
import torch
import pickle
import warnings


# Assuming GraphDF and GraphAF are in sibling directories or properly installed
from GraphDF import GraphDF  # Or from ggraph.GraphDF import GraphDF
from GraphAF import GraphAF  # Or from ggraph.GraphAF import GraphAF
# Assuming GraphEBM is in a sibling directory or properly installed
# Adjust the import path based on your project structure.
# If GraphEBM is in ggraph/GraphEBM/graphebm.py, and sample_graphs.py is in ggraph/
from GraphEBM.graphebm import GraphEBM  # Corrected import path assuming graphebm.py contains GraphEBM class

from aig_config import base_conf, NUM_NODE_FEATURES, NUM_EXPLICIT_EDGE_TYPES  # Assuming aig_config.py is in the same directory or accessible
from evaluate_aigs import run_standalone_evaluation


def main(args):
    # --- Device Setup ---
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    elif args.device == 'cuda' and torch.cuda.is_available():  # Check availability for cuda
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU.")
    # use_gpu = (device.type == 'cuda') # This variable is not used later

    print("\nUsing Model Configuration from aig_config.base_conf:")
    # Print general model config, not the whole base_conf if it's large
    if "model" in base_conf:
        for key, value in base_conf["model"].items():
            print(f"  {key}: {value}")
    else:
        print("  (Could not find 'model' key in base_config from aig_config.py)")
    print("-" * 30)

    # --- Instantiate Runner and Prepare Generation Arguments ---
    runner = None
    generation_args = {}

    if args.model == 'GraphDF':
        print(f"Instantiating GraphDF runner...")
        runner = GraphDF()  # Assuming GraphDF can be instantiated without args or uses a default config
        generation_args = {
            "model_conf_dict": base_conf.get("model", {}),  # Pass the model sub-config
            "checkpoint_path": args.checkpoint,
            "n_mols": args.num_samples,
            "num_min_node": args.min_nodes,  # GraphDF might use this
            "temperature": [args.temperature_node, args.temperature_edge]
        }
    elif args.model == 'GraphAF':
        print(f"Instantiating GraphAF runner...")
        runner = GraphAF()  # Assuming GraphAF can be instantiated without args
        generation_args = {
            "model_conf_dict": base_conf.get("model", {}),  # Pass the model sub-config
            "checkpoint_path": args.checkpoint,
            "n_mols": args.num_samples,
            "num_min_node": args.min_nodes,  # GraphAF might use this
            "temperature": args.temperature_af
        }
    elif args.model == 'GraphEBM':
        print(f"Instantiating GraphEBM runner...")
        # GraphEBM(n_atom, n_atom_type_actual, n_edge_type_actual, hidden, device=None)
        # Get necessary parameters from base_conf (aig_config.py)
        model_config = base_conf.get("model", {})
        if not model_config:
            raise ValueError("Missing 'model' configuration in base_conf from aig_config.py for GraphEBM.")

        runner = GraphEBM(
            n_atom=model_config.get("max_node_count", args.max_size),  # MAX_NODE_COUNT
            n_atom_type_actual=NUM_NODE_FEATURES,  # NUM_NODE_FEATURES
            n_edge_type_actual=NUM_EXPLICIT_EDGE_TYPES,  # NUM_EXPLICIT_EDGE_TYPES
            hidden=model_config.get("hidden", args.nhid),  # Hidden dim for EnergyFunc
            device=device
        )
        # generation_args for GraphEBM's run_rand_gen:
        # model_conf_dict (for Langevin dynamics), checkpoint_path, n_samples
        generation_args = {
            "model_conf_dict": model_config,  # This contains c, ld_step etc.
            "checkpoint_path": args.checkpoint,
            "n_samples": args.num_samples,
            # atomic_num_list is handled by gen_mol_from_one_shot_tensor in aig_generate_py_v2
            # num_min_node is not directly used by GraphEBM.run_rand_gen
        }
    else:
        raise ValueError(f"Unsupported model type: {args.model}. Choose from 'GraphDF', 'GraphAF', 'GraphEBM'.")

    print(f"\nStarting AIG generation with {args.model}...")
    # The run_rand_gen method in GraphEBM is expected to call gen_mol_from_one_shot_tensor,
    # which in turn uses the functions from aig_generate_py_v2 (construct_aig, correct_aig, valid_aig_can_with_seg)
    generated_graphs, pure_valids = runner.run_rand_gen(**generation_args)
    print(f"\nGeneration finished for {args.model}.")
    print(f"Generated graphs: {len(generated_graphs)}")
    print(f"Pure valids: {pure_valids}/1000")

    if generated_graphs is not None and len(generated_graphs) > 0:
        print(f"Successfully generated {len(generated_graphs)} graphs.")
        if args.evaluate:
            print("Starting evaluation...")
            run_standalone_evaluation(generated_graphs, results_filename=args.model + "_results.txt")
            print("Evaluation finished.")

        if args.save_graphs:  # Changed from args.save to args.save_graphs to match argparser
            output_filename = args.model + args.output_file
            output_path = os.path.join(args.output_dir, output_filename)
            if args.output_dir:  # Ensure output_dir is specified
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir, exist_ok=True)
                    print(f"Created output directory: {args.output_dir}")
            else:  # Default to current directory if output_dir is empty
                output_path = output_filename
                print(f"Warning: --output_dir not specified. Saving to current directory: {os.getcwd()}")

            with open(output_path, 'wb') as f:
                pickle.dump(generated_graphs, f)
            print(f"Saved {len(generated_graphs)} graphs to {output_path}")
    elif generated_graphs is not None and len(generated_graphs) == 0:
        print("Generation resulted in 0 graphs.")
    else:
        print("Generation failed or returned None.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample AIG graphs using a trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Essential Arguments ---
    parser.add_argument('--model', type=str, default='GraphDF', choices=['GraphDF', 'GraphAF', 'GraphEBM'],
                        help='Model type to use for generation.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint (.pth file).')
    parser.add_argument('--output_dir', type=str, default="./ggraph/data/generated_graphs",
                        help='Directory to save the generated AIGs (Pickle file). Empty means current directory.')
    parser.add_argument('--output_file', type=str, default="_aigs.pkl",
                        help='Filename for the generated AIGs (Pickle file). Will be prefixed by model name.')
    parser.add_argument('--num_samples', type=int, default=1000,  # Reduced default for quicker testing
                        help='Number of AIG samples to generate.')

    # Saving and Evaluating
    parser.add_argument('--save_graphs', action='store_true', help="Save generated graphs to a pickle file.")
    parser.add_argument('--evaluate', action='store_true', help="Run standalone evaluation on generated graphs.")

    # --- Generation Control Arguments ---
    parser.add_argument('--min_nodes', type=int, default=5,
                        help='Minimum number of actual nodes in generated AIGs (Note: GraphEBM may not use this directly).')
    # Temperature args for GraphDF/GraphAF
    parser.add_argument('--temperature_node', type=float, default=0.3,
                        help='Temperature for node type sampling in GraphDF.')
    parser.add_argument('--temperature_edge', type=float, default=0.3,
                        help='Temperature for edge type sampling in GraphDF.')
    parser.add_argument('--temperature_af', type=float, default=0.6,
                        help='Temperature for sampling in GraphAF.')

    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device for generation ("cuda" or "cpu").')

    # --- Model Architectural Arguments (MUST match training config of the checkpoint if not loaded from config) ---
    # These are more relevant if the model structure isn't fully defined by config files loaded by the model itself.
    # For GraphEBM, these are primarily derived from aig_config.py.
    # For GraphDF/AF, these might be needed if their __init__ requires them and doesn't get them from base_conf.
    parser.add_argument('--max_size', type=int, default=64,  # From your original args
                        help='Maximum number of nodes in the graph (e.g., MAX_NODE_COUNT from aig_config). ' \
                             'Ensure this matches training if model depends on it at init.')
    parser.add_argument('--edge_unroll', type=int, default=25,
                        help='Max look-back distance for edge generation (GraphAF/DF specific).')
    parser.add_argument('--num_flow_layer', type=int, default=12,
                        help='Number of flow layers (GraphAF/DF specific).')
    parser.add_argument('--num_rgcn_layer', type=int, default=3,
                        help='Number of layers in RGCN (GraphAF/DF specific).')
    parser.add_argument('--nhid', type=int, default=128,
                        help='Hidden dimension size. For GraphEBM, this can be used for EnergyFunc hidden dim.')
    parser.add_argument('--nout', type=int, default=128,
                        help='Output dimension size (GraphAF/DF specific).')

    args = parser.parse_args()

    # Update base_conf with any architectural args from command line if needed
    # This allows overriding aig_config.py for certain parameters if the model's __init__
    # doesn't already take them from a loaded config.
    # For GraphEBM, it primarily uses values from base_conf["model"] for its own __init__.
    if "model" not in base_conf: base_conf["model"] = {}  # Ensure model key exists
    base_conf["model"]["max_node_count"] = args.max_size  # Example: if GraphEBM needs this from base_conf
    # Add other relevant args to base_conf["model"] if your models expect them there.
    # However, GraphEBM's __init__ directly takes n_atom, n_atom_type_actual, etc.

    main(args)
