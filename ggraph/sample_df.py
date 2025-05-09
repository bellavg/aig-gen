# sample_graphs.py
import os
import sys
import argparse
import torch
import pickle
import warnings

# --- Attempt to import GraphDF and its AIG configuration ---
# This section helps Python find your GraphDF module.
# Option 1: If sample_graphs.py is in the parent directory of GraphDF
# (e.g., MyProject/sample_graphs.py and MyProject/GraphDF/)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_of_script = os.path.dirname(script_dir) # If GraphDF is a sibling of the script's dir
graphdf_potential_path = os.path.join(script_dir, 'GraphDF') # If GraphDF is a sub-dir of script's dir

# Add the directory *containing* the GraphDF package to sys.path
# This allows `from GraphDF...` imports
# Modify this if your structure is different.
# Example: If GraphDF is in the same dir as sample_graphs.py, use script_dir
# If GraphDF is a sub-directory of where sample_graphs.py is, this is fine.
# If GraphDF is a sibling directory, use parent_dir_of_script.

# For this example, let's assume GraphDF is a subdirectory or accessible via PYTHONPATH
# If your 'GraphDF' folder is a direct subfolder of where you place 'sample_graphs.py',
# you might need to add 'script_dir' to sys.path if 'GraphDF' is not a package itself.
# However, 'from GraphDF.graphdf...' implies 'GraphDF' is a package.
# Safest: ensure the directory *containing* 'GraphDF' folder is in PYTHONPATH.

try:
    from GraphDF.graphdf import GraphDF
    # Attempt to import the aig_config that graphdf.py would use
    import GraphDF.aig_config as aig_config_module
except ImportError as e:
    print(f"Error importing GraphDF or its aig_config: {e}")
    print("Ensure 'sample_graphs.py' can locate the 'GraphDF' directory and its contents.")
    print("This typically means the directory *containing* 'GraphDF' should be in your PYTHONPATH,")
    print("or 'GraphDF' should be structured as an installable package.")
    print("Searched for 'from GraphDF.graphdf import GraphDF' and 'import GraphDF.aig_config as aig_config_module'")
    sys.exit(1)
except Exception as e_general:
    print(f"An unexpected error occurred during initial imports: {e_general}")
    sys.exit(1)


def main(args):
    # --- Device Setup ---
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    else:
        device = torch.device('cpu')
        print("Using CPU.")
    use_gpu = (device.type == 'cuda')

    # --- Model Configuration ---
    # These parameters MUST match the training configuration of the checkpoint.
    # Values from aig_config are good defaults for AIG specific dimensions.
    try:
        node_dim = len(aig_config_module.NODE_TYPE_KEYS)
        # bond_dim for GraphFlowModel includes REG, INV, and NO_EDGE category
        bond_dim = len(aig_config_module.EDGE_TYPE_KEYS) + 1
        # Default max_size to what's in aig_config if not overridden by user
        max_size_default = aig_config_module.MAX_NODE_COUNT
    except AttributeError as ae:
        print(f"Error accessing attributes from aig_config_module (e.g., NODE_TYPE_KEYS): {ae}")
        print("Please ensure 'GraphDF/aig_config.py' is correctly formatted and accessible.")
        sys.exit(1)

    max_size = args.max_size if args.max_size is not None else max_size_default
    if args.max_size is None:
        print(f"Using default max_size from aig_config: {max_size}")


    model_conf = {
        "max_size": max_size,
        "node_dim": node_dim,
        "bond_dim": bond_dim,
        "use_gpu": use_gpu,
        "edge_unroll": args.edge_unroll,      # CRITICAL: Must match training
        "num_flow_layer": args.num_flow_layer,
        "num_rgcn_layer": args.num_rgcn_layer,
        "nhid": args.nhid,
        "nout": args.nout,
        "use_df": True # This flag might be used internally if GraphFlowModel handles both DF/AF
    }
    print("\nUsing Model Configuration:")
    for key, value in model_conf.items():
        print(f"  {key}: {value}")
    print("-" * 30)

    # --- Instantiate GraphDF Runner ---
    print(f"Instantiating GraphDF runner...")
    try:
        runner = GraphDF()
    except Exception as e_init:
        print(f"Error during GraphDF runner instantiation: {e_init}")
        sys.exit(1)

    # --- Prepare Generation Arguments ---
    generation_args = {
        "model_conf_dict": model_conf,
        "checkpoint_path": args.checkpoint,
        "num_samples": args.num_samples,
        "num_min_nodes": args.min_nodes,
        "temperature": [args.temperature_node, args.temperature_edge],
        "aig_node_type_strings": aig_config_module.NODE_TYPE_KEYS,
        "output_pickle_path": args.output_file
    }

    print(f"\nStarting AIG generation with GraphDF...")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        generated_graphs = runner.run_rand_gen(**generation_args)

        print(f"\nGeneration finished for GraphDF.")
        if generated_graphs is not None:
            print(f"Number of graphs returned by run_rand_gen: {len(generated_graphs)}")
            # run_rand_gen is expected to save the file itself.
            if os.path.exists(args.output_file):
                 print(f"Generated graphs saved in: {args.output_file}")
            else:
                 print(f"Warning: Generation method finished, but output file '{args.output_file}' was not found.")
                 print("The run_rand_gen method in GraphDF is responsible for saving the pickle file.")
                 if generated_graphs: # If it returned graphs but didn't save
                     warnings.warn("Attempting to manually save returned graphs as run_rand_gen might not have.")
                     try:
                         with open(args.output_file, 'wb') as f:
                             pickle.dump(generated_graphs, f)
                         print(f"Manually saved {len(generated_graphs)} graphs to {args.output_file}")
                     except Exception as save_e:
                         print(f"Error manually saving returned graphs: {save_e}")

        else:
            print("GraphDF's run_rand_gen method returned None.")
            if os.path.exists(args.output_file):
                 print(f"However, an output file was found at: {args.output_file}. Please check its contents.")


    except FileNotFoundError as fnf:
        print(f"\nERROR: Checkpoint file not found: {fnf}")
        print(f"Looked for: {args.checkpoint}")
    except NotImplementedError as nie:
        print(f"\nERROR: A method is not implemented, likely in GraphFlowModel or DisGraphAF: {nie}")
    except RuntimeError as rte:
        print(f"\nERROR: RuntimeError during generation: {rte}")
        print("This often indicates a mismatch between the model configuration (e.g., max_size, edge_unroll, dimensions)")
        print("and the loaded checkpoint, or an issue during model execution (e.g., on the wrong device, CUDA errors).")
        import traceback
        traceback.print_exc()
    except TypeError as te:
        print(f"\nERROR: TypeError during generation. This could be an issue with arguments to run_rand_gen or internal methods.")
        print(f"Error Details: {te}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred during generation with GraphDF: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample AIG graphs using a trained GraphDF model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Essential Arguments ---
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained GraphDF model checkpoint (.pth file).')
    parser.add_argument('--output_file', type=str, default="graphdf_generated_aigs.pkl",
                        help='Path to save the generated AIGs (Pickle file).')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of AIG samples to generate.')

    # --- Generation Control Arguments ---
    parser.add_argument('--min_nodes', type=int, default=5,
                        help='Minimum number of actual nodes in generated AIGs.')
    parser.add_argument('--temperature_node', type=float, default=0.7,
                        help='Temperature for node type sampling in GraphDF.')
    parser.add_argument('--temperature_edge', type=float, default=0.7,
                        help='Temperature for edge type sampling in GraphDF.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device for generation ("cuda" or "cpu").')

    # --- Model Architectural Arguments (MUST match training config of the checkpoint) ---
    # These defaults are based on common values or those hinted in your GraphFlowModel.
    # Adjust them if your trained model used different settings.
    parser.add_argument('--max_size', type=int, default=64,
                        help='Maximum number of nodes in the graph (e.g., 64 for AIGs). '\
                             'Defaults to MAX_NODE_COUNT from aig_config. MUST match training.')
    parser.add_argument('--edge_unroll', type=int, default=25,
                        help='Max look-back distance for edge generation (edge_unroll). CRITICAL: MUST match training.')
    parser.add_argument('--num_flow_layer', type=int, default=12,
                        help='Number of flow layers in DisGraphAF. MUST match training.')
    parser.add_argument('--num_rgcn_layer', type=int, default=3,
                        help='Number of layers in RGCN. MUST match training.')
    parser.add_argument('--nhid', type=int, default=128,
                        help='Hidden dimension size in RGCN/ST_Dis. MUST match training.')
    parser.add_argument('--nout', type=int, default=128,
                        help='Output dimension size (embedding size) from RGCN. MUST match training.')

    args = parser.parse_args()
    main(args)