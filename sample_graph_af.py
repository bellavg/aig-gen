# sample_graphs_af.py
import os
import sys
import argparse
import torch
import pickle
import warnings

# --- Attempt to import GraphAF and its AIG configuration ---
# This section helps Python find your GraphAF module.
# Ensure the directory *containing* the 'GraphAF' folder is in your PYTHONPATH,
# or GraphAF is structured as an installable package.
try:
    from GraphAF.graphaf import GraphAF
    # Attempt to import the aig_config that graphaf.py would use.
    # graphaf.py itself has logic to find aig_config (e.g., from GraphAF/aig_config.py)
    # For the sampling script, we'll try to import it assuming a similar structure.
    # If GraphAF is a package, and aig_config.py is inside it:
    import GraphAF.aig_config as aig_config_module
except ImportError as e:
    print(f"Error importing GraphAF or its aig_config: {e}")
    print("Ensure 'sample_graphs_af.py' can locate the 'GraphAF' directory and its contents.")
    print("This typically means the directory *containing* 'GraphAF' should be in your PYTHONPATH,")
    print("or 'GraphAF' should be structured as an installable package.")
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
    try:
        node_dim = len(aig_config_module.NODE_TYPE_KEYS) # Should be 4 for AIGs
        # For GraphAF (continuous flow) with AIGs, bond_dim typically includes
        # channels for REG, INV, and an explicit NO_EDGE category for the model.
        # The GraphAF/model/graphflow.py's generate_aig_raw_data implies this.
        bond_dim = len(aig_config_module.EDGE_TYPE_KEYS) + 1 # e.g., 2 (REG,INV) + 1 (NO_EDGE) = 3
        max_size_default = aig_config_module.MAX_NODE_COUNT
    except AttributeError as ae:
        print(f"Error accessing attributes from aig_config_module (e.g., NODE_TYPE_KEYS): {ae}")
        print("Please ensure 'GraphAF/aig_config.py' is correctly formatted and accessible.")
        sys.exit(1)

    max_size = args.max_size if args.max_size is not None else max_size_default
    if args.max_size is None:
        print(f"Using default max_size from aig_config: {max_size}")

    model_conf = {
        "max_size": max_size,
        "node_dim": node_dim,
        "bond_dim": bond_dim, # Used by MaskedGraphAF internally
        "use_gpu": use_gpu,
        "edge_unroll": args.edge_unroll,      # CRITICAL: Must match training
        "num_flow_layer": args.num_flow_layer,
        "num_rgcn_layer": args.num_rgcn_layer,
        "nhid": args.nhid,
        "nout": args.nout,
        "st_type": args.st_type,              # Specific to GraphAF (continuous flow)
        "deq_coeff": args.deq_coeff,          # Specific to GraphAF (continuous flow)
        # "use_df": False # Implicitly False as this is for GraphAF
    }
    print("\nUsing Model Configuration for GraphAF:")
    for key, value in model_conf.items():
        print(f"  {key}: {value}")
    print("-" * 30)

    # --- Instantiate GraphAF Runner ---
    print(f"Instantiating GraphAF runner...")
    try:
        runner = GraphAF()
    except Exception as e_init:
        print(f"Error during GraphAF runner instantiation: {e_init}")
        sys.exit(1)

    # --- Prepare Generation Arguments ---
    # These arguments should match the signature of GraphAF.run_rand_gen
    generation_args = {
        "model_conf_dict": model_conf,
        "checkpoint_path": args.checkpoint,
        "num_samples": args.num_samples,
        "num_min_nodes": args.min_nodes,
        "temperature": args.temperature, # Single temperature for GraphAF
        # GraphAF's run_rand_gen uses AIG_NODE_TYPE_KEYS from its own imported aig_config by default.
        # Pass it if you want to ensure override, otherwise it will use its internal default.
        "aig_node_type_strings_override": aig_config_module.NODE_TYPE_KEYS,
        "output_pickle_path": args.output_file
    }

    print(f"\nStarting AIG generation with GraphAF...")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        generated_graphs = runner.run_rand_gen(**generation_args)

        print(f"\nGeneration finished for GraphAF.")
        if generated_graphs is not None:
            print(f"Number of graphs returned by run_rand_gen: {len(generated_graphs)}")
            if os.path.exists(args.output_file):
                 print(f"Generated graphs saved in: {args.output_file}")
            else:
                 print(f"Warning: Generation method finished, but output file '{args.output_file}' was not found.")
                 print("The run_rand_gen method in GraphAF is responsible for saving the pickle file.")
                 if generated_graphs: # If it returned graphs but didn't save
                     warnings.warn("Attempting to manually save returned graphs as run_rand_gen might not have.")
                     try:
                         with open(args.output_file, 'wb') as f:
                             pickle.dump(generated_graphs, f)
                         print(f"Manually saved {len(generated_graphs)} graphs to {args.output_file}")
                     except Exception as save_e:
                         print(f"Error manually saving returned graphs: {save_e}")
        else:
            print("GraphAF's run_rand_gen method returned None.")
            if os.path.exists(args.output_file):
                 print(f"However, an output file was found at: {args.output_file}. Please check its contents.")

    except FileNotFoundError as fnf:
        print(f"\nERROR: Checkpoint file not found: {fnf}")
        print(f"Looked for: {args.checkpoint}")
    except NotImplementedError as nie:
        print(f"\nERROR: A method is not implemented, likely in GraphFlowModel or MaskedGraphAF: {nie}")
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
        print(f"\nERROR: An unexpected error occurred during generation with GraphAF: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample AIG graphs using a trained GraphAF model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Essential Arguments ---
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained GraphAF model checkpoint (.pth file).')
    parser.add_argument('--output_file', type=str, default="graphaf_generated_aigs.pkl",
                        help='Path to save the generated AIGs (Pickle file).')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of AIG samples to generate.')

    # --- Generation Control Arguments ---
    parser.add_argument('--min_nodes', type=int, default=5,
                        help='Minimum number of actual nodes in generated AIGs.')
    parser.add_argument('--temperature', type=float, default=0.75,
                        help='Temperature for sampling from the prior in GraphAF.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device for generation ("cuda" or "cpu").')

    # --- Model Architectural Arguments (MUST match training config of the checkpoint) ---
    parser.add_argument('--max_size', type=int, default=None,
                        help='Maximum number of nodes in the graph (e.g., 64 for AIGs). ' \
                             'Defaults to MAX_NODE_COUNT from aig_config. MUST match training.')
    parser.add_argument('--edge_unroll', type=int, default=12,
                        help='Max look-back distance for edge generation (edge_unroll). CRITICAL: MUST match training.')
    parser.add_argument('--num_flow_layer', type=int, default=12,
                        help='Number of flow layers in MaskedGraphAF. MUST match training.')
    parser.add_argument('--num_rgcn_layer', type=int, default=3,
                        help='Number of layers in RGCN. MUST match training.')
    parser.add_argument('--nhid', type=int, default=128,
                        help='Hidden dimension size in RGCN/ST-Nets. MUST match training.')
    parser.add_argument('--nout', type=int, default=128,
                        help='Output dimension size (embedding size) from RGCN. MUST match training.')
    parser.add_argument('--st_type', type=str, default='sigmoid', choices=['sigmoid', 'exp', 'softplus'],
                        help="Type of S and T network for MaskedGraphAF's affine coupling. MUST match training.")
    parser.add_argument('--deq_coeff', type=float, default=0.9,
                        help="Dequantization coefficient for GraphAF. MUST match training if dequantization was used.")


    args = parser.parse_args()
    main(args)
