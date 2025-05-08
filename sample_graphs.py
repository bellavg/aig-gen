import os
import json
import argparse
import torch
import warnings
import os.path as osp
import pickle
import networkx as nx
from collections import OrderedDict

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

# --- Import AIG Config ---
# Need this for node/edge type keys used in conversion helper
try:
    import data.aig_config
except ImportError:
    import G2PT.configs.aig as aig_config

if aig_config:
    AIG_NODE_TYPE_KEYS = getattr(aig_config, 'NODE_TYPE_KEYS', ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"])
    AIG_EDGE_TYPE_KEYS = getattr(aig_config, 'EDGE_TYPE_KEYS', ["EDGE_REG", "EDGE_INV"])
else:
    AIG_NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
    AIG_EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]
# --- End AIG Config Import ---


# --- Base Model Configuration Dictionary (Defaults for model instantiation) ---
# These should ideally match the defaults used in training if not overridden by args
# Crucially, edge_unroll MUST match the trained model.
base_model_conf = {
    "max_size": 64,
    "node_dim": 4,
    "bond_dim": 3,
    "use_gpu": True, # Will be determined by args.device
    "edge_unroll": 12, # *** Default, MUST be overridden by args to match training ***
    "num_flow_layer": 12,
    "num_rgcn_layer": 3,
    "nhid": 128,
    "nout": 128,
    "deq_coeff": 0.9,
    "st_type": "exp",
    "use_df": False # Set specifically for GraphDF if needed
}
base_model_ebm_conf = {
    "hidden": 64, "depth": 2, "swish_act": True, "add_self": False,
    "dropout": 0.0, "n_power_iterations": 1
}
# --- End Base Configuration ---

# --- Helper function to convert raw generated data to NetworkX ---
# (Copied from GraphDF/graphdf.py - ensure consistency or import)
def _convert_raw_to_aig_digraph(node_features_one_hot, typed_edges_list,
                                num_actual_nodes, aig_node_type_strings, aig_edge_type_strings):
    """ Converts raw discrete model output to a NetworkX DiGraph for AIGs. """
    graph = nx.DiGraph()
    node_features = node_features_one_hot.cpu().detach()
    for i in range(num_actual_nodes):
        try:
            if node_features[i].sum() == 0:
                node_type_label = "UNKNOWN_NODE_TYPE"
                warnings.warn(f"Node {i} has all-zero feature vector during conversion.")
            else:
                node_type_idx = torch.argmax(node_features[i]).item()
                node_type_label = aig_node_type_strings[node_type_idx] if 0 <= node_type_idx < len(aig_node_type_strings) else "UNKNOWN_NODE_TYPE"
            graph.add_node(i, type=node_type_label)
        except Exception as e:
            warnings.warn(f"Error processing node {i} during conversion: {e}")
            return None

    for u, v, edge_type_idx in typed_edges_list:
        if u < num_actual_nodes and v < num_actual_nodes:
            if 0 <= edge_type_idx < len(aig_edge_type_strings):
                edge_type_label = aig_edge_type_strings[edge_type_idx]
                graph.add_edge(u, v, type=edge_type_label)
            else:
                warnings.warn(f"Edge ({u}->{v}) has unexpected edge_type_idx {edge_type_idx}. Adding untyped.")
                graph.add_edge(u, v)
    return graph
# --- End Helper ---


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
    use_gpu = (device.type == 'cuda')
    # --- End Device Setup ---

    # --- Prepare Model Configuration ---
    # Start with base config and update with critical args
    model_conf = base_model_conf.copy()
    model_conf['use_gpu'] = use_gpu
    # *** Crucial: Use edge_unroll from args ***
    model_conf['edge_unroll'] = args.edge_unroll
    # Update other relevant params if provided via args (optional, but good practice)
    model_conf['max_size'] = getattr(args, 'max_size', model_conf['max_size'])
    model_conf['node_dim'] = getattr(args, 'node_dim', model_conf['node_dim'])
    model_conf['bond_dim'] = getattr(args, 'bond_dim', model_conf['bond_dim'])
    model_conf['num_flow_layer'] = getattr(args, 'num_flow_layer', model_conf['num_flow_layer'])
    model_conf['num_rgcn_layer'] = getattr(args, 'num_rgcn_layer', model_conf['num_rgcn_layer'])
    model_conf['nhid'] = getattr(args, 'gaf_nhid', model_conf['nhid'])
    model_conf['nout'] = getattr(args, 'gaf_nout', model_conf['nout'])
    if args.model == 'GraphDF': model_conf['use_df'] = True

    # Prepare EBM specific model config if needed
    model_ebm_conf = base_model_ebm_conf.copy()
    model_ebm_conf['hidden'] = getattr(args, 'ebm_hidden', model_ebm_conf['hidden'])
    model_ebm_conf['depth'] = getattr(args, 'ebm_depth', model_ebm_conf['depth'])
    model_ebm_conf['swish_act'] = getattr(args, 'ebm_swish_act', model_ebm_conf['swish_act'])
    model_ebm_conf['add_self'] = getattr(args, 'ebm_add_self', model_ebm_conf['add_self'])
    model_ebm_conf['dropout'] = getattr(args, 'ebm_dropout', model_ebm_conf['dropout'])
    model_ebm_conf['n_power_iterations'] = getattr(args, 'ebm_n_power_iterations', model_ebm_conf['n_power_iterations'])

    # --- Instantiate Model Runner ---
    print(f"Instantiating model runner for: {args.model}")
    runner = None
    if args.model == 'GraphDF':
        runner = GraphDF()
    elif args.model == 'GraphAF':
        runner = GraphAF()
    elif args.model == 'GraphEBM':
        try:
            # Pass combined config to EBM
            runner = GraphEBM(
                n_atom=model_conf['max_size'], n_atom_type=model_conf['node_dim'],
                n_edge_type=model_conf['bond_dim'], **model_ebm_conf, device=device
            )
        except Exception as e: print(f"Error instantiating GraphEBM: {e}"); exit(1)
    else: print(f"Error: Unknown model type '{args.model}'."); exit(1)
    if runner is None: print(f"Failed to instantiate runner for {args.model}"); exit(1)
    # --- End Model Instantiation ---

    print(f"Preparing to generate AIGs using {args.model}...")
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True); print(f"Created output directory: {output_dir}")

    # --- Prepare Generation Arguments ---
    # Arguments common to all runners' run_rand_gen (or similar method)
    generation_args = {
        "checkpoint_path": args.checkpoint,
        "output_pickle_path": args.output_file
    }

    # Add model-specific arguments based on the runner type
    if isinstance(runner, (GraphDF, GraphAF)):
        generation_args["model_conf_dict"] = model_conf # Pass the updated conf
        generation_args["num_samples"] = args.num_samples
        generation_args["num_min_nodes"] = args.min_nodes
        # Pass the correct temperature argument based on model type
        generation_args["temperature"] = args.temperature_df if args.model == 'GraphDF' else args.temperature_af
        generation_args["aig_node_type_strings"] = AIG_NODE_TYPE_KEYS
        # Add aig_edge_type_strings if _convert_raw_to_aig_digraph is used internally
        # generation_args["aig_edge_type_strings"] = AIG_EDGE_TYPE_KEYS
    elif isinstance(runner, GraphEBM):
        # Args specific to GraphEBM's run_rand_gen
        generation_args["n_samples"] = args.num_samples # Correct key name
        generation_args["c"] = args.ebm_c
        generation_args["ld_step"] = args.ebm_ld_step
        generation_args["ld_noise"] = args.ebm_ld_noise
        generation_args["ld_step_size"] = args.ebm_ld_step_size
        generation_args["clamp_lgd_grad"] = args.ebm_clamp_lgd_grad
        generation_args["num_min_nodes"] = args.min_nodes
        generation_args["aig_node_type_strings"] = AIG_NODE_TYPE_KEYS
        # Add aig_edge_type_strings if needed by EBM's conversion logic
        # generation_args["aig_edge_type_strings"] = AIG_EDGE_TYPE_KEYS

    # --- Run Generation ---
    try:
        print(f"Calling run_rand_gen for {args.model}...")
        # print(f"Arguments: {generation_args}") # Uncomment for debugging

        if not hasattr(runner, 'run_rand_gen'):
             raise NotImplementedError(f"{args.model} class does not have a 'run_rand_gen' method.")

        # ** Crucially, the runner's get_model method (called within run_rand_gen or explicitly before)
        # ** MUST use the model_conf (including the correct edge_unroll) passed here.
        # ** We assume GraphDF/GraphAF/GraphEBM's run_rand_gen handles model loading/instantiation correctly.
        generated_graphs = runner.run_rand_gen(**generation_args)

        print(f"\nGeneration finished for {args.model}.")
        if generated_graphs is not None and hasattr(generated_graphs, '__len__'):
            print(f"Number of graphs returned by run_rand_gen: {len(generated_graphs)}")
            # Verify the output file was actually created by run_rand_gen
            if not osp.exists(args.output_file):
                 print(f"Warning: Generation finished but output file '{args.output_file}' was not found.")
                 # Optionally save the returned graphs if the method didn't save them
                 # try:
                 #     with open(args.output_file, 'wb') as f: pickle.dump(generated_graphs, f)
                 #     print(f"Saved returned graphs to {args.output_file}")
                 # except Exception as save_e: print(f"Error saving returned graphs: {save_e}")
            else:
                 print(f"Generated graphs should be saved in: {args.output_file}")
        else:
            print("Generation method did not return a list of graphs or returned None.")

    except NotImplementedError as nie: print(f"\nError: Method not implemented: {nie}")
    except FileNotFoundError as fnf: print(f"\nError: Checkpoint file not found: {fnf}"); print(f"Looked for: {args.checkpoint}")
    except TypeError as te:
        print(f"\nTypeError during generation with {args.model}. Check argument names/types for run_rand_gen.")
        print(f"Arguments passed: {list(generation_args.keys())}")
        print(f"Error Details: {te}")
        import traceback; traceback.print_exc()
    except Exception as e:
        print(f"\nAn unexpected error occurred during generation with {args.model}: {type(e).__name__} - {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AIG graphs using trained models.")
    parser.add_argument('--model', type=str, required=True, choices=['GraphDF', 'GraphAF', 'GraphEBM'], help='Model type.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of AIG samples.')
    parser.add_argument('--output_file', type=str, default="generated_aigs.pkl", help='Output pickle file path.')
    parser.add_argument('--min_nodes', type=int, default=5, help='Minimum number of actual nodes.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device for generation.')

    # --- Model Architecture Args (Crucial for loading checkpoint correctly) ---
    parser.add_argument('--edge_unroll', type=int, default=base_model_conf['edge_unroll'],
                        help='Max look-back distance used during TRAINING.') # Clarify help text
    # Add others if needed, although checkpoint should ideally contain enough info
    parser.add_argument('--max_size', type=int, default=base_model_conf['max_size'])
    parser.add_argument('--node_dim', type=int, default=base_model_conf['node_dim'])
    parser.add_argument('--bond_dim', type=int, default=base_model_conf['bond_dim'])
    parser.add_argument('--num_flow_layer', type=int, default=base_model_conf['num_flow_layer'])
    parser.add_argument('--num_rgcn_layer', type=int, default=base_model_conf['num_rgcn_layer'])
    parser.add_argument('--gaf_nhid', type=int, default=base_model_conf['nhid'])
    parser.add_argument('--gaf_nout', type=int, default=base_model_conf['nout'])
    # Add EBM architecture args if they aren't saved/loaded via checkpoint
    parser.add_argument('--ebm_hidden', type=int, default=base_model_ebm_conf['hidden'])
    parser.add_argument('--ebm_depth', type=int, default=base_model_ebm_conf['depth'])
    # ... add other EBM architecture params if needed ...


    # --- Model Specific Generation Args ---
    parser.add_argument('--temperature_df', type=float, nargs=2, default=[0.7, 0.7], help='Temp [node, edge] (GraphDF).') # Updated default
    parser.add_argument('--temperature_af', type=float, default=0.75, help='Temperature (GraphAF).')
    parser.add_argument('--ebm_c', type=float, default=0.0, help='Dequantization scaling (GraphEBM).') # Use EBM defaults
    parser.add_argument('--ebm_ld_step', type=int, default=150, help='Langevin steps (GraphEBM).')
    parser.add_argument('--ebm_ld_noise', type=float, default=0.005, help='Langevin noise (GraphEBM).')
    parser.add_argument('--ebm_ld_step_size', type=float, default=30, help='Langevin step size (GraphEBM).')
    parser.add_argument('--ebm_clamp_lgd_grad', action=argparse.BooleanOptionalAction, default=True, help='Clamp LGD grad (GraphEBM).')

    args = parser.parse_args()
    main(args)
