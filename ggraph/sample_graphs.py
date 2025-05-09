# sample_graphs.py
import os
import sys
import argparse
import torch
import pickle
import warnings
from GraphDF import GraphDF
from GraphAF import GraphAF
from aig_config import *
from evaluate_aigs import run_standalone_evaluation


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

    print("\nUsing Model Configuration:")
    for key, value in base_conf.items():
        print(f"  {key}: {value}")
    print("-" * 30)

    # --- Prepare Generation Arguments ---
    generation_args = {
        "model_conf_dict": base_conf["model"],
        "checkpoint_path": args.checkpoint,
        "n_mols": args.num_samples,
        "num_min_node": args.min_nodes,
    }

    # --- Instantiate GraphDF Runner ---
    print(f"Instantiating GraphDF runner...")
    if args.model == 'GraphDF':
        runner = GraphDF()
        generation_args["temperature"]: [args.temperature_node, args.temperature_edge]
    elif args.model == 'GraphAF':
        runner = GraphAF()
        generation_args["temperature"]: args.temperature_af


    print(f"\nStarting AIG generation with {args.model}...")
    generated_graphs = runner.run_rand_gen(**generation_args)
    print(f"\nGeneration finished for GraphDF.")
    if generated_graphs is not None and len(generated_graphs) > 0:
        if args.evaluate:
            run_standalone_evaluation(generated_graphs, results_filename=args.model+"results.txt")

        if args.save:
            output_filename = args.model+args.output_file
            output_path = os.path.join(args.output_dir, output_filename)
            if args.output_dir and not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)
                print(f"Created output directory: {args.output_dir}")
                with open(output_path, 'wb') as f:
                    pickle.dump(generated_graphs, f)
                print(f"Manually saved {len(generated_graphs)} graphs to {args.output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample AIG graphs using a trained GraphDF model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Essential Arguments ---
    parser.add_argument('--model', type=str, default='GraphDF',
                        help='model type')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained GraphDF model checkpoint (.pth file).')
    parser.add_argument('--output_dir', type=str, default="./ggraph/data/generated_graphs",
                        help='Path to save the generated AIGs (Pickle file).')
    parser.add_argument('--output_file', type=str, default="_aigs.pkl",
                        help='Path to save the generated AIGs (Pickle file).')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of AIG samples to generate.')

    # Saving and Evaluating
    parser.add_argument('--save_graphs', action='store_true')
    parser.add_argument('--evaluate', action='store_true')

    # --- Generation Control Arguments ---
    parser.add_argument('--min_nodes', type=int, default=5,
                        help='Minimum number of actual nodes in generated AIGs.')
    parser.add_argument('--temperature_node', type=float, default=0.3,
                        help='Temperature for node type sampling in GraphDF.')
    parser.add_argument('--temperature_edge', type=float, default=0.3,
                        help='Temperature for edge type sampling in GraphDF.')
    parser.add_argument('--temperature_af', type=float, default=0.6,
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