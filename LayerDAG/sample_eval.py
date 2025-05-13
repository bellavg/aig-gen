#!/usr/bin/env python3
import os
import torch
import pickle
import networkx as nx
from argparse import ArgumentParser
from pprint import pprint
from tqdm import tqdm
import math

# Assuming setup_utils.py is in PYTHONPATH or project root
from setup_utils import set_seed
# Assuming src.model is in PYTHONPATH or LayerDAG/src
from src.model import DiscreteDiffusion, EdgeDiscreteDiffusion, LayerDAG

# Assuming evaluate_aigs.py and aig_config.py are in the same directory or PYTHONPATH
# If evaluate_aigs is a module: from evaluate_aigs_module import run_standalone_evaluation
# If aig_config is a module: from aig_config_module import DECODING_NODE_TYPE_NX (and other constants if needed)
# For simplicity, assuming direct import if files are co-located or PYTHONPATH is set.
import evaluate_aigs


# from aig_config import DECODING_NODE_TYPE_NX # Not strictly needed if evaluate_aigs handles raw vectors

def convert_sample_to_nx(edge_index_tensor, node_features_tensor, conditional_info_tensor=None):
    """
    Converts a single sampled graph (from LayerDAG model output) to a NetworkX DiGraph.
    Node features (types) are stored as lists (from tensors) directly on nodes.
    The evaluate_aigs.py script is expected to decode these raw feature vectors.

    Args:
        edge_index_tensor (torch.Tensor): Tensor of shape [2, num_edges] for edge connections.
                                          Row 0: destination nodes, Row 1: source nodes.
        node_features_tensor (torch.Tensor): Tensor of shape [num_nodes, num_node_features]
                                             representing one-hot encoded node types.
        conditional_info_tensor (torch.Tensor, optional): Conditional information for the graph.

    Returns:
        nx.DiGraph: The constructed NetworkX directed graph.
    """
    G = nx.DiGraph()

    # Add nodes with their one-hot encoded features (as lists)
    num_nodes = node_features_tensor.shape[0]
    for i in range(num_nodes):
        # Store the raw feature vector; evaluate_aigs.py's get_node_type_str_from_attrs can decode it.
        G.add_node(i, type=node_features_tensor[i].cpu().tolist())

    # Add edges
    if edge_index_tensor.numel() > 0:  # Check if there are any edges
        # LayerDAG edge_index: dim 0 is dst, dim 1 is src.
        # NetworkX add_edge expects (source, destination)
        sources = edge_index_tensor[1].cpu().tolist()
        destinations = edge_index_tensor[0].cpu().tolist()
        for src, dst in zip(sources, destinations):
            G.add_edge(src, dst)  # AIGs typically have untyped edges from this model's perspective

    # Optionally, add conditional info as a graph-level attribute
    if conditional_info_tensor is not None:
        G.graph['condition'] = conditional_info_tensor.cpu().tolist()

    return G


def sample_graphs_aig(args, device, model):
    """
    Samples a specified number of AIGs using the trained LayerDAG model.

    Args:
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): The device to run sampling on (e.g., 'cuda', 'cpu').
        model (LayerDAG): The trained LayerDAG model.

    Returns:
        list: A list of generated nx.DiGraph objects.
    """
    generated_nx_graphs = []
    num_batches = math.ceil(args.num_samples / args.batch_size)

    print(f"Starting AIG sampling for {args.num_samples} graphs in {num_batches} batches...")
    for i in tqdm(range(num_batches), desc="Sampling Batches"):
        current_batch_size = min(args.batch_size, args.num_samples - len(generated_nx_graphs))
        if current_batch_size <= 0:
            break

        # For unconditional AIG sampling, conditions (raw_y_batch) are typically None.
        # Adjust if your LayerDAG model's sample method expects a different format for unconditional.
        batch_edge_indices, batch_node_features, batch_conditions = model.sample(
            device=device,
            num_samples=current_batch_size,
            raw_y_batch=None,  # Assuming unconditional sampling for AIGs
            min_num_steps_n=args.min_num_steps_n,
            max_num_steps_n=args.max_num_steps_n,
            min_num_steps_e=args.min_num_steps_e,
            max_num_steps_e=args.max_num_steps_e
        )

        for j in range(len(batch_edge_indices)):
            edge_index_j = batch_edge_indices[j]
            node_features_j = batch_node_features[j]
            condition_j = batch_conditions[j] if batch_conditions and j < len(batch_conditions) else None

            nx_graph = convert_sample_to_nx(edge_index_j, node_features_j, condition_j)
            generated_nx_graphs.append(nx_graph)

    print(f"Successfully generated {len(generated_nx_graphs)} AIGs.")
    return generated_nx_graphs


def main(args):
    torch.set_num_threads(args.num_threads)
    set_seed(args.seed)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Using device: {device}")

    print(f"Loading model checkpoint from: {args.model_path}")
    try:
        ckpt = torch.load(args.model_path, map_location=device)
    except FileNotFoundError:
        print(f"ERROR: Model checkpoint file not found at {args.model_path}")
        return
    except Exception as e:
        print(f"ERROR: Failed to load model checkpoint: {e}")
        return

    # Verify that the model was trained for AIGs
    ckpt_dataset = ckpt.get('dataset', 'unknown')
    print(f"Checkpoint dataset type: {ckpt_dataset}")
    if ckpt_dataset != 'aig':
        print(f"WARNING: Model checkpoint is for dataset '{ckpt_dataset}', but this script is for 'aig'.")
        # Decide if you want to exit or proceed with caution
        # exit(1)

    # Instantiate diffusion processes and model from checkpoint configurations
    try:
        node_diffusion_config = ckpt['node_diffusion_config']
        edge_diffusion_config = ckpt['edge_diffusion_config']
        model_specific_config = ckpt['model_config']  # This is LayerDAG's own config

        node_diffusion = DiscreteDiffusion(**node_diffusion_config)
        edge_diffusion = EdgeDiscreteDiffusion(**edge_diffusion_config)

        model = LayerDAG(device=device,  # Pass device here
                         node_diffusion=node_diffusion,
                         edge_diffusion=edge_diffusion,
                         **model_specific_config)  # Pass other model params

        print("Model Configuration from checkpoint:")
        pprint(model_specific_config)

        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)  # Ensure model is on the correct device
        model.eval()
        print("Model loaded and set to evaluation mode.")
    except KeyError as e:
        print(f"ERROR: Missing key in checkpoint: {e}. Checkpoint structure might be different.")
        return
    except Exception as e:
        print(f"ERROR: Failed to instantiate model from checkpoint: {e}")
        return

    # Sample AIGs
    generated_aigs_nx = sample_graphs_aig(args, device, model)

    if not generated_aigs_nx:
        print("No AIGs were generated. Exiting.")
        return

    # Save generated AIGs to a pickle file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print(f"Saving {len(generated_aigs_nx)} generated AIGs to: {args.output_path}")
    try:
        with open(args.output_path, 'wb') as f:
            pickle.dump(generated_aigs_nx, f)
        print("Generated AIGs saved successfully.")
    except Exception as e:
        print(f"ERROR: Failed to save generated AIGs: {e}")
        return

    # Evaluate the generated AIGs using evaluate_aigs.py
    print("\nStarting evaluation of generated AIGs...")
    try:
        evaluate_aigs.run_standalone_evaluation(
            input_source=args.output_path,  # Path to the .pkl file we just saved
            results_filename=args.eval_results_file,
            train_pkl_dir=args.train_pkl_dir,
            train_pkl_prefix=args.train_pkl_prefix,
            num_train_pkl_files=args.num_train_pkl_files
        )
        print("Evaluation complete.")
    except AttributeError:
        print("ERROR: Could not find 'run_standalone_evaluation' in 'evaluate_aigs'.")
        print("Ensure evaluate_aigs.py is in the correct location and does not have import errors.")
    except Exception as e:
        print(f"ERROR: An error occurred during AIG evaluation: {e}")


if __name__ == '__main__':
    parser = ArgumentParser(description="Sample AIGs using a trained LayerDAG model and evaluate them.")

    # Model and Sampling Arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained LayerDAG model checkpoint (.pth).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the generated AIGs (as a .pkl file of NetworkX graphs).")
    parser.add_argument("--num_samples", type=int, default=1000, help="Total number of AIGs to sample.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for sampling.")  # Adjusted default
    parser.add_argument("--num_threads", type=int, default=4, help="Number of CPU threads for PyTorch.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")

    # Diffusion Step Arguments (optional, defaults usually come from model training)
    parser.add_argument("--min_num_steps_n", type=int, default=None, help="Min diffusion steps for nodes.")
    parser.add_argument("--max_num_steps_n", type=int, default=None, help="Max diffusion steps for nodes.")
    parser.add_argument("--min_num_steps_e", type=int, default=None, help="Min diffusion steps for edges.")
    parser.add_argument("--max_num_steps_e", type=int, default=None, help="Max diffusion steps for edges.")

    # Evaluation Arguments (for evaluate_aigs.py)
    parser.add_argument("--eval_results_file", type=str, default="sampled_aig_evaluation_results.txt",
                        help="Filename to save the evaluation summary text from evaluate_aigs.py.")
    parser.add_argument('--train_pkl_dir', type=str, default=None,  # Default to None if not calculating novelty
                        help='(Optional) Path to the directory containing training PKL files for novelty calculation.')
    parser.add_argument('--train_pkl_prefix', type=str, default="real_aigs_part_",
                        help='(Optional) Prefix of the training PKL files.')
    parser.add_argument('--num_train_pkl_files', type=int, default=0,  # Default to 0 if not calculating novelty
                        help='(Optional) Number of training PKL files to load for novelty.')

    args = parser.parse_args()
    main(args)
