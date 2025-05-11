import torch
import torch.nn as nn
import networkx as nx
from tqdm import tqdm
import wandb # If you use wandb
import numpy as np # For the conversion functions

# DiGress existing utilities
from src.analysis.spectre_utils import SpectreSamplingMetrics, degree_stats, \
    eval_fraction_isomorphic, eval_fraction_unique_non_isomorphic_valid

# --- AIG-specific imports - YOU NEED TO ENSURE THESE PATHS ARE CORRECT ---
# From your aig_config.py (ensure this file is accessible in your Python path)
# Example: if aig_config.py is in src/

from src.aig_config import check_validity as aig_check_validity # Renamed to avoid potential conflicts
from src.aig_config import NODE_TYPE_KEYS, EDGE_TYPE_KEYS, \
                           NUM_NODE_FEATURES, NUM_EDGE_FEATURES


# From your aig_custom_dataset.py or a shared utility file where it's defined
# This function converts PyG Data objects to NetworkX DiGraph for AIGs
# Example: if aig_custom_dataset.py is in src/datasets/
from src.datasets.aig_custom_dataset import convert_pyg_to_nx_for_aig_validation



def convert_raw_model_output_to_nx_aig(node_features_tensor: torch.Tensor,
                                       edge_index_tensor: torch.Tensor,
                                       edge_features_tensor: torch.Tensor,
                                       num_nodes_int: int) -> nx.DiGraph | None:
    """
    Converts raw model output tensors (node features, edge index, edge features, num_nodes)
    to a NetworkX DiGraph, applying AIG-specific type decoding.

    Args:
        node_features_tensor: Tensor of shape (max_nodes_in_batch, NUM_NODE_FEATURES)
        edge_index_tensor: Tensor of shape (2, num_edges)
        edge_features_tensor: Tensor of shape (num_edges, NUM_EDGE_FEATURES)
        num_nodes_int: The actual number of nodes for this graph.

    Returns:
        A NetworkX DiGraph with 'type' attributes on nodes and edges, or None if conversion fails.
    """
    nx_graph = nx.DiGraph()

    # Validate inputs carefully
    if not (0 <= num_nodes_int <= node_features_tensor.shape[0]):
        # print(f"Error converting Model Output: num_nodes_int ({num_nodes_int}) is out of bounds "
        #       f"for node_features_tensor shape ({node_features_tensor.shape[0]}).")
        return None
    if node_features_tensor.shape[1] != NUM_NODE_FEATURES:
        # print(f"Error converting Model Output: Node features tensor has incorrect feature dimension "
        #       f"({node_features_tensor.shape[1]} vs expected {NUM_NODE_FEATURES}).")
        return None

    # Process nodes up to num_nodes_int
    for i in range(num_nodes_int):
        node_feature_vector = node_features_tensor[i].cpu().numpy()
        # Robust one-hot check and decoding (ensure this matches your aig_config logic)
        if not (np.isclose(np.sum(node_feature_vector), 1.0) and
                np.all((np.isclose(node_feature_vector, 0.0)) | (np.isclose(node_feature_vector, 1.0)))):
            node_type_str = "UNKNOWN_TYPE_ATTRIBUTE" # Consistent with your validation
        else:
            type_index = np.argmax(node_feature_vector)
            if not (0 <= type_index < len(NODE_TYPE_KEYS)): # Check bounds for NODE_TYPE_KEYS
                node_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
            else:
                node_type_str = NODE_TYPE_KEYS[type_index]
        nx_graph.add_node(i, type=node_type_str)

    # Process edges
    num_edges = edge_index_tensor.shape[1]
    if num_edges > 0:
        if edge_features_tensor.shape[0] != num_edges or edge_features_tensor.shape[1] != NUM_EDGE_FEATURES:
            # print(f"Error converting Model Output: Edge features tensor has incorrect shape.")
            return None

        for i in range(num_edges):
            src_node = edge_index_tensor[0, i].item()
            tgt_node = edge_index_tensor[1, i].item()

            # Ensure edge indices are valid for the current graph size
            if not (0 <= src_node < num_nodes_int and 0 <= tgt_node < num_nodes_int):
                # print(f"Warning: Model generated edge ({src_node}, {tgt_node}) with out-of-bounds node index "
                #       f"for num_nodes {num_nodes_int}. Skipping edge.")
                continue

            edge_feature_vector = edge_features_tensor[i].cpu().numpy()
            if not (np.isclose(np.sum(edge_feature_vector), 1.0) and
                    np.all((np.isclose(edge_feature_vector, 0.0)) | (np.isclose(edge_feature_vector, 1.0)))):
                edge_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
            else:
                type_index = np.argmax(edge_feature_vector)
                if not (0 <= type_index < len(EDGE_TYPE_KEYS)): # Check bounds for EDGE_TYPE_KEYS
                    edge_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
                else:
                    edge_type_str = EDGE_TYPE_KEYS[type_index]
            nx_graph.add_edge(src_node, tgt_node, type=edge_type_str)

    # Optionally, add graph-level attributes if your model predicts them
    # e.g., nx_graph.graph['name'] = "generated_aig_" + str(np.random.randint(10000))
    return nx_graph


class AIGSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        # metrics_list: Defines which metrics from the forward pass will be computed and logged.
        # Add 'aig_structural_validity', 'aig_acyclicity'.
        # You can also include 'degree' if you adapt it for in/out degrees,
        # or other DiGress metrics if they are relevant and adapted for directed graphs.
        super().__init__(datamodule=datamodule,
                         compute_emd=False, # Typically False unless using MMD with EMD kernel for some metric
                         metrics_list=['aig_structural_validity', 'aig_acyclicity', 'degree'])
        self.local_rank = datamodule.cfg.general.local_rank # Store local_rank

    # Override loader_to_nx to use your AIG-specific PyG to NX conversion for reference graphs
    def loader_to_nx(self, loader):
        networkx_graphs = []
        if self.local_rank == 0:
            print(f"AIGSamplingMetrics: Loading reference graphs using 'convert_pyg_to_nx_for_aig_validation'...")
        for i, batch in enumerate(tqdm(loader, desc="Loading Ref Graphs")):
            data_list = batch.to_data_list()
            for j, pyg_data in enumerate(data_list):
                # Use your existing function from aig_custom_dataset.py
                nx_graph = convert_pyg_to_nx_for_aig_validation(pyg_data)
                if nx_graph is not None:
                    networkx_graphs.append(nx_graph)
                # else:
                #     if self.local_rank == 0:
                #         print(f"Warning: Reference graph (batch {i}, item {j}) failed PyG to NX conversion.")
        if self.local_rank == 0:
            print(f"AIGSamplingMetrics: Loaded {len(networkx_graphs)} reference graphs for this split.")
        return networkx_graphs

    def forward(self, generated_graphs_raw: list, name: str, current_epoch: int, val_counter: int, local_rank: int, test: bool = False):
        """
        Calculates metrics for generated AIGs.
        `generated_graphs_raw`: A list of tuples from the sampling process, typically:
                                (node_features, edge_index, edge_features, num_nodes_tensor)
                                where node_features are (max_n_nodes, n_node_feat),
                                edge_index is (2, n_edges),
                                edge_features are (n_edges, n_edge_feat).
        """
        # self.test_graphs and self.val_graphs are loaded by the overridden loader_to_nx
        reference_graphs_nx = self.test_graphs if test else self.val_graphs

        if local_rank == 0: # Use the passed local_rank for printing
            print(f"AIGSamplingMetrics (Rank {local_rank}): Computing metrics for {len(generated_graphs_raw)} generated AIGs "
                  f"against {len(reference_graphs_nx)} reference AIGs (Split: {'test' if test else 'val'}).")

        generated_graphs_nx = []
        if local_rank == 0:
            print("AIGSamplingMetrics: Converting raw model outputs to NetworkX DiGraphs...")

        for i, graph_raw_data in enumerate(tqdm(generated_graphs_raw, desc="Converting Generated Graphs", disable=(local_rank!=0))):
            node_features, edge_index, edge_features, num_nodes_tensor = graph_raw_data
            num_nodes_int = num_nodes_tensor.item() # Actual number of nodes for this graph

            # Important: Ensure tensors are sliced to actual num_nodes_int BEFORE conversion
            # This assumes node_features is padded up to max_nodes in batch.
            current_node_features = node_features[:num_nodes_int]

            # Edge index and features might also need filtering if nodes were removed
            # or if edges refer to nodes beyond num_nodes_int.
            # convert_raw_model_output_to_nx_aig should handle this internally.

            nx_graph = convert_raw_model_output_to_nx_aig(
                current_node_features, edge_index, edge_features, num_nodes_int
            )
            if nx_graph is not None:
                generated_graphs_nx.append(nx_graph)
            # else:
            #     if local_rank == 0:
            #         print(f"Warning: Generated graph {i} (num_nodes: {num_nodes_int}) failed raw to NX conversion.")


        if not generated_graphs_nx:
            if local_rank == 0:
                print("AIGSamplingMetrics: No generated graphs were successfully converted to NetworkX. Skipping further metrics.")
            return # Nothing to evaluate

        to_log = {}

        # 1. AIG Structural Validity (using your aig_check_validity function)
        if 'aig_structural_validity' in self.metrics_list:
            if local_rank == 0:
                print("AIGSamplingMetrics: Computing AIG structural validity...")
            num_structurally_valid = 0
            for g_idx, g in enumerate(generated_graphs_nx):
                if aig_check_validity(g): # Your function from aig_config.py
                    num_structurally_valid += 1
                # else:
                #     if local_rank == 0 and num_structurally_valid < 5: # Log first few failures
                #         print(f"Generated graph {g_idx} failed structural validity.")
            structural_validity_fraction = num_structurally_valid / len(generated_graphs_nx) if generated_graphs_nx else 0.0
            to_log['aig_metrics/structural_validity_fraction'] = structural_validity_fraction
            if wandb.run and local_rank == 0: # Log only from rank 0
                wandb.summary[f'{name}_aig_structural_validity_fraction'] = structural_validity_fraction

        # 2. Acyclicity
        if 'aig_acyclicity' in self.metrics_list:
            if local_rank == 0:
                print("AIGSamplingMetrics: Computing AIG acyclicity...")
            num_acyclic = 0
            for g in generated_graphs_nx:
                if nx.is_directed_acyclic_graph(g):
                    num_acyclic += 1
            acyclicity_fraction = num_acyclic / len(generated_graphs_nx) if generated_graphs_nx else 0.0
            to_log['aig_metrics/acyclicity_fraction'] = acyclicity_fraction
            if wandb.run and local_rank == 0:
                wandb.summary[f'{name}_aig_acyclicity_fraction'] = acyclicity_fraction

        # 3. Standard DiGress Metrics (e.g., degree)
        if 'degree' in self.metrics_list:
            if local_rank == 0:
                print("AIGSamplingMetrics: Computing degree stats...")
            # IMPORTANT: The standard 'degree_stats' is for undirected graphs and sums degrees.
            # For AIGs, you'll want separate in-degree and out-degree distributions.
            # You would need to create a 'directed_degree_stats' function.
            # As a placeholder, using the existing one:
            degree_mmd = degree_stats(reference_graphs_nx, generated_graphs_nx, is_parallel=True,
                                      compute_emd=self.compute_emd)
            to_log['graph_stats/degree_mmd_placeholder'] = degree_mmd # Mark as placeholder
            if wandb.run and local_rank == 0:
                wandb.summary[f'{name}_degree_mmd_placeholder'] = degree_mmd

        # 4. Uniqueness, Isomorphism, and Novelty using DiGress functions
        if local_rank == 0:
            print("AIGSamplingMetrics: Computing uniqueness, novelty, and combined validity fractions...")

        # This is the validity function that eval_fraction_unique_non_isomorphic_valid will use.
        def combined_aig_validity_for_eval_fractions(g_nx_eval):
            return aig_check_validity(g_nx_eval) # Your function

        # self.train_graphs are loaded by the overridden loader_to_nx
        train_graphs_nx = self.train_graphs # Should be list of AIG nx.DiGraphs

        # These functions require graphs to be non-empty for some internal checks.
        # Filter out any potentially empty graphs from generated_graphs_nx if necessary,
        # though your conversion and validity checks should ideally handle this.
        eval_generated_graphs_nx = [g for g in generated_graphs_nx if g.number_of_nodes() > 0]
        eval_train_graphs_nx = [g for g in train_graphs_nx if g.number_of_nodes() > 0]


        if not eval_generated_graphs_nx:
            if local_rank == 0:
                print("AIGSamplingMetrics: No non-empty generated graphs to evaluate for uniqueness/novelty.")
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = 0.0, 0.0, 0.0
            frac_non_iso_to_train = 0.0
        elif not eval_train_graphs_nx:
            if local_rank == 0:
                 print("AIGSamplingMetrics: No non-empty training graphs for novelty comparison. Novelty might be skewed.")
            # Calculate uniqueness based on generated graphs only
            # Validity for frac_unique_non_iso_valid will still be checked against generated.
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(
                    eval_generated_graphs_nx,
                    [], # Pass empty list for train_graphs if none are suitable
                    validity_func=combined_aig_validity_for_eval_fractions
                )
            frac_non_iso_to_train = 1.0 # All are non-isomorphic to an empty training set
        else:
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(
                    eval_generated_graphs_nx,
                    eval_train_graphs_nx,
                    validity_func=combined_aig_validity_for_eval_fractions
                )
            frac_non_iso_to_train = 1.0 - eval_fraction_isomorphic(eval_generated_graphs_nx, eval_train_graphs_nx)


        to_log.update({
            'sampling_quality/frac_unique_aigs': frac_unique,
            'sampling_quality/frac_unique_non_iso_aigs': frac_unique_non_iso,
            'sampling_quality/frac_unique_non_iso_structurally_valid_aigs': frac_unique_non_iso_valid,
            'sampling_quality/frac_non_iso_to_train_aigs': frac_non_iso_to_train
        })

        if local_rank == 0:
            print(f"AIGSamplingMetrics (Rank {local_rank}): Final metrics to log for {name}: {to_log}")
        if wandb.run and local_rank == 0: # Log only from rank 0
            # Prefix with 'name' (e.g., 'val_sampling_quality/frac_unique_aigs') for clarity in wandb
            wandb.log({f"{name}_{k.replace('/', '_')}": v for k, v in to_log.items()}, commit=True)

    def reset(self):
        # This method is called by PyTorch Lightning.
        # If your metrics have internal states that need resetting between epochs/phases, do it here.
        # For the current implementation, most calculations are stateless per call to forward.
        pass
