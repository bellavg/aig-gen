#TODO check

import torch
import torch.nn as nn
import networkx as nx
from tqdm import tqdm
import wandb  # If you use wandb
import numpy as np  # For the conversion functions
from typing import Union, List
import warnings
# DiGress existing utilities
# Ensure these are correctly imported from your DiGress project structure

from src.analysis.spectre_utils import SpectreSamplingMetrics, degree_stats, \
    eval_fraction_isomorphic, eval_fraction_unique_non_isomorphic_valid

# --- AIG-specific imports - YOU NEED TO ENSURE THESE PATHS ARE CORRECT ---
from src.aig_config import check_validity as aig_check_validity
from src.aig_config import NODE_TYPE_KEYS, EDGE_TYPE_KEYS, \
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES  # NUM_EDGE_FEATURES = number of *actual* AIG types

# This should be the updated function from your aig_dataset_aligned_v2 artifact
from src.datasets.aig_custom_dataset import convert_pyg_to_nx_for_aig_validation


def convert_raw_model_output_to_nx_aig(node_features_tensor: torch.Tensor,
                                       edge_index_tensor: torch.Tensor,
                                       edge_features_tensor: torch.Tensor,
                                       # Model output, shape (num_edges, NUM_EDGE_FEATURES + 1)
                                       num_nodes_int: int) -> Union[nx.DiGraph, None]:
    """
    Converts raw model output tensors to a NetworkX DiGraph for AIGs.
    The model's edge_features_tensor is expected to have NUM_EDGE_FEATURES + 1 channels,
    where channel 0 is for "no specific AIG type" and channels 1 to NUM_EDGE_FEATURES
    (after shifting) correspond to actual AIG edge types.

    Args:
        node_features_tensor: Tensor of shape (max_nodes_in_batch, NUM_NODE_FEATURES)
        edge_index_tensor: Tensor of shape (2, num_edges_in_prediction)
        edge_features_tensor: Tensor of shape (num_edges_in_prediction, NUM_EDGE_FEATURES + 1)
        num_nodes_int: The actual number of nodes for this graph.

    Returns:
        A NetworkX DiGraph with 'type' attributes, or None if conversion fails.
    """
    nx_graph = nx.DiGraph()

    # Validate inputs
    if not (0 <= num_nodes_int <= node_features_tensor.shape[0]):
        # warnings.warn(f"Convert Model Output: num_nodes_int ({num_nodes_int}) out of bounds for node_features ({node_features_tensor.shape[0]}).")
        return None
    if node_features_tensor.shape[1] != NUM_NODE_FEATURES:
        # warnings.warn(f"Convert Model Output: Node features tensor has incorrect feature dim ({node_features_tensor.shape[1]} vs {NUM_NODE_FEATURES}).")
        return None

    # Process nodes up to num_nodes_int
    for i in range(num_nodes_int):
        node_feature_vector = node_features_tensor[i].cpu().numpy()
        if not (np.isclose(np.sum(node_feature_vector), 1.0) and
                np.all((np.isclose(node_feature_vector, 0.0)) | (np.isclose(node_feature_vector, 1.0)))):
            node_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
        else:
            type_index = np.argmax(node_feature_vector)
            if not (0 <= type_index < len(NODE_TYPE_KEYS)):
                node_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
            else:
                node_type_str = NODE_TYPE_KEYS[type_index]
        nx_graph.add_node(i, type=node_type_str)

    # Process edges
    num_edges_in_prediction = edge_index_tensor.shape[1]

    # The model's edge_features_tensor has NUM_EDGE_FEATURES + 1 channels
    expected_edge_feature_dim = NUM_EDGE_FEATURES + 1

    if num_edges_in_prediction > 0:
        if edge_features_tensor.shape[0] != num_edges_in_prediction or \
                edge_features_tensor.shape[1] != expected_edge_feature_dim:
            # warnings.warn(f"Convert Model Output: Edge features tensor shape mismatch. "
            #               f"Got {edge_features_tensor.shape}, expected ({num_edges_in_prediction}, {expected_edge_feature_dim}).")
            return None

        for i in range(num_edges_in_prediction):
            src_node = edge_index_tensor[0, i].item()
            tgt_node = edge_index_tensor[1, i].item()

            if not (0 <= src_node < num_nodes_int and 0 <= tgt_node < num_nodes_int):
                # warnings.warn(f"Convert Model Output: Edge ({src_node}, {tgt_node}) out of bounds for num_nodes {num_nodes_int}. Skipping edge.")
                continue

            edge_feature_vector_model = edge_features_tensor[i].cpu().numpy()  # Length NUM_EDGE_FEATURES + 1

            # Check if one-hot (model output might be logits/probabilities before argmax in sampling)
            # For simplicity, we assume it's been made one-hot or take argmax directly.
            # If it's already one-hot from the model's final layer:
            if not (np.isclose(np.sum(edge_feature_vector_model), 1.0) and
                    np.all(
                        (np.isclose(edge_feature_vector_model, 0.0)) | (np.isclose(edge_feature_vector_model, 1.0)))):
                # If not one-hot, but should be (e.g. after softmax), this indicates an issue or need for argmax.
                # For robust decoding, let's assume we always take argmax if not perfectly one-hot.
                # However, if the model is *supposed* to output one-hot, this is a problem.
                # For now, we'll proceed as if it's one-hot or argmax has been applied.
                # If it's truly just probabilities, the argmax below is the correct interpretation.
                pass  # Allow non-perfect one-hot if it's probabilities

            shifted_type_index = np.argmax(edge_feature_vector_model)

            if shifted_type_index == 0:
                # Model predicted "no specific AIG type" for this existing edge.
                # This could mean the edge itself is low probability or it's a generic edge.
                # For AIG validation, we need a specific type.
                edge_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
                # warnings.warn(f"Convert Model Output: Edge ({src_node}-{tgt_node}) was predicted as 'no specific AIG type' (index 0). Marked UNKNOWN.")
            else:
                actual_aig_type_index = shifted_type_index - 1  # Convert back to 0-indexed for actual AIG types
                if not (0 <= actual_aig_type_index < len(EDGE_TYPE_KEYS)):
                    edge_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
                    # warnings.warn(f"Convert Model Output: Edge ({src_node}-{tgt_node}) decoded to invalid actual_aig_type_index {actual_aig_type_index}.")
                else:
                    edge_type_str = EDGE_TYPE_KEYS[actual_aig_type_index]
            nx_graph.add_edge(src_node, tgt_node, type=edge_type_str)
    return nx_graph


class AIGSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['aig_structural_validity', 'aig_acyclicity', 'degree'])
        self.local_rank = datamodule.cfg.general.local_rank  # Store local_rank

        # Ensure reference graphs are loaded using the correct conversion
        # The super().__init__ might call loader_to_nx if SpectreSamplingMetrics does.
        # If not, or to be explicit:
        if not self.train_graphs and hasattr(datamodule, 'train_dataloader'):
            self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        if not self.val_graphs and hasattr(datamodule, 'val_dataloader'):
            self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        if not self.test_graphs and hasattr(datamodule, 'test_dataloader'):
            self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())

    def loader_to_nx(self, loader) -> List[nx.DiGraph]:  # Added return type hint
        """ Converts a PyG DataLoader's batches into a list of NetworkX DiGraphs
            using the AIG-specific conversion function for reference/dataset graphs.
        """
        networkx_graphs = []
        # if self.local_rank == 0: # Check if local_rank is available
        #     print(f"AIGSamplingMetrics: Loading reference graphs using 'convert_pyg_to_nx_for_aig_validation'...")

        if loader is None or loader.dataset is None or len(loader.dataset) == 0:
            # if self.local_rank == 0:
            #     print("AIGSamplingMetrics: Loader or its dataset is empty. Cannot load reference graphs.")
            return networkx_graphs

        for i, batch in enumerate(tqdm(loader, desc="Loading Ref Graphs",
                                       disable=(self.local_rank != 0 and hasattr(self, 'local_rank')))):
            data_list = batch.to_data_list()
            for j, pyg_data in enumerate(data_list):
                # pyg_data.edge_attr here is already (NumEdges, NUM_EDGE_FEATURES + 1)
                # from AIGCustomDataset.process()
                nx_graph = convert_pyg_to_nx_for_aig_validation(pyg_data)
                if nx_graph is not None:
                    networkx_graphs.append(nx_graph)
                # else:
                #     if self.local_rank == 0:
                #         print(f"Warning: Reference graph (batch {i}, item {j}) failed PyG to NX conversion.")
        # if self.local_rank == 0:
        #     print(f"AIGSamplingMetrics: Loaded {len(networkx_graphs)} reference graphs for this split.")
        return networkx_graphs

    def forward(self, generated_graphs_raw: list, name: str, current_epoch: int, val_counter: int, local_rank: int,
                test: bool = False):
        """
        Calculates metrics for generated AIGs.
        `generated_graphs_raw`: A list of tuples from the sampling process:
                                (node_features, edge_index, edge_features, num_nodes_tensor)
                                - node_features: (max_n_nodes, NUM_NODE_FEATURES)
                                - edge_index: (2, n_edges)
                                - edge_features: (n_edges, NUM_EDGE_FEATURES + 1) <--- IMPORTANT: Model output
        """
        reference_graphs_nx = self.test_graphs if test else self.val_graphs

        # if local_rank == 0:
        #     print(f"AIGMetrics (Rank {local_rank}): Evaluating {len(generated_graphs_raw)} generated AIGs "
        #           f"vs {len(reference_graphs_nx)} refs (Split: {'test' if test else 'val'}).")

        generated_graphs_nx = []
        # if local_rank == 0:
        #     print("AIGMetrics: Converting raw model outputs to NetworkX DiGraphs...")

        for i, graph_raw_data in enumerate(
                tqdm(generated_graphs_raw, desc="Converting Generated Graphs", disable=(local_rank != 0))):
            node_features, edge_index, edge_features_model, num_nodes_tensor = graph_raw_data
            num_nodes_int = num_nodes_tensor.item()

            # node_features from model is (max_nodes_in_batch, NUM_NODE_FEATURES)
            # edge_features_model from model is (num_edges_pred, NUM_EDGE_FEATURES + 1)

            # Slicing node_features to actual num_nodes_int is not needed here,
            # as convert_raw_model_output_to_nx_aig handles num_nodes_int internally.
            # However, it's good practice if the conversion function didn't.
            # current_node_features = node_features[:num_nodes_int] # This is fine

            nx_graph = convert_raw_model_output_to_nx_aig(
                node_features, edge_index, edge_features_model, num_nodes_int
            )  # Pass full node_features, function will use num_nodes_int
            if nx_graph is not None:
                generated_graphs_nx.append(nx_graph)

        if not generated_graphs_nx:
            if local_rank == 0:
                print("AIGMetrics: No generated graphs successfully converted. Skipping metrics.")
            return {}  # Return empty dict if no graphs

        to_log = {}
        # 1. AIG Structural Validity
        if 'aig_structural_validity' in self.metrics_list:
            # if local_rank == 0: print("AIGMetrics: Computing AIG structural validity...")
            num_structurally_valid = sum(1 for g in generated_graphs_nx if aig_check_validity(g))
            structural_validity_fraction = num_structurally_valid / len(
                generated_graphs_nx) if generated_graphs_nx else 0.0
            to_log['aig_metrics/structural_validity_fraction'] = structural_validity_fraction
            if wandb.run and local_rank == 0:
                wandb.summary[f'{name}_aig_structural_validity_fraction'] = structural_validity_fraction

        # 2. Acyclicity
        if 'aig_acyclicity' in self.metrics_list:
            # if local_rank == 0: print("AIGMetrics: Computing AIG acyclicity...")
            num_acyclic = sum(1 for g in generated_graphs_nx if nx.is_directed_acyclic_graph(g))
            acyclicity_fraction = num_acyclic / len(generated_graphs_nx) if generated_graphs_nx else 0.0
            to_log['aig_metrics/acyclicity_fraction'] = acyclicity_fraction
            if wandb.run and local_rank == 0:
                wandb.summary[f'{name}_aig_acyclicity_fraction'] = acyclicity_fraction

        # # 3. Degree Stats (Placeholder - needs directed version)
        # if 'degree' in self.metrics_list:
        #     # if local_rank == 0: print("AIGMetrics: Computing degree stats (placeholder)...")
        #     # TODO: Implement or use a directed_degree_stats function for AIGs
        #     # (separate in-degree and out-degree MMD/EMD)
        #     if reference_graphs_nx and generated_graphs_nx:  # Ensure lists are not empty
        #         degree_mmd = degree_stats(reference_graphs_nx, generated_graphs_nx, is_parallel=True,
        #                                   compute_emd=self.compute_emd)
        #         to_log['graph_stats/degree_mmd_placeholder'] = degree_mmd
        #         if wandb.run and local_rank == 0:
        #             wandb.summary[f'{name}_degree_mmd_placeholder'] = degree_mmd
        #     else:
                to_log['graph_stats/degree_mmd_placeholder'] = -1.0  # Indicate not computed

        # 4. Uniqueness, Isomorphism, Novelty
        # if local_rank == 0: print("AIGMetrics: Computing uniqueness, novelty...")

        def combined_aig_validity_for_eval_fractions(g_nx_eval):
            return aig_check_validity(g_nx_eval)

        train_graphs_nx = self.train_graphs

        eval_generated_graphs_nx = [g for g in generated_graphs_nx if g.number_of_nodes() > 0]
        eval_train_graphs_nx = [g for g in train_graphs_nx if g.number_of_nodes() > 0]

        frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = 0.0, 0.0, 0.0
        frac_non_iso_to_train = 0.0

        if not eval_generated_graphs_nx:
            # if local_rank == 0: print("AIGMetrics: No non-empty generated graphs for uniqueness/novelty.")
            pass  # Defaults are 0.0
        elif not eval_train_graphs_nx:
            # if local_rank == 0: print("AIGMetrics: No non-empty training graphs for novelty. Novelty will be 1.0.")
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(eval_generated_graphs_nx, [],
                                                          validity_func=combined_aig_validity_for_eval_fractions)
            frac_non_iso_to_train = 1.0
        else:
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(eval_generated_graphs_nx, eval_train_graphs_nx,
                                                          validity_func=combined_aig_validity_for_eval_fractions)
            frac_non_iso_to_train = 1.0 - eval_fraction_isomorphic(eval_generated_graphs_nx, eval_train_graphs_nx)

        to_log.update({
            'sampling_quality/frac_unique_aigs': frac_unique,
            'sampling_quality/frac_unique_non_iso_aigs': frac_unique_non_iso,
            'sampling_quality/frac_unique_non_iso_structurally_valid_aigs': frac_unique_non_iso_valid,
            'sampling_quality/frac_non_iso_to_train_aigs': frac_non_iso_to_train
        })

        # if local_rank == 0:
        #     print(f"AIGMetrics (Rank {local_rank}): Final metrics for {name}: {to_log}")
        if wandb.run and local_rank == 0:
            wandb.log({f"{name}_{k.replace('/', '_')}": v for k, v in to_log.items()}, commit=True)

        return to_log  # Return the metrics dictionary

    def reset(self):
        super().reset() if hasattr(super(), 'reset') else None
        # If you added any stateful attributes, reset them here.
        # For this version, it seems mostly stateless per forward call.
