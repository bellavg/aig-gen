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

            if not (np.isclose(np.sum(edge_feature_vector_model), 1.0) and
                    np.all(
                        (np.isclose(edge_feature_vector_model, 0.0)) | (np.isclose(edge_feature_vector_model, 1.0)))):
                pass

            shifted_type_index = np.argmax(edge_feature_vector_model)

            if shifted_type_index == 0:
                edge_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
            else:
                actual_aig_type_index = shifted_type_index - 1
                if not (0 <= actual_aig_type_index < len(EDGE_TYPE_KEYS)):
                    edge_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
                else:
                    edge_type_str = EDGE_TYPE_KEYS[actual_aig_type_index]
            nx_graph.add_edge(src_node, tgt_node, type=edge_type_str)
    return nx_graph


class AIGSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        # CRITICAL FIX: Initialize local_rank *before* calling super().__init__
        # Use .get() to provide a default value if 'local_rank' is not in the config.
        # Default to 0 for non-distributed runs.
        self.local_rank = datamodule.cfg.general.get('local_rank', 0)

        super().__init__(datamodule=datamodule,
                         compute_emd=False,  # AIGs typically don't use EMD for these structural/graph stats
                         metrics_list=['aig_structural_validity', 'aig_acyclicity', 'degree'])

        if not self.train_graphs and hasattr(datamodule,
                                             'train_dataloader') and datamodule.train_dataloader() is not None:
            self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        if not self.val_graphs and hasattr(datamodule, 'val_dataloader') and datamodule.val_dataloader() is not None:
            self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        if not self.test_graphs and hasattr(datamodule, 'test_dataloader') and datamodule.test_dataloader() is not None:
            self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())

    def loader_to_nx(self, loader) -> List[nx.DiGraph]:
        """ Converts a PyG DataLoader's batches into a list of NetworkX DiGraphs
            using the AIG-specific conversion function for reference/dataset graphs.
        """
        networkx_graphs = []

        if loader is None or not hasattr(loader, 'dataset') or loader.dataset is None or len(loader.dataset) == 0:
            # if self.local_rank == 0:
            #     print("AIGSamplingMetrics.loader_to_nx: Loader or its dataset is empty/invalid. Cannot load reference graphs.")
            return networkx_graphs

        for i, batch in enumerate(tqdm(loader, desc="Loading Ref Graphs",
                                       disable=(self.local_rank != 0))):  # self.local_rank is now guaranteed to exist
            data_list = batch.to_data_list()
            for j, pyg_data in enumerate(data_list):
                nx_graph = convert_pyg_to_nx_for_aig_validation(pyg_data)
                if nx_graph is not None:
                    networkx_graphs.append(nx_graph)
        return networkx_graphs

    def forward(self, generated_graphs_raw: list, name: str, current_epoch: int, val_counter: int,
                local_rank: int,
                test: bool = False):
        """
        Calculates metrics for generated AIGs.
        `local_rank_param`: The local rank passed to this function, typically from the trainer.
        """
        reference_graphs_nx = self.test_graphs if test else self.val_graphs
        current_rank_for_tqdm = local_rank

        generated_graphs_nx = []
        for i, graph_raw_data in enumerate(
                tqdm(generated_graphs_raw, desc="Converting Generated Graphs", disable=(current_rank_for_tqdm != 0))):
            node_features, edge_index, edge_features_model, num_nodes_tensor = graph_raw_data
            num_nodes_int = num_nodes_tensor.item()

            nx_graph = convert_raw_model_output_to_nx_aig(
                node_features, edge_index, edge_features_model, num_nodes_int
            )
            if nx_graph is not None:
                generated_graphs_nx.append(nx_graph)

        if not generated_graphs_nx:
            if current_rank_for_tqdm == 0:
                print("AIGMetrics: No generated graphs successfully converted. Skipping metrics.")
            return {}

        to_log = {}
        # 1. AIG Structural Validity
        if 'aig_structural_validity' in self.metrics_list:
            num_structurally_valid = sum(1 for g in generated_graphs_nx if aig_check_validity(g))
            structural_validity_fraction = num_structurally_valid / len(
                generated_graphs_nx) if generated_graphs_nx else 0.0
            to_log['aig_metrics/structural_validity_fraction'] = structural_validity_fraction
            if wandb.run and current_rank_for_tqdm == 0:
                wandb.summary[f'{name}_aig_structural_validity_fraction'] = structural_validity_fraction

        # 2. Acyclicity
        if 'aig_acyclicity' in self.metrics_list:
            num_acyclic = sum(1 for g in generated_graphs_nx if nx.is_directed_acyclic_graph(g))
            acyclicity_fraction = num_acyclic / len(generated_graphs_nx) if generated_graphs_nx else 0.0
            to_log['aig_metrics/acyclicity_fraction'] = acyclicity_fraction
            if wandb.run and current_rank_for_tqdm == 0:
                wandb.summary[f'{name}_aig_acyclicity_fraction'] = acyclicity_fraction

        # 3. Degree Stats
        # if 'degree' in self.metrics_list:
        #     if reference_graphs_nx and generated_graphs_nx:
        #         ref_undirected = [g.to_undirected(as_view=False) for g in
        #                           reference_graphs_nx]  # Use as_view=False for safety if graphs are modified
        #         gen_undirected = [g.to_undirected(as_view=False) for g in generated_graphs_nx]
        #         degree_mmd = degree_stats(ref_undirected, gen_undirected, is_parallel=True,
        #                                   compute_emd=self.compute_emd)
        #         to_log['graph_stats/degree_mmd_undirected'] = degree_mmd
        #         if wandb.run and current_rank_for_tqdm == 0:
        #             wandb.summary[f'{name}_degree_mmd_undirected'] = degree_mmd
        #     else:
        #         to_log['graph_stats/degree_mmd_undirected'] = -1.0

                # 4. Uniqueness, Isomorphism, Novelty

        def combined_aig_validity_for_eval_fractions(g_nx_eval):
            return aig_check_validity(g_nx_eval)

        train_graphs_nx = self.train_graphs

        eval_generated_graphs_nx = [g for g in generated_graphs_nx if g.number_of_nodes() > 0]
        eval_train_graphs_nx = [g for g in train_graphs_nx if g.number_of_nodes() > 0] if train_graphs_nx else []

        frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = 0.0, 0.0, 0.0
        frac_non_iso_to_train = 0.0

        if not eval_generated_graphs_nx:
            pass
        elif not eval_train_graphs_nx:
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(eval_generated_graphs_nx, [],
                                                          validity_func=combined_aig_validity_for_eval_fractions)
            frac_non_iso_to_train = 1.0 if eval_generated_graphs_nx else 0.0
        else:
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(eval_generated_graphs_nx, eval_train_graphs_nx,
                                                          validity_func=combined_aig_validity_for_eval_fractions)
            if eval_generated_graphs_nx:
                frac_non_iso_to_train = 1.0 - eval_fraction_isomorphic(eval_generated_graphs_nx, eval_train_graphs_nx)
            else:
                frac_non_iso_to_train = 0.0

        to_log.update({
            'sampling_quality/frac_unique_aigs': frac_unique,
            'sampling_quality/frac_unique_non_iso_aigs': frac_unique_non_iso,
            'sampling_quality/frac_unique_non_iso_structurally_valid_aigs': frac_unique_non_iso_valid,
            'sampling_quality/frac_non_iso_to_train_aigs': frac_non_iso_to_train
        })

        if wandb.run and current_rank_for_tqdm == 0:
            wandb.log({f"{name}_{k.replace('/', '_')}": v for k, v in to_log.items()}, commit=True)

        return to_log

    def reset(self):
        super().reset() if hasattr(super(), 'reset') else None
