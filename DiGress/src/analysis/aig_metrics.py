import torch
import torch.nn as nn
import torch.nn.functional as F  # Make sure F is imported
import networkx as nx
from tqdm import tqdm
import wandb
import numpy as np
from typing import Union, List, Dict, Any
import warnings

# DiGress existing utilities / Spectre-based metrics
from src.analysis.spectre_utils import SpectreSamplingMetrics, degree_stats, \
    eval_fraction_isomorphic, eval_fraction_unique_non_isomorphic_valid

# --- AIG-specific imports - Ensure these paths are correct ---
from src.aig_config import check_validity as aig_check_validity
from src.aig_config import NODE_TYPE_KEYS, EDGE_TYPE_KEYS, \
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES

from src.datasets.aig_custom_dataset import convert_pyg_to_nx_for_aig_validation


def convert_raw_model_output_to_nx_aig(node_features_tensor: torch.Tensor,
                                       edge_index_tensor: torch.Tensor,
                                       edge_features_tensor: torch.Tensor,
                                       num_nodes_int: int) -> Union[nx.DiGraph, None]:
    """
    Converts raw model output tensors (node features, sparse edge_index, and sparse edge_features)
    to a canonical NetworkX DiGraph for AIGs.
    Edges in the output DiGraph are directed from the node with the smaller ID
    to the node with the larger ID if they were to be made undirected, but here we respect raw_src/raw_tgt.
    This function is called AFTER the model's raw output (class indices) has been
    converted back to one-hot features and sparse edge representation.

    Args:
        node_features_tensor: One-hot tensor of shape (actual_num_nodes, NUM_NODE_FEATURES).
        edge_index_tensor: Tensor of shape (2, num_edges_in_prediction).
        edge_features_tensor: One-hot tensor of shape (num_edges_in_prediction, NUM_EDGE_FEATURES + 1).
                              Channel 0 is for "no specific AIG type" / padding,
                              channels 1 onwards map to actual AIG edge types.
        num_nodes_int: The actual number of nodes for this specific graph.

    Returns:
        A NetworkX DiGraph with 'type' attributes, or None if conversion fails.
    """
    nx_graph = nx.DiGraph()

    # Validate inputs
    if not (isinstance(node_features_tensor, torch.Tensor) and
            node_features_tensor.ndim == 2 and
            node_features_tensor.shape[0] == num_nodes_int and
            node_features_tensor.shape[1] == NUM_NODE_FEATURES):
        warnings.warn(
            f"Convert Model Output (to NX): Node features tensor has incorrect format. "
            f"Shape: {node_features_tensor.shape if isinstance(node_features_tensor, torch.Tensor) else 'Not a Tensor'}, Expected: ({num_nodes_int}, {NUM_NODE_FEATURES}). Skipping graph."
        )
        return None

    # Process nodes
    for i in range(num_nodes_int):
        node_feature_vector = node_features_tensor[i].cpu().numpy()
        if not (np.isclose(np.sum(node_feature_vector), 1.0) and
                np.all((np.isclose(node_feature_vector, 0.0)) | (np.isclose(node_feature_vector, 1.0)))):
            node_type_str = "UNKNOWN_TYPE_NON_ONE_HOT"
            warnings.warn(f"Convert Model Output (to NX): Node {i} features not one-hot: {node_feature_vector}")
        else:
            type_index = np.argmax(node_feature_vector)
            if not (0 <= type_index < len(NODE_TYPE_KEYS)):
                node_type_str = "UNKNOWN_TYPE_BAD_INDEX"
                warnings.warn(
                    f"Convert Model Output (to NX): Node {i} type index {type_index} out of bounds for NODE_TYPE_KEYS (len {len(NODE_TYPE_KEYS)}).")
            else:
                node_type_str = NODE_TYPE_KEYS[type_index]
        nx_graph.add_node(i, type=node_type_str)

    # Process edges
    num_edges_in_prediction = edge_index_tensor.shape[1]
    expected_edge_feature_dim = NUM_EDGE_FEATURES + 1  # Includes the "no specific type" channel

    if num_edges_in_prediction > 0:
        if not (isinstance(edge_index_tensor, torch.Tensor) and
                edge_index_tensor.ndim == 2 and
                edge_index_tensor.shape[0] == 2):
            warnings.warn(
                f"Convert Model Output (to NX): Edge index tensor has incorrect dimensions. "
                f"Shape: {edge_index_tensor.shape if isinstance(edge_index_tensor, torch.Tensor) else 'Not a Tensor'}, Expected: (2, NumEdges). Skipping graph."
            )
            return None

        if not (isinstance(edge_features_tensor, torch.Tensor) and
                edge_features_tensor.ndim == 2 and
                edge_features_tensor.shape[0] == num_edges_in_prediction and
                edge_features_tensor.shape[1] == expected_edge_feature_dim):
            warnings.warn(
                f"Convert Model Output (to NX): Edge features tensor shape mismatch. "
                f"Got {edge_features_tensor.shape if isinstance(edge_features_tensor, torch.Tensor) else 'Not a Tensor'}, expected ({num_edges_in_prediction}, {expected_edge_feature_dim}). Skipping graph."
            )
            return None

        for i in range(num_edges_in_prediction):
            raw_src_node = edge_index_tensor[0, i].item()
            raw_tgt_node = edge_index_tensor[1, i].item()

            if not (0 <= raw_src_node < num_nodes_int and 0 <= raw_tgt_node < num_nodes_int):
                warnings.warn(
                    f"Convert Model Output (to NX): Edge ({raw_src_node} -> {raw_tgt_node}) contains node IDs "
                    f"out of bounds for num_nodes ({num_nodes_int}). Skipping this edge."
                )
                continue

            edge_feature_vector_one_hot = edge_features_tensor[i].cpu().numpy()
            if not (np.isclose(np.sum(edge_feature_vector_one_hot), 1.0) and
                    np.all((np.isclose(edge_feature_vector_one_hot, 0.0)) | (
                    np.isclose(edge_feature_vector_one_hot, 1.0)))):
                warnings.warn(
                    f"Convert Model Output (to NX): Edge ({raw_src_node}-{raw_tgt_node}) one-hot feature vector "
                    f"is not valid: {edge_feature_vector_one_hot}. Argmax will still be used."
                )

            shifted_type_index = np.argmax(edge_feature_vector_one_hot)  # Index from 0 to NUM_EDGE_FEATURES
            edge_type_str: str

            if shifted_type_index == 0:  # This is the "no specific AIG type" or padding channel
                edge_type_str = "EDGE_GENERIC_OR_PADDING"  # Or however you want to label these
                # This case implies the edge exists structurally but has no specific AIG type assigned by the model,
                # or it's an edge that should be ignored if channel 0 truly means "no edge".
                # Given the previous logic, if argmax is 0, it means the model predicted the "no specific type" channel.
            else:
                actual_aig_type_index = shifted_type_index - 1  # Convert to 0-based index for EDGE_TYPE_KEYS
                if not (0 <= actual_aig_type_index < len(EDGE_TYPE_KEYS)):
                    edge_type_str = "UNKNOWN_TYPE_BAD_INDEX"
                    warnings.warn(
                        f"Convert Model Output (to NX): Edge ({raw_src_node}-{raw_tgt_node}) decoded to invalid "
                        f"actual_aig_type_index {actual_aig_type_index} from shifted index {shifted_type_index} (len EDGE_TYPE_KEYS: {len(EDGE_TYPE_KEYS)})."
                    )
                else:
                    edge_type_str = EDGE_TYPE_KEYS[actual_aig_type_index]

            # For AIGs, usually, we might want to enforce a canonical direction (e.g. min_id -> max_id)
            # if the underlying graph is conceptually undirected but represented directed.
            # However, the input `edge_index_tensor` from the model's output processing
            # (in AIGSamplingMetrics.forward) already iterates u,v and adds [u,v],
            # so we use raw_src_node and raw_tgt_node directly here.
            # If canonical direction is strictly needed for validation, it should be enforced here or before.
            # For now, respecting the (potentially directed) edge from the processed model output.

            # Add edge if it doesn't exist, or update if it does (though less likely for new graph construction)
            if not nx_graph.has_edge(raw_src_node, raw_tgt_node):
                nx_graph.add_edge(raw_src_node, raw_tgt_node, type=edge_type_str)
            # else:
            #    # Potentially handle cases where an edge might be defined multiple times if input is noisy,
            #    # though the sparse conversion should ideally handle this.
            #    # For now, we assume the first definition is fine or that duplicates are filtered before this.
            #    pass

    return nx_graph


class AIGSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule: Any):
        self.local_rank = datamodule.cfg.general.get('local_rank', 0)
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['aig_structural_validity', 'aig_acyclicity', 'degree'])
        self.train_graphs: List[nx.Graph] = []
        self.val_graphs: List[nx.Graph] = []
        self.test_graphs: List[nx.Graph] = []

        if hasattr(datamodule, 'train_dataloader') and datamodule.train_dataloader() is not None:
            self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        if hasattr(datamodule, 'val_dataloader') and datamodule.val_dataloader() is not None:
            self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        if hasattr(datamodule, 'test_dataloader') and datamodule.test_dataloader() is not None:
            self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())

    def loader_to_nx(self, loader: Any) -> List[nx.Graph]:
        networkx_graphs: List[nx.Graph] = []
        if loader is None or not hasattr(loader, 'dataset') or loader.dataset is None or len(loader.dataset) == 0:
            if self.local_rank == 0:
                warnings.warn(
                    "AIGSamplingMetrics.loader_to_nx: Loader or its dataset is empty/invalid. Cannot load reference graphs.")
            return networkx_graphs
        for i, batch in enumerate(tqdm(loader, desc="Loading Reference Graphs for Metrics",
                                       disable=(self.local_rank != 0))):
            data_list = batch.to_data_list()
            for j, pyg_data in enumerate(data_list):
                # convert_pyg_to_nx_for_aig_validation expects edge_attr to be (num_edges, NUM_EDGE_FEATURES + 1)
                # The data from AIGDataset.process() should already be in this format.
                nx_graph = convert_pyg_to_nx_for_aig_validation(pyg_data)
                if nx_graph is not None:
                    networkx_graphs.append(nx_graph)
        return networkx_graphs

    def forward(self, generated_graphs_raw: list, name: str, current_epoch: int, val_counter: int,
                local_rank: int,
                test: bool = False) -> Dict[str, float]:
        """
        Calculates and logs metrics for generated AIGs.
        Args:
            generated_graphs_raw: A list of 2-element raw graph data from the model's sampling.
                                  Expected structure per item: [node_indices_tensor, edge_indices_matrix]
                                  - node_indices_tensor: (n,) tensor of node type class indices.
                                  - edge_indices_matrix: (n, n) tensor of edge type class indices.
            name: Name of the run/experiment.
            current_epoch: Current training epoch.
            val_counter: Validation counter.
            local_rank: The local rank of the current process.
            test: Boolean flag indicating if this is a test phase.
        Returns:
            A dictionary of calculated metrics.
        """
        reference_graphs_nx: List[nx.Graph] = self.test_graphs if test else self.val_graphs
        generated_graphs_nx_directed: List[nx.DiGraph] = []

        for i, graph_raw_data_item in enumerate(
                tqdm(generated_graphs_raw, desc="Converting Generated Graphs for Metrics", disable=(local_rank != 0))):

            if not (isinstance(graph_raw_data_item, (list, tuple)) and len(graph_raw_data_item) == 2):
                if local_rank == 0:
                    warnings.warn(
                        f"AIGMetrics: Item {i} in generated_graphs_raw has unexpected format. "
                        f"Expected 2 elements (node_indices, edge_indices_matrix), got {len(graph_raw_data_item) if graph_raw_data_item is not None else 'None'}. Skipping this graph."
                    )
                continue

            node_indices_tensor, edge_indices_matrix = graph_raw_data_item

            if not isinstance(node_indices_tensor, torch.Tensor) or not isinstance(edge_indices_matrix, torch.Tensor):
                if local_rank == 0:
                    warnings.warn(f"AIGMetrics: Item {i} data is not in tensor format. Skipping.")
                continue

            if node_indices_tensor.ndim == 0:  # Handle scalar tensor case if a graph has 0 nodes effectively
                if local_rank == 0:
                    warnings.warn(f"AIGMetrics: Item {i} node_indices_tensor is scalar (likely 0 nodes). Skipping.")
                continue

            num_nodes_int = node_indices_tensor.shape[0]
            if num_nodes_int == 0:
                if local_rank == 0:
                    warnings.warn(f"AIGMetrics: Item {i} has 0 nodes. Skipping.")
                continue

            # --- Convert model output (class indices) to one-hot features for convert_raw_model_output_to_nx_aig ---

            # 1. Node features: Convert node class indices to one-hot
            if node_indices_tensor.is_floating_point():  # Should not happen if it's class indices
                if local_rank == 0:
                    warnings.warn(
                        f"AIGMetrics: Item {i} node_indices_tensor is float, expected long/int for F.one_hot. Attempting cast.")
                node_indices_tensor = node_indices_tensor.long()
            elif node_indices_tensor.dtype not in [torch.long, torch.int]:  # Check for other non-integer types
                if local_rank == 0:
                    warnings.warn(
                        f"AIGMetrics: Item {i} node_indices_tensor has dtype {node_indices_tensor.dtype}, expected long/int. Attempting cast.")
                node_indices_tensor = node_indices_tensor.long()

            actual_node_features_one_hot = F.one_hot(node_indices_tensor, num_classes=NUM_NODE_FEATURES).float()

            # 2. Edge features: Convert (N,N) edge class index matrix to sparse edge_index and one-hot edge_features
            adj_edges_list = []
            adj_edge_features_one_hot_list = []
            expected_edge_feature_output_dim = NUM_EDGE_FEATURES + 1  # This is the dim the model predicts (includes "no type" channel)

            for u_idx in range(num_nodes_int):
                for v_idx in range(num_nodes_int):
                    # For AIGs, we assume directed edges from the model's (N,N) output.
                    # If the model is only meant to predict an undirected adjacency and then canonicalize,
                    # this loop might need adjustment (e.g., range v_idx from u_idx + 1 and add both directions,
                    # or only add one and let convert_raw_model_output_to_nx_aig handle canonicalization if needed).
                    # Given the DiGress structure, (N,N) output for E implies directed edges.

                    edge_class_idx = edge_indices_matrix[u_idx, v_idx].item()  # Class index from 0 to NUM_EDGE_FEATURES

                    if edge_class_idx != 0:  # If it's not the "no specific AIG type" / "no edge" channel
                        adj_edges_list.append([u_idx, v_idx])

                        # Create one-hot vector for this edge type.
                        # The edge_class_idx (0 to NUM_EDGE_FEATURES) directly corresponds to the channel.
                        if not (0 <= edge_class_idx < expected_edge_feature_output_dim):
                            if local_rank == 0:
                                warnings.warn(
                                    f"AIGMetrics: Item {i}, edge ({u_idx},{v_idx}): invalid edge_class_idx {edge_class_idx} "
                                    f"for one-hot encoding with {expected_edge_feature_output_dim} classes. Skipping edge."
                                )
                            continue

                        edge_one_hot = F.one_hot(torch.tensor(edge_class_idx).long(),
                                                 num_classes=expected_edge_feature_output_dim).float()
                        adj_edge_features_one_hot_list.append(edge_one_hot)

            if not adj_edges_list:
                current_edge_index = torch.empty((2, 0), dtype=torch.long, device=actual_node_features_one_hot.device)
                current_edge_features_model_one_hot = torch.empty((0, expected_edge_feature_output_dim),
                                                                  dtype=actual_node_features_one_hot.dtype,
                                                                  device=actual_node_features_one_hot.device)
            else:
                current_edge_index = torch.tensor(adj_edges_list, dtype=torch.long,
                                                  device=actual_node_features_one_hot.device).t().contiguous()
                current_edge_features_model_one_hot = torch.stack(adj_edge_features_one_hot_list).to(
                    device=actual_node_features_one_hot.device)

            # --- Call the conversion function with correctly formatted one-hot features ---
            nx_di_graph = convert_raw_model_output_to_nx_aig(
                actual_node_features_one_hot,
                current_edge_index,
                current_edge_features_model_one_hot,
                num_nodes_int
            )
            if nx_di_graph is not None:
                generated_graphs_nx_directed.append(nx_di_graph)

        if not generated_graphs_nx_directed:
            if local_rank == 0:
                print(
                    "AIGMetrics: No generated graphs were successfully converted to NetworkX DiGraphs. Skipping metrics calculation.")
            return {}

        to_log: Dict[str, float] = {}

        if 'aig_structural_validity' in self.metrics_list:
            num_structurally_valid = sum(1 for g in generated_graphs_nx_directed if aig_check_validity(g))
            structural_validity_fraction = num_structurally_valid / len(
                generated_graphs_nx_directed) if generated_graphs_nx_directed else 0.0
            to_log['aig_metrics/structural_validity_fraction'] = structural_validity_fraction
            if wandb.run and local_rank == 0:
                wandb.summary[f'{name}_aig_structural_validity_fraction'] = structural_validity_fraction

        if 'aig_acyclicity' in self.metrics_list:
            num_acyclic = sum(1 for g in generated_graphs_nx_directed if nx.is_directed_acyclic_graph(g))
            acyclicity_fraction = num_acyclic / len(
                generated_graphs_nx_directed) if generated_graphs_nx_directed else 0.0
            to_log['aig_metrics/acyclicity_fraction'] = acyclicity_fraction
            if wandb.run and local_rank == 0:
                wandb.summary[f'{name}_aig_acyclicity_fraction'] = acyclicity_fraction

        if 'degree' in self.metrics_list:
            current_ref_graphs = self.test_graphs if test else self.val_graphs
            if current_ref_graphs and generated_graphs_nx_directed:
                # Degree stats typically use undirected graphs.
                # The reference graphs from loader_to_nx are already DiGraphs from convert_pyg_to_nx_for_aig_validation
                # which itself creates DiGraphs. If degree stats need undirected, convert both ref and gen.
                ref_undirected_for_degree = [g.to_undirected(as_view=False) for g in current_ref_graphs if
                                             g.number_of_nodes() > 0]
                gen_undirected_for_degree = [g.to_undirected(as_view=False) for g in generated_graphs_nx_directed if
                                             g.number_of_nodes() > 0]
                if ref_undirected_for_degree and gen_undirected_for_degree:
                    degree_mmd = degree_stats(ref_undirected_for_degree, gen_undirected_for_degree,
                                              is_parallel=True,
                                              compute_emd=self.compute_emd)
                    to_log['graph_stats/degree_mmd_undirected'] = degree_mmd
                    if wandb.run and local_rank == 0:
                        wandb.summary[f'{name}_degree_mmd_undirected'] = degree_mmd
                else:
                    to_log['graph_stats/degree_mmd_undirected'] = -1.0  # Indicate not computed
            else:
                to_log['graph_stats/degree_mmd_undirected'] = -1.0  # Indicate not computed

        def combined_aig_validity_for_eval_fractions(g_eval_nx: nx.DiGraph) -> bool:
            return aig_check_validity(g_eval_nx)  # Assuming aig_check_validity works on DiGraph

        # For novelty/uniqueness, we use the directed graphs as generated and converted.
        # The reference train_graphs are also DiGraphs from loader_to_nx.
        eval_generated_graphs_nx = [g for g in generated_graphs_nx_directed if g.number_of_nodes() > 0]
        eval_train_graphs_nx = [g for g in self.train_graphs if
                                g.number_of_nodes() > 0]  # self.train_graphs are already DiGraphs

        frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = 0.0, 0.0, 0.0
        frac_non_iso_to_train = 0.0

        if not eval_generated_graphs_nx:
            if local_rank == 0:
                print("AIGMetrics: No valid generated graphs for uniqueness/novelty checks.")
        elif not eval_train_graphs_nx:  # If no training graphs, all valid unique generated are novel
            if local_rank == 0:
                print(
                    "AIGMetrics: No training graphs for novelty comparison. Calculating uniqueness/validity of generated set.")
            # Validity here refers to combined_aig_validity_for_eval_fractions (which is aig_check_validity)
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(
                    eval_generated_graphs_nx,
                    [],  # No training graphs to compare against for isomorphism
                    validity_func=combined_aig_validity_for_eval_fractions
                )
            frac_non_iso_to_train = 1.0  # All are non-isomorphic to an empty training set
        else:
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(
                    eval_generated_graphs_nx,
                    eval_train_graphs_nx,
                    validity_func=combined_aig_validity_for_eval_fractions
                )
            frac_non_iso_to_train = 1.0 - eval_fraction_isomorphic(  # This needs to handle DiGraphs correctly
                eval_generated_graphs_nx,
                eval_train_graphs_nx
            )

        to_log.update({
            'sampling_quality/frac_unique_aigs': frac_unique,
            'sampling_quality/frac_unique_non_iso_aigs': frac_unique_non_iso,
            'sampling_quality/frac_unique_non_iso_structurally_valid_aigs': frac_unique_non_iso_valid,
            'sampling_quality/frac_non_iso_to_train_aigs': frac_non_iso_to_train
        })

        if wandb.run and local_rank == 0:
            wandb_log_data = {f"{name}_{k.replace('/', '_')}": v for k, v in to_log.items()}
            wandb.log(wandb_log_data, commit=True)

        if local_rank == 0:
            print(f"AIGMetrics Epoch {current_epoch} ({'Test' if test else 'Val'}):")
            for key, val in to_log.items():
                print(f"  {key}: {val:.4f}")
        return to_log

    def reset(self):
        super().reset() if hasattr(super(), 'reset') else None
        pass
