import torch
import torch.nn as nn
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
from src.abstract_metrics import compute_ratios  # Added import for compute_ratios


def convert_raw_model_output_to_nx_aig(node_features_tensor: torch.Tensor,
                                       edge_index_tensor: torch.Tensor,
                                       edge_features_tensor: torch.Tensor,
                                       num_nodes_int: int) -> Union[nx.DiGraph, None]:
    """
    Converts raw model output tensors (assumed to represent an undirected graph
    via its edge_index and edge_features) to a canonical NetworkX DiGraph for AIGs.
    Edges in the output DiGraph are directed from the node with the smaller ID
    to the node with the larger ID.

    The model's edge_features_tensor is expected to have NUM_EDGE_FEATURES + 1 channels,
    where channel 0 is for "no specific AIG type" / padding, and channels 1 onwards
    (after shifting by -1) correspond to actual AIG edge types.

    Args:
        node_features_tensor: Tensor of shape (actual_num_nodes, NUM_NODE_FEATURES).
                              Node features from the model's sampling output.
        edge_index_tensor: Tensor of shape (2, num_edges_in_prediction).
                           Edge index from the model's sampling output. This might contain
                           symmetric pairs (e.g., (u,v) and (v,u)).
        edge_features_tensor: Tensor of shape (num_edges_in_prediction, NUM_EDGE_FEATURES + 1).
                              Edge features from the model's sampling output.
        num_nodes_int: The actual number of nodes for this specific graph.

    Returns:
        A NetworkX DiGraph with 'type' attributes and canonical edge direction,
        or None if conversion fails or inputs are invalid.
    """
    nx_graph = nx.DiGraph()

    # Validate inputs
    # For node_features_tensor, its first dimension IS num_nodes_int due to slicing in sample_batch
    if node_features_tensor.shape[0] != num_nodes_int:
        warnings.warn(
            f"Convert Model Output: num_nodes_int ({num_nodes_int}) does not match "
            f"node_features_tensor.shape[0] ({node_features_tensor.shape[0]}). Skipping graph."
        )
        return None
    if node_features_tensor.ndim != 2 or node_features_tensor.shape[1] != NUM_NODE_FEATURES:
        warnings.warn(
            f"Convert Model Output: Node features tensor has incorrect dimensions. "
            f"Shape: {node_features_tensor.shape}, Expected: ({num_nodes_int}, {NUM_NODE_FEATURES}). Skipping graph."
        )
        return None

    # Process nodes up to num_nodes_int
    for i in range(num_nodes_int):
        node_feature_vector = node_features_tensor[i].cpu().numpy()
        if not (np.isclose(np.sum(node_feature_vector), 1.0) and
                np.all((np.isclose(node_feature_vector, 0.0)) | (np.isclose(node_feature_vector, 1.0)))):
            node_type_str = "UNKNOWN_TYPE_NON_ONE_HOT"
            warnings.warn(f"Convert Model Output: Node {i} features not one-hot: {node_feature_vector}")
        else:
            type_index = np.argmax(node_feature_vector)
            if not (0 <= type_index < len(NODE_TYPE_KEYS)):
                node_type_str = "UNKNOWN_TYPE_BAD_INDEX"
                warnings.warn(
                    f"Convert Model Output: Node {i} type index {type_index} out of bounds for NODE_TYPE_KEYS (len {len(NODE_TYPE_KEYS)}).")
            else:
                node_type_str = NODE_TYPE_KEYS[type_index]
        nx_graph.add_node(i, type=node_type_str)

    # Process edges
    num_edges_in_prediction = edge_index_tensor.shape[1]
    expected_edge_feature_dim = NUM_EDGE_FEATURES + 1

    if num_edges_in_prediction > 0:
        if edge_index_tensor.ndim != 2 or edge_index_tensor.shape[0] != 2:
            warnings.warn(
                f"Convert Model Output: Edge index tensor has incorrect dimensions. "
                f"Shape: {edge_index_tensor.shape}, Expected: (2, NumEdges). Skipping graph."
            )
            return None
        if edge_features_tensor.ndim != 2 or \
                edge_features_tensor.shape[0] != num_edges_in_prediction or \
                edge_features_tensor.shape[1] != expected_edge_feature_dim:
            warnings.warn(
                f"Convert Model Output: Edge features tensor shape mismatch. "
                f"Got {edge_features_tensor.shape}, expected ({num_edges_in_prediction}, {expected_edge_feature_dim}). Skipping graph."
            )
            return None

        for i in range(num_edges_in_prediction):
            raw_src_node = edge_index_tensor[0, i].item()
            raw_tgt_node = edge_index_tensor[1, i].item()

            if not (0 <= raw_src_node < num_nodes_int and 0 <= raw_tgt_node < num_nodes_int):
                warnings.warn(
                    f"Convert Model Output: Edge ({raw_src_node} -> {raw_tgt_node}) contains node IDs "
                    f"out of bounds for num_nodes ({num_nodes_int}). Skipping this edge."
                )
                continue

            edge_feature_vector_model = edge_features_tensor[i].cpu().numpy()
            if not (np.isclose(np.sum(edge_feature_vector_model), 1.0) and
                    np.all(
                        (np.isclose(edge_feature_vector_model, 0.0)) | (np.isclose(edge_feature_vector_model, 1.0)))):
                warnings.warn(
                    f"Convert Model Output: Edge ({raw_src_node}-{raw_tgt_node}) feature vector from model "
                    f"is not one-hot: {edge_feature_vector_model}. Argmax will still be used."
                )

            shifted_type_index = np.argmax(edge_feature_vector_model)
            edge_type_str: str

            if shifted_type_index == 0:
                edge_type_str = "EDGE_GENERIC_OR_PADDING"  # This type will likely be ignored or handled as "no edge" by validation
            else:
                actual_aig_type_index = shifted_type_index - 1
                if not (0 <= actual_aig_type_index < len(EDGE_TYPE_KEYS)):
                    edge_type_str = "UNKNOWN_TYPE_BAD_INDEX"
                    warnings.warn(
                        f"Convert Model Output: Edge ({raw_src_node}-{raw_tgt_node}) decoded to invalid "
                        f"actual_aig_type_index {actual_aig_type_index} from shifted index {shifted_type_index}."
                    )
                else:
                    edge_type_str = EDGE_TYPE_KEYS[actual_aig_type_index]

            # Only add edge if it's a valid AIG type (not padding/generic)
            if edge_type_str != "EDGE_GENERIC_OR_PADDING" and edge_type_str != "UNKNOWN_TYPE_BAD_INDEX":
                src_final = min(raw_src_node, raw_tgt_node)
                tgt_final = max(raw_src_node, raw_tgt_node)

                if src_final == tgt_final:  # Self-loop
                    # AIGs typically don't have self-loops, but if they were allowed:
                    # if not nx_graph.has_edge(src_final, tgt_final):
                    #    nx_graph.add_edge(src_final, tgt_final, type=edge_type_str)
                    warnings.warn(
                        f"Convert Model Output: Self-loop detected ({src_final}-{tgt_final}) and ignored for AIG.")
                    pass
                elif not nx_graph.has_edge(src_final, tgt_final):
                    nx_graph.add_edge(src_final, tgt_final, type=edge_type_str)
                # If an edge (u,v) already exists, we might have a multi-graph scenario from the model.
                # For simplicity here, we only add the first encountered edge.
                # Or, one could check if the type is different and handle accordingly.
    return nx_graph


class AIGSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule: Any):
        self.local_rank = datamodule.cfg.general.get('local_rank', 0)
        # Initialize SpectreSamplingMetrics with AIG-specific metrics
        super().__init__(datamodule=datamodule,
                         compute_emd=False,  # EMD might not be directly applicable or desired for AIGs
                         metrics_list=['aig_structural_validity', 'aig_acyclicity', 'degree'])

        # Store reference graphs (converted to NetworkX)
        self.train_graphs: List[nx.Graph] = []
        self.val_graphs: List[nx.Graph] = []
        self.test_graphs: List[nx.Graph] = []

        # Load reference graphs during initialization
        # Ensure dataloaders are available and correctly configured
        if hasattr(datamodule, 'train_dataloader') and datamodule.train_dataloader() is not None:
            self.train_graphs = self.loader_to_nx(datamodule.train_dataloader(), "Train")
        if hasattr(datamodule, 'val_dataloader') and datamodule.val_dataloader() is not None:
            self.val_graphs = self.loader_to_nx(datamodule.val_dataloader(), "Validation")
        if hasattr(datamodule, 'test_dataloader') and datamodule.test_dataloader() is not None:
            self.test_graphs = self.loader_to_nx(datamodule.test_dataloader(), "Test")

    def loader_to_nx(self, loader: Any, split_name: str) -> List[nx.Graph]:
        """Converts PyG graphs from a dataloader to NetworkX graphs for AIG validation."""
        networkx_graphs: List[nx.Graph] = []
        if loader is None or not hasattr(loader, 'dataset') or loader.dataset is None or len(loader.dataset) == 0:
            if self.local_rank == 0:
                warnings.warn(
                    f"AIGSamplingMetrics.loader_to_nx ({split_name}): Loader or its dataset is empty/invalid. Cannot load reference graphs.")
            return networkx_graphs

        # Iterate through the dataset in the loader
        for i, pyg_data in enumerate(tqdm(loader.dataset, desc=f"Loading {split_name} Reference Graphs for Metrics",
                                          disable=(self.local_rank != 0))):
            # Assuming pyg_data is directly the graph object, not a batch
            nx_graph = convert_pyg_to_nx_for_aig_validation(pyg_data)
            if nx_graph is not None:
                networkx_graphs.append(nx_graph)
        return networkx_graphs

    # MODIFIED: Added ref_metrics parameter with a default value
    def forward(self, generated_graphs_raw: list, name: str, current_epoch: int, val_counter: int,
                local_rank: int, test: bool = False, ref_metrics: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Calculates and logs metrics for generated AIGs.

        Args:
            generated_graphs_raw: A list of 2-element raw graph data from the model's sampling.
                                  Expected structure per item: (node_features_tensor, dense_E_feat_matrix)
                                  - node_features_tensor: (n, NUM_NODE_FEATURES)
                                  - dense_E_feat_matrix: (n, n, NUM_EDGE_FEATURES + 1)
            name: Name of the run/experiment.
            current_epoch: Current training epoch.
            val_counter: Validation counter.
            local_rank: The local rank of the current process.
            test: Boolean flag indicating if this is a test phase.
            ref_metrics: Optional dictionary containing reference metrics for ratio calculation.

        Returns:
            A dictionary of calculated metrics.
        """
        reference_graphs_nx: List[nx.Graph] = self.test_graphs if test else self.val_graphs
        generated_graphs_nx_directed: List[nx.DiGraph] = []

        # Convert raw model output to NetworkX DiGraphs
        for i, graph_raw_data_item in enumerate(
                tqdm(generated_graphs_raw, desc="Converting Generated Graphs for Metrics", disable=(local_rank != 0))):

            if not (isinstance(graph_raw_data_item, (list, tuple)) and len(graph_raw_data_item) == 2):
                if local_rank == 0:
                    warnings.warn(
                        f"AIGMetrics: Item {i} in generated_graphs_raw has unexpected format. "
                        f"Expected 2 elements (node_features, dense_edge_features), got {len(graph_raw_data_item) if isinstance(graph_raw_data_item, (list, tuple)) else 'NotListOrTuple'}. Skipping this graph."
                    )
                continue

            node_features_tensor, dense_E_feat_matrix = graph_raw_data_item

            if not isinstance(node_features_tensor, torch.Tensor):
                if local_rank == 0:
                    warnings.warn(f"AIGMetrics: Item {i} node_features_tensor is not a tensor. Skipping.")
                continue
            if not isinstance(dense_E_feat_matrix, torch.Tensor):
                if local_rank == 0:
                    warnings.warn(f"AIGMetrics: Item {i} dense_E_feat_matrix is not a tensor. Skipping.")
                continue

            num_nodes_int = node_features_tensor.shape[0]

            # Ensure dense_E_feat_matrix has the correct shape (N, N, C)
            if dense_E_feat_matrix.shape != (num_nodes_int, num_nodes_int, NUM_EDGE_FEATURES + 1):
                if local_rank == 0:
                    warnings.warn(
                        f"AIGMetrics: Item {i} dense_E_feat_matrix has unexpected shape {dense_E_feat_matrix.shape}. "
                        f"Expected ({num_nodes_int}, {num_nodes_int}, {NUM_EDGE_FEATURES + 1}). Skipping this graph."
                    )
                continue

            # Convert dense edge features to sparse edge_index and edge_features
            adj_edges_list = []
            adj_edge_features_list = []
            edge_feature_dim = dense_E_feat_matrix.shape[-1]  # Should be NUM_EDGE_FEATURES + 1

            for u in range(num_nodes_int):
                for v in range(u + 1, num_nodes_int):  # Iterate upper triangle for undirected representation
                    edge_one_hot_feat = dense_E_feat_matrix[u, v]

                    # Channel 0 in the last dim of E means "no specific AIG type" or "no edge"
                    # Actual AIG edges will have argmax > 0.
                    if torch.argmax(edge_one_hot_feat).item() != 0:
                        adj_edges_list.append([u, v])
                        adj_edges_list.append([v, u])  # Add symmetric edge for edge_index format
                        adj_edge_features_list.append(edge_one_hot_feat)
                        adj_edge_features_list.append(edge_one_hot_feat)  # Features for symmetric edge

            if not adj_edges_list:  # Handle graphs with no edges
                current_edge_index = torch.empty((2, 0), dtype=torch.long, device=node_features_tensor.device)
                current_edge_features_model = torch.empty((0, edge_feature_dim), dtype=node_features_tensor.dtype,
                                                          device=node_features_tensor.device)
            else:
                current_edge_index = torch.tensor(adj_edges_list, dtype=torch.long,
                                                  device=node_features_tensor.device).t().contiguous()
                current_edge_features_model = torch.stack(adj_edge_features_list).to(device=node_features_tensor.device)

            nx_di_graph = convert_raw_model_output_to_nx_aig(
                node_features_tensor,
                current_edge_index,
                current_edge_features_model,
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

        # Calculate AIG-specific structural validity
        if 'aig_structural_validity' in self.metrics_list:
            num_structurally_valid = sum(1 for g in generated_graphs_nx_directed if aig_check_validity(g))
            structural_validity_fraction = num_structurally_valid / len(
                generated_graphs_nx_directed) if generated_graphs_nx_directed else 0.0
            to_log['aig_metrics/structural_validity_fraction'] = structural_validity_fraction
            if wandb.run and local_rank == 0:  # Log to wandb summary
                wandb.summary[f'{name}_aig_structural_validity_fraction'] = structural_validity_fraction

        # Calculate AIG acyclicity
        if 'aig_acyclicity' in self.metrics_list:
            num_acyclic = sum(1 for g in generated_graphs_nx_directed if nx.is_directed_acyclic_graph(g))
            acyclicity_fraction = num_acyclic / len(
                generated_graphs_nx_directed) if generated_graphs_nx_directed else 0.0
            to_log['aig_metrics/acyclicity_fraction'] = acyclicity_fraction
            if wandb.run and local_rank == 0:  # Log to wandb summary
                wandb.summary[f'{name}_aig_acyclicity_fraction'] = acyclicity_fraction

        # Calculate degree statistics (MMD)
        if 'degree' in self.metrics_list:
            # Use the appropriate set of reference graphs (test or validation)
            current_ref_graphs_for_degree = self.test_graphs if test else self.val_graphs
            if current_ref_graphs_for_degree and generated_graphs_nx_directed:
                # Filter out empty graphs if any, and convert reference to undirected for degree comparison
                ref_undirected_for_degree = [g.to_undirected(as_view=False) for g in current_ref_graphs_for_degree if
                                             g.number_of_nodes() > 0]
                gen_undirected_for_degree = [g.to_undirected(as_view=False) for g in generated_graphs_nx_directed if
                                             g.number_of_nodes() > 0]

                if ref_undirected_for_degree and gen_undirected_for_degree:
                    degree_mmd = degree_stats(ref_undirected_for_degree, gen_undirected_for_degree,
                                              is_parallel=True,  # Can set to False if issues arise
                                              compute_emd=self.compute_emd)  # self.compute_emd is False
                    to_log['graph_stats/degree_mmd_undirected'] = degree_mmd
                    if wandb.run and local_rank == 0:  # Log to wandb summary
                        wandb.summary[f'{name}_degree_mmd_undirected'] = degree_mmd
                else:
                    to_log['graph_stats/degree_mmd_undirected'] = -1.0  # Indicate failure or no comparable graphs
            else:
                to_log['graph_stats/degree_mmd_undirected'] = -1.0  # Indicate failure or no comparable graphs

        # Combined validity function for uniqueness/novelty checks
        def combined_aig_validity_for_eval_fractions(g_eval_nx: nx.DiGraph) -> bool:
            # For AIGs, structural validity is key. Acyclicity is also important.
            return aig_check_validity(g_eval_nx) and nx.is_directed_acyclic_graph(g_eval_nx)

        # Prepare canonical training graphs for novelty comparison (DiGraphs)
        # This conversion should ideally happen once, perhaps in __init__ if train_graphs don't change
        canonical_train_graphs_nx_di: List[nx.DiGraph] = []
        if self.train_graphs:  # self.train_graphs are already nx.Graph (potentially DiGraph from AIG validation)
            for g_train_ref in self.train_graphs:
                # Ensure they are DiGraphs for isomorphism checks with generated DiGraphs
                if isinstance(g_train_ref, nx.DiGraph):
                    canonical_train_graphs_nx_di.append(g_train_ref)
                else:  # If they were stored as UndirectedGraph, convert preserving types
                    temp_g_dir = nx.DiGraph()
                    temp_g_dir.add_nodes_from(g_train_ref.nodes(data=True))
                    for u, v, data_edge in g_train_ref.edges(data=True):
                        edge_attrs_copy = data_edge.copy()
                        # Assuming canonical direction (smaller to larger) was already handled by convert_pyg_to_nx_for_aig_validation
                        src_node, tgt_node = (u, v)  # Or apply min/max if not guaranteed
                        if not temp_g_dir.has_edge(src_node, tgt_node):
                            temp_g_dir.add_edge(src_node, tgt_node, **edge_attrs_copy)
                    canonical_train_graphs_nx_di.append(temp_g_dir)

        # Filter out empty graphs for evaluation
        eval_generated_graphs_nx = [g for g in generated_graphs_nx_directed if g.number_of_nodes() > 0]
        eval_train_graphs_nx = [g for g in canonical_train_graphs_nx_di if g.number_of_nodes() > 0]

        frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = 0.0, 0.0, 0.0
        frac_non_iso_to_train = 0.0  # Novelty

        if not eval_generated_graphs_nx:
            if local_rank == 0:
                print("AIGMetrics: No valid generated graphs for uniqueness/novelty checks.")
        elif not eval_train_graphs_nx:  # If no training graphs, all non-isomorphic generated are novel
            if local_rank == 0:
                print(
                    "AIGMetrics: No training graphs for novelty comparison. Calculating uniqueness/validity of generated set.")
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(
                    eval_generated_graphs_nx,
                    [],  # No train graphs to compare against for novelty within this call
                    validity_func=combined_aig_validity_for_eval_fractions,
                    is_canonical=True  # Assuming convert_raw_model_output_to_nx_aig produces canonical graphs
                )
            frac_non_iso_to_train = 1.0  # All valid unique non-iso are considered novel
        else:
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(
                    eval_generated_graphs_nx,
                    eval_train_graphs_nx,  # Pass train graphs for novelty calculation
                    validity_func=combined_aig_validity_for_eval_fractions,
                    is_canonical=True
                )
            # eval_fraction_isomorphic calculates fraction of gen_graphs isomorphic to ANY train_graph
            # So, 1.0 - this_fraction = fraction of generated graphs NOT isomorphic to ANY train graph (i.e., novel)
            # This should be applied to the set of unique, valid generated graphs for a more meaningful novelty.

            # Let's get unique, valid, non-isomorphic generated graphs first
            unique_valid_non_iso_generated = []
            if eval_generated_graphs_nx:
                unique_graphs_set = []
                for g in eval_generated_graphs_nx:
                    is_new = True
                    for ug in unique_graphs_set:
                        if nx.is_isomorphic(g, ug, node_match=lambda n1, n2: n1['type'] == n2['type'],
                                            edge_match=lambda e1, e2: e1['type'] == e2['type']):
                            is_new = False
                            break
                    if is_new:
                        if combined_aig_validity_for_eval_fractions(g):
                            unique_graphs_set.append(g)
                unique_valid_non_iso_generated = unique_graphs_set

            if unique_valid_non_iso_generated:
                isomorphic_to_train_count = 0
                for gen_g in unique_valid_non_iso_generated:
                    for train_g in eval_train_graphs_nx:
                        if nx.is_isomorphic(gen_g, train_g, node_match=lambda n1, n2: n1['type'] == n2['type'],
                                            edge_match=lambda e1, e2: e1['type'] == e2['type']):
                            isomorphic_to_train_count += 1
                            break  # Found a match in training set
                frac_non_iso_to_train = 1.0 - (isomorphic_to_train_count / len(unique_valid_non_iso_generated))
            else:
                frac_non_iso_to_train = 0.0

        to_log.update({
            'sampling_quality/frac_unique_aigs': frac_unique,
            'sampling_quality/frac_unique_non_iso_aigs': frac_unique_non_iso,
            'sampling_quality/frac_unique_non_iso_valid_aigs': frac_unique_non_iso_valid,  # Renamed for clarity
            'sampling_quality/frac_novel_aigs': frac_non_iso_to_train  # Renamed for clarity
        })

        # MODIFIED: Add ratio computation using ref_metrics if available
        if ref_metrics:
            # Determine which set of reference metrics to use (val or test)
            current_ref_set = ref_metrics.get("test" if test else "val")
            if current_ref_set:
                # Define which keys from 'to_log' should have ratios computed
                # These should be metrics that are also present in the reference set
                ratio_keys = [
                    'aig_metrics/structural_validity_fraction',
                    'aig_metrics/acyclicity_fraction',
                    'graph_stats/degree_mmd_undirected',  # Ensure this is comparable
                    'sampling_quality/frac_unique_non_iso_valid_aigs'
                ]
                # Filter ratio_keys to only those present in both to_log and current_ref_set
                valid_ratio_keys = [k for k in ratio_keys if k in to_log and k in current_ref_set]

                ratios = compute_ratios(
                    gen_metrics=to_log,
                    ref_metrics=current_ref_set,
                    metrics_keys=valid_ratio_keys
                )
                to_log.update(ratios)
            elif local_rank == 0:
                warnings.warn(f"AIGMetrics: ref_metrics provided, but no matching set for {'test' if test else 'val'}")

        # Log all computed metrics to wandb
        if wandb.run and local_rank == 0:
            # Prefix with "Val_" or "Test_" for clarity in wandb
            wandb_prefix = f"{'Test' if test else 'Val'}"
            wandb_log_data = {f"{wandb_prefix}_{k.replace('/', '_')}": v for k, v in to_log.items()}
            wandb.log(wandb_log_data, step=current_epoch, commit=True)  # Use epoch as step

        # Print metrics to console
        if local_rank == 0:
            print(f"AIGMetrics Epoch {current_epoch} ({'Test' if test else 'Val'}, Counter: {val_counter}):")
            for key, val in to_log.items():
                print(f"  {key}: {val:.4f}")
        return to_log

    def reset(self):
        """Resets any state if necessary (e.g., for iterative validation)."""
        # Call superclass reset if it exists and does something.
        if hasattr(super(), 'reset') and callable(super().reset):
            super().reset()
        # No AIG-specific state to reset in this version beyond what super might handle.
        pass
