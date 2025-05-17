import torch
import torch.nn as nn  # Often in context, kept for completeness if other parts rely on it
import networkx as nx
from tqdm import tqdm
import wandb  # If you use wandb
import numpy as np
from typing import Union, List, Dict, Any  # Added Dict, Any for broader type hinting
import warnings

# DiGress existing utilities / Spectre-based metrics
# Ensure these are correctly imported from your DiGress project structure
from src.analysis.spectre_utils import SpectreSamplingMetrics, degree_stats, \
    eval_fraction_isomorphic, eval_fraction_unique_non_isomorphic_valid

# --- AIG-specific imports - Ensure these paths are correct ---
from src.aig_config import check_validity as aig_check_validity
from src.aig_config import NODE_TYPE_KEYS, EDGE_TYPE_KEYS, \
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES

# This should be the updated function that returns nx.Graph
# (as modified in our previous exchange if that was applied)
from src.datasets.aig_custom_dataset import convert_pyg_to_nx_for_aig_validation


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
        node_features_tensor: Tensor of shape (max_nodes_in_batch, NUM_NODE_FEATURES).
                              Node features from the model's sampling output.
        edge_index_tensor: Tensor of shape (2, num_edges_in_prediction).
                           Edge index from the model's sampling output. This might contain
                           symmetric pairs (e.g., (u,v) and (v,u)) if the model's
                           internal representation was symmetric.
        edge_features_tensor: Tensor of shape (num_edges_in_prediction, NUM_EDGE_FEATURES + 1).
                              Edge features from the model's sampling output.
        num_nodes_int: The actual number of nodes for this specific graph in the batch.

    Returns:
        A NetworkX DiGraph with 'type' attributes and canonical edge direction,
        or None if conversion fails or inputs are invalid.
    """
    # Create a DiGraph to store canonically directed edges
    nx_graph = nx.DiGraph()

    # Validate inputs
    if not (0 <= num_nodes_int <= node_features_tensor.shape[0]):
        warnings.warn(
            f"Convert Model Output: num_nodes_int ({num_nodes_int}) is out of bounds "
            f"for node_features_tensor shape ({node_features_tensor.shape[0]}). Skipping graph."
        )
        return None
    if node_features_tensor.ndim != 2 or node_features_tensor.shape[1] != NUM_NODE_FEATURES:
        warnings.warn(
            f"Convert Model Output: Node features tensor has incorrect dimensions. "
            f"Shape: {node_features_tensor.shape}, Expected: (N, {NUM_NODE_FEATURES}). Skipping graph."
        )
        return None

    # Process nodes up to num_nodes_int
    for i in range(num_nodes_int):
        node_feature_vector = node_features_tensor[i].cpu().numpy()
        # Check for one-hot encoding
        if not (np.isclose(np.sum(node_feature_vector), 1.0) and
                np.all((np.isclose(node_feature_vector, 0.0)) | (np.isclose(node_feature_vector, 1.0)))):
            node_type_str = "UNKNOWN_TYPE_NON_ONE_HOT"  # More specific than just UNKNOWN_TYPE_ATTRIBUTE
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
    expected_edge_feature_dim = NUM_EDGE_FEATURES + 1  # Model outputs this many channels

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

            # Ensure nodes in edge_index are within the valid range for this graph
            if not (0 <= raw_src_node < num_nodes_int and 0 <= raw_tgt_node < num_nodes_int):
                warnings.warn(
                    f"Convert Model Output: Edge ({raw_src_node} -> {raw_tgt_node}) contains node IDs "
                    f"out of bounds for num_nodes ({num_nodes_int}). Skipping this edge."
                )
                continue

            edge_feature_vector_model = edge_features_tensor[i].cpu().numpy()

            # Check for one-hot encoding (model output after softmax should be probabilities, argmax gives one-hot choice)
            if not (np.isclose(np.sum(edge_feature_vector_model), 1.0) and
                    np.all(
                        (np.isclose(edge_feature_vector_model, 0.0)) | (np.isclose(edge_feature_vector_model, 1.0)))):
                # This might occur if the input `edge_features_tensor` is not one-hot (e.g. raw logits or probabilities)
                # For DiGress, the sampling step `sample_discrete_features` followed by `F.one_hot` should ensure this.
                # If this warning appears, it indicates an issue in how `edge_features_tensor` was produced.
                warnings.warn(
                    f"Convert Model Output: Edge ({raw_src_node}-{raw_tgt_node}) feature vector from model "
                    f"is not one-hot: {edge_feature_vector_model}. Argmax will still be used."
                )
                # Proceeding with argmax as per original logic

            shifted_type_index = np.argmax(edge_feature_vector_model)
            edge_type_str: str

            if shifted_type_index == 0:
                # This channel is for "no specific AIG type" or potentially "no edge" if the model uses it that way.
                # For AIGs, an edge usually has a specific type (REG/INV).
                # If this means "no actual AIG edge", we might skip adding it.
                # However, to be consistent with how UNKNOWN_TYPE_ATTRIBUTE is handled,
                # we can assign a generic type or skip. Skipping seems more appropriate if channel 0 means "no edge".
                # For now, let's assume if an edge is in edge_index, it's intended to exist,
                # and channel 0 means "generic" or "unknown AIG type".
                edge_type_str = "EDGE_GENERIC_OR_PADDING"  # A more descriptive name
                # warnings.warn(f"Convert Model Output: Edge ({raw_src_node}-{raw_tgt_node}) mapped to channel 0 (generic/padding).")
            else:
                actual_aig_type_index = shifted_type_index - 1  # Convert from model output channel to actual AIG type index
                if not (0 <= actual_aig_type_index < len(EDGE_TYPE_KEYS)):
                    edge_type_str = "UNKNOWN_TYPE_BAD_INDEX"
                    warnings.warn(
                        f"Convert Model Output: Edge ({raw_src_node}-{raw_tgt_node}) decoded to invalid "
                        f"actual_aig_type_index {actual_aig_type_index} from shifted index {shifted_type_index}."
                    )
                else:
                    edge_type_str = EDGE_TYPE_KEYS[actual_aig_type_index]

            # Enforce smaller_id -> larger_id direction for canonical DiGraph
            src_final = min(raw_src_node, raw_tgt_node)
            tgt_final = max(raw_src_node, raw_tgt_node)

            if src_final == tgt_final:
                # Handle self-loops if they are meaningful for AIGs and should be preserved.
                # AIGs typically do not have self-loops on AND gates.
                # If they are not allowed, aig_check_validity should catch them.
                # For now, if the model produces it, we add it.
                if not nx_graph.has_edge(src_final, tgt_final):  # Add only if not already added
                    nx_graph.add_edge(src_final, tgt_final, type=edge_type_str)
                # else: if self-loop already exists, attributes might be overwritten if different.
            elif not nx_graph.has_edge(src_final, tgt_final):
                # Add the edge only if this canonical directed edge doesn't exist yet.
                # This handles cases where edge_index_tensor might have both (u,v) and (v,u)
                # due to the model operating on an undirected representation.
                nx_graph.add_edge(src_final, tgt_final, type=edge_type_str)
            # If the edge (src_final, tgt_final) was already added (e.g., from processing (raw_tgt_node, raw_src_node)
            # if it was also in edge_index_tensor), its attributes would be overwritten if different.
            # However, for a symmetric model output, attributes for (u,v) and (v,u) should be the same.

    return nx_graph


class AIGSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule: Any):  # Using Any for datamodule for broader compatibility
        # Initialize local_rank from the datamodule's config
        self.local_rank = datamodule.cfg.general.get('local_rank', 0)

        super().__init__(datamodule=datamodule,
                         compute_emd=False,  # AIGs typically don't use EMD for these structural/graph stats
                         metrics_list=['aig_structural_validity', 'aig_acyclicity', 'degree'])

        # Initialize graph lists; loader_to_nx will populate them.
        # These will store nx.Graph objects as per the modified convert_pyg_to_nx_for_aig_validation
        self.train_graphs: List[nx.Graph] = []
        self.val_graphs: List[nx.Graph] = []
        self.test_graphs: List[nx.Graph] = []

        # Load reference graphs
        if hasattr(datamodule, 'train_dataloader') and datamodule.train_dataloader() is not None:
            self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        if hasattr(datamodule, 'val_dataloader') and datamodule.val_dataloader() is not None:
            self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        if hasattr(datamodule, 'test_dataloader') and datamodule.test_dataloader() is not None:
            self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())

    def loader_to_nx(self, loader: Any) -> List[nx.Graph]:
        """
        Converts a PyTorch Geometric DataLoader's batches into a list of NetworkX Graphs
        using the AIG-specific validation conversion function.
        `convert_pyg_to_nx_for_aig_validation` is assumed to return nx.Graph objects.
        """
        networkx_graphs: List[nx.Graph] = []

        if loader is None or not hasattr(loader, 'dataset') or loader.dataset is None or len(loader.dataset) == 0:
            if self.local_rank == 0:
                warnings.warn(
                    "AIGSamplingMetrics.loader_to_nx: Loader or its dataset is empty/invalid. Cannot load reference graphs.")
            return networkx_graphs

        # Disable tqdm if not on rank 0 for cleaner logs in distributed training
        for i, batch in enumerate(tqdm(loader, desc="Loading Reference Graphs for Metrics",
                                       disable=(self.local_rank != 0))):
            data_list = batch.to_data_list()
            for j, pyg_data in enumerate(data_list):
                # This function (as modified previously) should return nx.Graph
                nx_graph = convert_pyg_to_nx_for_aig_validation(pyg_data)
                if nx_graph is not None:
                    networkx_graphs.append(nx_graph)
        return networkx_graphs

    def forward(self, generated_graphs_raw: list, name: str, current_epoch: int, val_counter: int,
                local_rank: int,  # local_rank passed from the trainer/model callback
                test: bool = False) -> Dict[str, float]:
        """
        Calculates and logs metrics for generated AIGs.

        Args:
            generated_graphs_raw: A list of raw graph data tuples/lists from the model's sampling.
                                  Expected structure per item:
                                  (node_features_tensor, edge_index_tensor, edge_features_tensor, num_nodes_tensor)
            name: Name of the run/experiment.
            current_epoch: Current training epoch.
            val_counter: Validation counter.
            local_rank: The local rank of the current process (for distributed training).
            test: Boolean flag indicating if this is a test phase.

        Returns:
            A dictionary of calculated metrics.
        """
        # Reference graphs are List[nx.Graph]
        reference_graphs_nx: List[nx.Graph] = self.test_graphs if test else self.val_graphs

        # Convert raw model outputs to canonical DiGraphs (smaller_id -> larger_id)
        generated_graphs_nx_directed: List[nx.DiGraph] = []
        for i, graph_raw_data_item in enumerate(
                tqdm(generated_graphs_raw, desc="Converting Generated Graphs for Metrics", disable=(local_rank != 0))):
            if not (isinstance(graph_raw_data_item, (list, tuple)) and len(graph_raw_data_item) == 4):
                warnings.warn(
                    f"AIGMetrics: Item {i} in generated_graphs_raw has unexpected format. "
                    f"Expected 4 elements, got {len(graph_raw_data_item)}. Skipping this graph."
                )
                continue

            node_features, edge_index, edge_features_model, num_nodes_tensor = graph_raw_data_item

            if not isinstance(num_nodes_tensor, torch.Tensor) or num_nodes_tensor.numel() != 1:
                warnings.warn(
                    f"AIGMetrics: Item {i} num_nodes_tensor is not a scalar tensor. Skipping this graph."
                )
                continue
            num_nodes_int = num_nodes_tensor.item()

            # This function now returns a DiGraph with smaller_id -> larger_id edges
            nx_di_graph = convert_raw_model_output_to_nx_aig(
                node_features, edge_index, edge_features_model, num_nodes_int
            )
            if nx_di_graph is not None:
                generated_graphs_nx_directed.append(nx_di_graph)

        if not generated_graphs_nx_directed:
            if local_rank == 0:
                print(
                    "AIGMetrics: No generated graphs were successfully converted to NetworkX DiGraphs. Skipping metrics calculation.")
            return {}

        to_log: Dict[str, float] = {}

        # 1. AIG Structural Validity
        # `aig_check_validity` (as modified previously) handles nx.Graph or nx.DiGraph.
        # Since generated_graphs_nx_directed contains canonical DiGraphs, it's directly compatible.
        if 'aig_structural_validity' in self.metrics_list:
            num_structurally_valid = sum(1 for g in generated_graphs_nx_directed if aig_check_validity(g))
            structural_validity_fraction = num_structurally_valid / len(
                generated_graphs_nx_directed) if generated_graphs_nx_directed else 0.0
            to_log['aig_metrics/structural_validity_fraction'] = structural_validity_fraction
            if wandb.run and local_rank == 0:
                wandb.summary[f'{name}_aig_structural_validity_fraction'] = structural_validity_fraction

        # 2. Acyclicity (expects DiGraph)
        if 'aig_acyclicity' in self.metrics_list:
            num_acyclic = sum(1 for g in generated_graphs_nx_directed if nx.is_directed_acyclic_graph(g))
            acyclicity_fraction = num_acyclic / len(
                generated_graphs_nx_directed) if generated_graphs_nx_directed else 0.0
            to_log['aig_metrics/acyclicity_fraction'] = acyclicity_fraction
            if wandb.run and local_rank == 0:
                wandb.summary[f'{name}_aig_acyclicity_fraction'] = acyclicity_fraction

        # 3. Degree Stats (MMD for undirected degrees)
        if 'degree' in self.metrics_list:
            # reference_graphs_nx are List[nx.Graph] (undirected)
            # generated_graphs_nx_directed are List[nx.DiGraph] (canonically directed)
            # For degree MMD, convert generated DiGraphs to undirected to compare with reference undirected graphs.

            # Ensure reference graphs are loaded for the correct split
            current_ref_graphs = self.test_graphs if test else self.val_graphs

            if current_ref_graphs and generated_graphs_nx_directed:
                # Reference graphs are already nx.Graph (undirected)
                ref_undirected_for_degree = [g for g in current_ref_graphs if g.number_of_nodes() > 0]
                # Convert generated canonical DiGraphs to undirected for degree comparison
                gen_undirected_for_degree = [g.to_undirected(as_view=False) for g in generated_graphs_nx_directed if
                                             g.number_of_nodes() > 0]

                if ref_undirected_for_degree and gen_undirected_for_degree:
                    degree_mmd = degree_stats(ref_undirected_for_degree, gen_undirected_for_degree,
                                              is_parallel=True,
                                              # Consider setting based on actual parallelism needs/setup
                                              compute_emd=self.compute_emd)  # self.compute_emd is False by default
                    to_log['graph_stats/degree_mmd_undirected'] = degree_mmd
                    if wandb.run and local_rank == 0:
                        wandb.summary[f'{name}_degree_mmd_undirected'] = degree_mmd
                else:
                    to_log['graph_stats/degree_mmd_undirected'] = -1.0  # Indicate missing data for calc
            else:
                to_log['graph_stats/degree_mmd_undirected'] = -1.0  # Indicate missing data for calc

        # 4. Uniqueness, Isomorphism, Novelty
        # For these, we need to compare canonical DiGraphs with other canonical DiGraphs.
        # `generated_graphs_nx_directed` are already canonical DiGraphs.
        # `self.train_graphs` (from loader_to_nx) are nx.Graph objects. Convert them.

        def combined_aig_validity_for_eval_fractions(g_eval_nx: nx.DiGraph) -> bool:
            # This function will receive canonical DiGraphs.
            return aig_check_validity(g_eval_nx)

        # Convert train_graphs (List[nx.Graph]) to canonical DiGraphs for isomorphism checks
        canonical_train_graphs_nx_di: List[nx.DiGraph] = []
        if self.train_graphs:  # self.train_graphs is List[nx.Graph]
            for g_undir in self.train_graphs:
                temp_g_dir = nx.DiGraph()
                temp_g_dir.add_nodes_from(g_undir.nodes(data=True))
                for u, v, data_edge in g_undir.edges(data=True):  # Iterate undirected edges
                    edge_attrs_copy = data_edge.copy()
                    # Apply canonical direction
                    src_node, tgt_node = (u, v) if u < v else (v, u)
                    if not temp_g_dir.has_edge(src_node, tgt_node):  # Ensure unique directed edges
                        temp_g_dir.add_edge(src_node, tgt_node, **edge_attrs_copy)
                canonical_train_graphs_nx_di.append(temp_g_dir)

        # Filter out empty graphs before passing to eval functions
        eval_generated_graphs_nx = [g for g in generated_graphs_nx_directed if g.number_of_nodes() > 0]
        eval_train_graphs_nx = [g for g in canonical_train_graphs_nx_di if g.number_of_nodes() > 0]

        frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = 0.0, 0.0, 0.0
        frac_non_iso_to_train = 0.0  # Novelty

        if not eval_generated_graphs_nx:
            # No generated graphs to evaluate
            if local_rank == 0:
                print("AIGMetrics: No valid generated graphs for uniqueness/novelty checks.")
        elif not eval_train_graphs_nx:
            # No training graphs to compare against for novelty, but can check uniqueness/validity of generated
            if local_rank == 0:
                print(
                    "AIGMetrics: No training graphs for novelty comparison. Calculating uniqueness/validity of generated set.")
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(
                    eval_generated_graphs_nx,
                    [],  # Pass empty list for train_graphs
                    validity_func=combined_aig_validity_for_eval_fractions
                )
            frac_non_iso_to_train = 1.0  # All are considered novel if no training set to compare
        else:
            # Both generated and training graphs are available
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(
                    eval_generated_graphs_nx,
                    eval_train_graphs_nx,  # Compare against canonical training DiGraphs
                    validity_func=combined_aig_validity_for_eval_fractions
                )

            # Novelty: fraction of generated graphs not isomorphic to any in the training set
            frac_non_iso_to_train = 1.0 - eval_fraction_isomorphic(
                eval_generated_graphs_nx,
                eval_train_graphs_nx
            )

        to_log.update({
            'sampling_quality/frac_unique_aigs': frac_unique,
            'sampling_quality/frac_unique_non_iso_aigs': frac_unique_non_iso,
            'sampling_quality/frac_unique_non_iso_structurally_valid_aigs': frac_unique_non_iso_valid,
            'sampling_quality/frac_non_iso_to_train_aigs': frac_non_iso_to_train  # Novelty
        })

        if wandb.run and local_rank == 0:
            # Log all metrics collected in to_log
            wandb_log_data = {f"{name}_{k.replace('/', '_')}": v for k, v in to_log.items()}
            wandb.log(wandb_log_data, commit=True)  # Ensure commit=True if this is the final log for the step

        if local_rank == 0:
            print(f"AIGMetrics Epoch {current_epoch} ({'Test' if test else 'Val'}):")
            for key, val in to_log.items():
                print(f"  {key}: {val:.4f}")

        return to_log

    def reset(self):
        """Resets any stateful metrics if necessary."""
        super().reset() if hasattr(super(), 'reset') else None
        # The reference graphs (self.train_graphs, etc.) are loaded once in __init__.
        # If this reset is part of a new validation/test phase where the datamodule
        # might have changed or needs reloading, that logic would need to be added here
        # or __init__ would need to be recalled. For typical epoch-based validation,
        # keeping them loaded is fine.
        pass


