import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from tqdm import tqdm
import wandb
import numpy as np
from typing import Union, List, Dict, Any
import warnings
from collections import Counter
import concurrent.futures  # For parallel degree calculation
from functools import partial  # For parallel degree calculation

# DiGress existing utilities / Spectre-based metrics
from src.analysis.spectre_utils import SpectreSamplingMetrics, \
    eval_fraction_isomorphic, eval_fraction_unique_non_isomorphic_valid
# Import the MMD computation function
from src.analysis.dist_helper import compute_mmd, gaussian_emd  # Or your preferred kernel

# --- AIG-specific imports - Ensure these paths are correct ---
# Make sure VALIDITY_ERROR_KEYS is defined in and imported from aig_config
from src.aig_config import NODE_TYPE_KEYS, EDGE_TYPE_KEYS, \
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES, VALIDITY_ERROR_KEYS
# Import your modified check_validity function
from src.aig_config import check_validity as aig_check_validity_with_code

from src.datasets.aig_custom_dataset import convert_pyg_to_nx_for_aig_validation


# --- Helper functions for Directed Degree Statistics ---
# Ideally, these would go into src/analysis/spectre_utils.py or src/analysis/dist_helper.py

def directed_degree_worker(G_directed: nx.DiGraph) -> Dict[str, np.ndarray]:
    """
    Calculates in-degree and out-degree histograms for a directed graph.
    Returns a dictionary with 'in_degree_hist' and 'out_degree_hist'.
    """
    if not isinstance(G_directed, nx.DiGraph):
        # This case should ideally not be hit if inputs are correct
        warnings.warn("directed_degree_worker received a non-DiGraph. Attempting conversion.")
        G_directed = nx.DiGraph(G_directed)

    in_degrees = [d for _, d in G_directed.in_degree()]
    out_degrees = [d for _, d in G_directed.out_degree()]

    max_in_degree = 0
    if in_degrees:
        max_in_degree = np.max(in_degrees)

    max_out_degree = 0
    if out_degrees:
        max_out_degree = np.max(out_degrees)

    # np.arange creates bins up to max_degree + 1, so histogram covers up to max_degree
    # Bins are [0,1), [1,2), ..., [max_degree, max_degree+1)
    # The length of the histogram will be max_degree + 1
    in_hist = np.histogram(in_degrees, bins=np.arange(0, max_in_degree + 2), density=False)[0]
    out_hist = np.histogram(out_degrees, bins=np.arange(0, max_out_degree + 2), density=False)[0]

    return {'in_degree_hist': in_hist, 'out_degree_hist': out_hist}


def directed_degree_stats_mmd(graph_ref_list: List[nx.DiGraph],
                              graph_pred_list: List[nx.DiGraph],
                              kernel_func=gaussian_emd,  # Pass the actual kernel function
                              is_parallel=True,
                              **kernel_kwargs) -> Dict[str, float]:
    """
    Computes MMD for in-degree and out-degree distributions between two lists of directed graphs.
    """
    sample_ref_in_hists = []
    sample_ref_out_hists = []
    sample_pred_in_hists = []
    sample_pred_out_hists = []

    # Filter out empty graphs to prevent errors in degree calculation
    graph_ref_list_non_empty = [G for G in graph_ref_list if G.number_of_nodes() > 0]
    graph_pred_list_non_empty = [G for G in graph_pred_list if G.number_of_nodes() > 0]

    if not graph_ref_list_non_empty or not graph_pred_list_non_empty:
        warnings.warn("One or both graph lists are empty after filtering, cannot compute directed degree MMDs.")
        return {'in_degree_mmd': -1.0, 'out_degree_mmd': -1.0}

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for res_dict in executor.map(directed_degree_worker, graph_ref_list_non_empty):
                sample_ref_in_hists.append(res_dict['in_degree_hist'])
                sample_ref_out_hists.append(res_dict['out_degree_hist'])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for res_dict in executor.map(directed_degree_worker, graph_pred_list_non_empty):
                sample_pred_in_hists.append(res_dict['in_degree_hist'])
                sample_pred_out_hists.append(res_dict['out_degree_hist'])
    else:
        for G_ref in graph_ref_list_non_empty:
            hists = directed_degree_worker(G_ref)
            sample_ref_in_hists.append(hists['in_degree_hist'])
            sample_ref_out_hists.append(hists['out_degree_hist'])
        for G_pred in graph_pred_list_non_empty:
            hists = directed_degree_worker(G_pred)
            sample_pred_in_hists.append(hists['in_degree_hist'])
            sample_pred_out_hists.append(hists['out_degree_hist'])

    mmd_in_degree = -1.0
    mmd_out_degree = -1.0

    if sample_ref_in_hists and sample_pred_in_hists:  # Ensure lists are not empty
        mmd_in_degree = compute_mmd(sample_ref_in_hists, sample_pred_in_hists, kernel=kernel_func, **kernel_kwargs)
    if sample_ref_out_hists and sample_pred_out_hists:  # Ensure lists are not empty
        mmd_out_degree = compute_mmd(sample_ref_out_hists, sample_pred_out_hists, kernel=kernel_func, **kernel_kwargs)

    return {'in_degree_mmd': mmd_in_degree, 'out_degree_mmd': mmd_out_degree}


# --- End of Helper functions ---


def convert_raw_model_output_to_nx_aig(node_features_tensor: torch.Tensor,
                                       edge_index_tensor: torch.Tensor,
                                       edge_features_tensor: torch.Tensor,
                                       num_nodes_int: int) -> Union[nx.DiGraph, None]:
    nx_graph = nx.DiGraph()

    if not (isinstance(node_features_tensor, torch.Tensor) and
            node_features_tensor.ndim == 2 and
            node_features_tensor.shape[0] == num_nodes_int and
            node_features_tensor.shape[1] == NUM_NODE_FEATURES):
        # warnings.warn(f"Convert Model Output (to NX): Node features tensor incorrect format.") # Reduced verbosity
        return None

    for i in range(num_nodes_int):
        node_feature_vector = node_features_tensor[i].cpu().numpy()
        if not (np.isclose(np.sum(node_feature_vector), 1.0) and
                np.all((np.isclose(node_feature_vector, 0.0)) | (np.isclose(node_feature_vector, 1.0)))):
            node_type_str = "UNKNOWN_TYPE_NON_ONE_HOT"
        else:
            type_index = np.argmax(node_feature_vector)
            if not (0 <= type_index < len(NODE_TYPE_KEYS)):
                node_type_str = "UNKNOWN_TYPE_BAD_INDEX"
            else:
                node_type_str = NODE_TYPE_KEYS[type_index]
        nx_graph.add_node(i, type=node_type_str)

    num_edges_in_prediction = edge_index_tensor.shape[1]
    expected_edge_feature_dim = NUM_EDGE_FEATURES + 1

    if num_edges_in_prediction > 0:
        if not (isinstance(edge_index_tensor, torch.Tensor) and edge_index_tensor.ndim == 2 and edge_index_tensor.shape[
            0] == 2):
            # warnings.warn(f"Convert Model Output (to NX): Edge index tensor incorrect dimensions.")
            return None
        if not (isinstance(edge_features_tensor, torch.Tensor) and edge_features_tensor.ndim == 2 and
                edge_features_tensor.shape[0] == num_edges_in_prediction and
                edge_features_tensor.shape[1] == expected_edge_feature_dim):
            # warnings.warn(f"Convert Model Output (to NX): Edge features tensor incorrect shape.")
            return None

        for i in range(num_edges_in_prediction):
            raw_src_node = edge_index_tensor[0, i].item()
            raw_tgt_node = edge_index_tensor[1, i].item()

            if not (0 <= raw_src_node < num_nodes_int and 0 <= raw_tgt_node < num_nodes_int):
                continue

            edge_feature_vector_one_hot = edge_features_tensor[i].cpu().numpy()
            shifted_type_index = np.argmax(edge_feature_vector_one_hot)

            if shifted_type_index == 0:
                continue  # Class 0 is "EDGE_GENERIC_OR_PADDING" or "no edge", so don't add
            else:
                actual_aig_type_index = shifted_type_index - 1
                if not (0 <= actual_aig_type_index < len(EDGE_TYPE_KEYS)):
                    edge_type_str = "UNKNOWN_EDGE_TYPE_BAD_INDEX"
                else:
                    edge_type_str = EDGE_TYPE_KEYS[actual_aig_type_index]

            if not nx_graph.has_edge(raw_src_node, raw_tgt_node):
                nx_graph.add_edge(raw_src_node, raw_tgt_node, type=edge_type_str)
    return nx_graph


class AIGSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule: Any):
        nn.Module.__init__(self)
        self.compute_emd = datamodule.cfg.model.get('compute_emd_metrics', False)
        self.metrics_list = ['aig_structural_validity', 'aig_acyclicity', 'degree', 'aig_validity_breakdown']
        self.local_rank = datamodule.cfg.general.get('local_rank', 0)
        self.train_graphs: List[nx.DiGraph] = []
        self.val_graphs: List[nx.DiGraph] = []
        self.test_graphs: List[nx.DiGraph] = []

        # Load reference graphs
        if hasattr(datamodule, 'train_dataloader') and datamodule.train_dataloader() is not None:
            if self.local_rank == 0: print("AIGSamplingMetrics: Loading training reference graphs...")
            self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        if hasattr(datamodule, 'val_dataloader') and datamodule.val_dataloader() is not None:
            if self.local_rank == 0: print("AIGSamplingMetrics: Loading validation reference graphs...")
            self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        if hasattr(datamodule, 'test_dataloader') and datamodule.test_dataloader() is not None:
            if self.local_rank == 0: print("AIGSamplingMetrics: Loading test reference graphs...")
            self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())

    def loader_to_nx(self, loader: Any) -> List[nx.DiGraph]:
        networkx_graphs: List[nx.DiGraph] = []
        if loader is None or not hasattr(loader, 'dataset') or loader.dataset is None or len(loader.dataset) == 0:
            if self.local_rank == 0:
                warnings.warn(
                    "AIGSamplingMetrics.loader_to_nx: Loader or its dataset is empty/invalid. Cannot load reference graphs.")
            return networkx_graphs

        tqdm_desc = "Loading Reference Graphs for Metrics (AIG)"
        disable_tqdm = self.local_rank != 0

        for i, batch in enumerate(tqdm(loader, desc=tqdm_desc, disable=disable_tqdm)):
            data_list = batch.to_data_list()
            for j, pyg_data in enumerate(data_list):
                nx_graph = convert_pyg_to_nx_for_aig_validation(pyg_data)
                if nx_graph is not None:
                    networkx_graphs.append(nx_graph)
        return networkx_graphs

    def forward(self, generated_graphs_raw: list, name: str, current_epoch: int, val_counter: int,
                local_rank: int, test: bool = False) -> Dict[str, float]:

        reference_graphs_nx: List[nx.DiGraph] = self.test_graphs if test else self.val_graphs
        generated_graphs_nx_directed: List[nx.DiGraph] = []

        graphs_mostly_padding_edges = 0
        actual_aig_edges_formed = 0

        disable_tqdm_conversion = local_rank != 0

        for i, graph_raw_data_item in enumerate(
                tqdm(generated_graphs_raw, desc="Converting Generated Graphs for Metrics (AIG)",
                     disable=disable_tqdm_conversion)):

            if not (isinstance(graph_raw_data_item, (list, tuple)) and len(graph_raw_data_item) == 2):
                if local_rank == 0: warnings.warn(f"AIGMetrics (Graph {i}): Invalid raw data format. Skipping.")
                continue
            node_indices_tensor, edge_indices_matrix = graph_raw_data_item
            if not isinstance(node_indices_tensor, torch.Tensor) or not isinstance(edge_indices_matrix, torch.Tensor):
                if local_rank == 0: warnings.warn(f"AIGMetrics (Graph {i}): Data not in tensor format. Skipping.")
                continue
            if node_indices_tensor.ndim == 0 or node_indices_tensor.shape[0] == 0:  # Check for 0-dim or 0-size
                if local_rank == 0: warnings.warn(
                    f"AIGMetrics (Graph {i}): Graph has 0 nodes or scalar node tensor. Skipping.")
                continue

            num_nodes_int = node_indices_tensor.shape[0]

            if node_indices_tensor.is_floating_point() or node_indices_tensor.dtype not in [torch.long, torch.int]:
                node_indices_tensor = node_indices_tensor.long()

            actual_node_features_one_hot = F.one_hot(node_indices_tensor, num_classes=NUM_NODE_FEATURES).float()

            adj_edges_list = []
            adj_edge_features_one_hot_list = []
            expected_edge_feature_output_dim = NUM_EDGE_FEATURES + 1

            num_specific_edges_in_graph = 0
            total_possible_edges_in_graph = num_nodes_int * (num_nodes_int - 1) if num_nodes_int > 1 else 0

            for u_idx in range(num_nodes_int):
                for v_idx in range(num_nodes_int):
                    if u_idx == v_idx: continue
                    edge_class_idx = edge_indices_matrix[u_idx, v_idx].item()
                    if edge_class_idx != 0:
                        num_specific_edges_in_graph += 1
                        actual_aig_edges_formed += 1
                        adj_edges_list.append([u_idx, v_idx])
                        if not (0 <= edge_class_idx < expected_edge_feature_output_dim):
                            if local_rank == 0:  # Log only on rank 0 to avoid spam
                                warnings.warn(
                                    f"AIGMetrics (Graph {i}, Edge {u_idx}->{v_idx}): Invalid edge_class_idx {edge_class_idx}. Defaulting to padding-like one-hot.")
                            edge_one_hot_np = np.zeros(expected_edge_feature_output_dim)
                            edge_one_hot_np[0] = 1.0
                            edge_one_hot = torch.tensor(edge_one_hot_np,
                                                        device=actual_node_features_one_hot.device).float()
                        else:
                            edge_one_hot = F.one_hot(
                                torch.tensor(edge_class_idx, device=actual_node_features_one_hot.device).long(),
                                num_classes=expected_edge_feature_output_dim).float()
                        adj_edge_features_one_hot_list.append(edge_one_hot)

            if total_possible_edges_in_graph > 0 and \
                    (num_specific_edges_in_graph / total_possible_edges_in_graph < 0.05):
                graphs_mostly_padding_edges += 1

            if not adj_edges_list:
                current_edge_index = torch.empty((2, 0), dtype=torch.long, device=actual_node_features_one_hot.device)
                current_edge_features_model_one_hot = torch.empty((0, expected_edge_feature_output_dim),
                                                                  dtype=torch.float32,
                                                                  device=actual_node_features_one_hot.device)
            else:
                current_edge_index = torch.tensor(adj_edges_list, dtype=torch.long,
                                                  device=actual_node_features_one_hot.device).t().contiguous()
                current_edge_features_model_one_hot = torch.stack(adj_edge_features_one_hot_list).to(
                    dtype=torch.float32,
                    device=actual_node_features_one_hot.device)

            nx_di_graph = convert_raw_model_output_to_nx_aig(
                actual_node_features_one_hot, current_edge_index,
                current_edge_features_model_one_hot, num_nodes_int
            )
            if nx_di_graph is not None:
                generated_graphs_nx_directed.append(nx_di_graph)

        if not generated_graphs_nx_directed:
            if local_rank == 0: print("AIGMetrics: No generated graphs to evaluate.")
            return {}

        to_log: Dict[str, float] = {}
        validity_error_counts = Counter()
        num_total_generated = len(generated_graphs_nx_directed)

        for g in generated_graphs_nx_directed:
            error_code_or_true = aig_check_validity_with_code(g, return_error_code=True)
            if isinstance(error_code_or_true, str):  # Should be a string if return_error_code=True
                validity_error_counts[error_code_or_true] += 1
            elif error_code_or_true is True:  # Fallback if it returns boolean
                validity_error_counts[VALIDITY_ERROR_KEYS[0]] += 1  # Count as "VALID"

        if 'aig_structural_validity' in self.metrics_list:
            num_structurally_valid = validity_error_counts.get(VALIDITY_ERROR_KEYS[0], 0)
            structural_validity_fraction = num_structurally_valid / num_total_generated if num_total_generated > 0 else 0.0
            to_log['aig_metrics/structural_validity_fraction'] = structural_validity_fraction
            if wandb.run and local_rank == 0:
                wandb.summary[f'{name}_aig_structural_validity_fraction'] = structural_validity_fraction

        if 'aig_acyclicity' in self.metrics_list:
            # Assuming VALIDITY_ERROR_KEYS[1] is "NOT_DAG" or similar
            # If not, this part needs adjustment based on your actual error keys.
            not_dag_key = "NOT_DAG"  # Default, adjust if your key is different
            if len(VALIDITY_ERROR_KEYS) > 1 and VALIDITY_ERROR_KEYS[1] == not_dag_key:  # Check if it's the expected key
                num_non_dag = validity_error_counts.get(not_dag_key, 0)
            else:  # Fallback if "NOT_DAG" key is not as expected or VALIDITY_ERROR_KEYS is short
                num_non_dag = sum(1 for g in generated_graphs_nx_directed if not nx.is_directed_acyclic_graph(g))

            num_acyclic = num_total_generated - num_non_dag
            acyclicity_fraction = num_acyclic / num_total_generated if num_total_generated > 0 else 0.0
            to_log['aig_metrics/acyclicity_fraction'] = acyclicity_fraction
            if wandb.run and local_rank == 0:
                wandb.summary[f'{name}_aig_acyclicity_fraction'] = acyclicity_fraction

        if 'aig_validity_breakdown' in self.metrics_list:
            if local_rank == 0: print("\n--- AIG Validity Error Breakdown ---")
            for err_key in VALIDITY_ERROR_KEYS:
                count = validity_error_counts.get(err_key, 0)
                fraction = count / num_total_generated if num_total_generated > 0 else 0.0
                to_log[f'aig_validity_errors/{err_key}'] = fraction
                if local_rank == 0 and count > 0:
                    print(f"  {err_key}: {count} ({fraction:.2%})")
            if local_rank == 0: print("------------------------------------\n")

        frac_mostly_padding_edges = graphs_mostly_padding_edges / num_total_generated if num_total_generated > 0 else 0.0
        to_log['aig_metrics/frac_graphs_mostly_padding_edges'] = frac_mostly_padding_edges
        avg_actual_edges = actual_aig_edges_formed / num_total_generated if num_total_generated > 0 else 0.0
        to_log['aig_metrics/avg_actual_aig_edges_formed'] = avg_actual_edges

        if 'degree' in self.metrics_list:
            current_ref_graphs = self.test_graphs if test else self.val_graphs
            if current_ref_graphs and generated_graphs_nx_directed:
                # Use the new directed_degree_stats_mmd
                degree_mmds = directed_degree_stats_mmd(
                    current_ref_graphs,
                    generated_graphs_nx_directed,
                    is_parallel=True,  # Ensure your helper supports this
                    # kernel_func=gaussian_emd, # Pass the kernel function explicitly
                    # compute_emd=self.compute_emd # This might be redundant if kernel implies it
                )
                to_log['graph_stats/in_degree_mmd'] = degree_mmds.get('in_degree_mmd', -1.0)
                to_log['graph_stats/out_degree_mmd'] = degree_mmds.get('out_degree_mmd', -1.0)
                if wandb.run and local_rank == 0:
                    wandb.summary[f'{name}_in_degree_mmd'] = degree_mmds.get('in_degree_mmd', -1.0)
                    wandb.summary[f'{name}_out_degree_mmd'] = degree_mmds.get('out_degree_mmd', -1.0)
            else:
                to_log['graph_stats/in_degree_mmd'] = -1.0
                to_log['graph_stats/out_degree_mmd'] = -1.0

        def combined_aig_validity_for_eval_fractions(g_eval_nx: nx.DiGraph) -> bool:
            return aig_check_validity_with_code(g_eval_nx, return_error_code=False)

        eval_generated_graphs_nx = [g for g in generated_graphs_nx_directed if g.number_of_nodes() > 0]
        eval_train_graphs_nx = [g for g in self.train_graphs if g.number_of_nodes() > 0]
        frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = 0.0, 0.0, 0.0
        frac_non_iso_to_train = 0.0

        if not eval_generated_graphs_nx:
            if local_rank == 0: print("AIGMetrics: No non-empty generated graphs for uniqueness/novelty checks.")
        elif not eval_train_graphs_nx:
            if local_rank == 0: print(
                "AIGMetrics: No training graphs for novelty. Calculating uniqueness/validity of generated set only.")
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(eval_generated_graphs_nx, [],
                                                          validity_func=combined_aig_validity_for_eval_fractions)
            frac_non_iso_to_train = 1.0
        else:
            frac_unique, frac_unique_non_iso, frac_unique_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(eval_generated_graphs_nx, eval_train_graphs_nx,
                                                          validity_func=combined_aig_validity_for_eval_fractions)
            isomorphic_fraction = eval_fraction_isomorphic(eval_generated_graphs_nx, eval_train_graphs_nx)
            frac_non_iso_to_train = 1.0 - isomorphic_fraction

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
            print(f"\n--- AIGMetrics Results Epoch {current_epoch} ({'Test' if test else 'Val'}) ---")
            for key, val in to_log.items():
                # Only print validity errors that actually occurred, or the "VALID" count
                if 'aig_validity_errors/' in key and val == 0.0 and not key.endswith('/VALID'): continue
                print(f"  {key}: {val:.4f}")
            print("-----------------------------------------------------\n")

        return to_log

    def reset(self):
        if hasattr(super(), 'reset') and callable(super().reset):
            super().reset()
        pass

#