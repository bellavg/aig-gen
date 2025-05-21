# generate_and_evaluate_aigs.py

import argparse
import numpy as np
import torch
import networkx as nx
import pickle
import os
import logging  # Keep logging for this script
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Tuple
import sys
import io
import warnings
from tqdm import tqdm

# --- Imports from GraphRNN project ---
from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP
from aig_config import (
    NODE_TYPE_KEYS, EDGE_TYPE_KEYS,
    MIN_AND_COUNT, MIN_PO_COUNT, MAX_NODE_COUNT,
    DECODING_NODE_TYPE_NX, DECODING_EDGE_TYPE_NX,
    NODE_TYPE_ENCODING, EDGE_LABEL_ENCODING,
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES
)

# --- Logger Setup ---
# logger = logging.getLogger("generate_evaluate_aigs")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Using print for main feedback, but logger can be re-enabled if desired.

NODE_CONST0_EVAL = NODE_TYPE_KEYS[0]
NODE_PI_EVAL = NODE_TYPE_KEYS[1]
NODE_AND_EVAL = NODE_TYPE_KEYS[2]
NODE_PO_EVAL = NODE_TYPE_KEYS[3]


# ==============================================================================
# Functions from/adapted from generate.py
# ==============================================================================

def m_seq_to_adj_mat(m_seq, m_window_size):
    num_generated_sequences = m_seq.shape[0]
    n = num_generated_sequences + 1
    adj_mat = np.zeros((n, n), dtype=int)
    for i in range(num_generated_sequences):
        prev_nodes_connections = m_seq[i, :]
        current_node_idx = i + 1
        num_to_take_from_m_vec = min(current_node_idx, m_window_size)
        start_col_index = max(0, current_node_idx - num_to_take_from_m_vec)
        assign_slice = list(reversed(prev_nodes_connections[:num_to_take_from_m_vec]))
        adj_mat[current_node_idx, start_col_index: current_node_idx] = assign_slice
    return adj_mat


def sample_bernoulli(p_tensor):
    return torch.bernoulli(p_tensor).int()


def sample_softmax(logits_tensor):
    probabilities = torch.softmax(logits_tensor, dim=-1)
    # Ensure probabilities sum to 1, handle potential numerical issues
    probabilities = probabilities / torch.sum(probabilities, dim=-1, keepdim=True).clamp(
        min=1e-8)  # Add clamp for stability
    sampled_indices = torch.multinomial(probabilities, 1).squeeze(-1)
    one_hot = torch.nn.functional.one_hot(sampled_indices, num_classes=logits_tensor.shape[-1])
    return one_hot.float()


def rnn_edge_gen(edge_rnn, h_for_edge_model, num_edges_to_predict,
                 m_window_size, sample_fun_edges, edge_feature_len, attempts=None):
    device = h_for_edge_model.device
    adj_vec_output = torch.zeros([1, 1, m_window_size, edge_feature_len], device=device)
    if num_edges_to_predict == 0:
        return adj_vec_output
    edge_rnn.set_first_layer_hidden(h_for_edge_model.squeeze(1))
    current_edge_input_features = torch.ones([1, 1, edge_feature_len], device=device)
    for i in range(num_edges_to_predict):
        edge_pred_output = edge_rnn(current_edge_input_features, return_logits=True)
        sampled_edge_features = sample_fun_edges(edge_pred_output.squeeze(0).squeeze(0))
        adj_vec_output[0, 0, i, :] = sampled_edge_features
        current_edge_input_features[0, 0, :] = sampled_edge_features
    return adj_vec_output


def mlp_edge_gen(edge_mlp, h_for_edge_model, num_edges_to_predict,
                 m_window_size, sample_fun_edges, edge_feature_len, attempts=1):
    device = h_for_edge_model.device
    adj_vec_output = torch.zeros([1, 1, m_window_size, edge_feature_len], device=device)
    if num_edges_to_predict == 0:
        return adj_vec_output
    edge_predictions_all = edge_mlp(h_for_edge_model, return_logits=True)
    for _ in range(attempts):
        for i in range(num_edges_to_predict):
            sampled_edge_features = sample_fun_edges(edge_predictions_all[0, 0, i, :])
            adj_vec_output[0, 0, i, :] = sampled_edge_features
        if (adj_vec_output[0, 0, :num_edges_to_predict, :] != 0).any():
            break
    return adj_vec_output


def generate_single_graph(num_target_nodes, node_model, edge_model,
                          m_window_size, num_node_classes, edge_feature_len_model,
                          edge_gen_function, generation_mode, edge_sample_attempts=1):
    device = next(node_model.parameters()).device
    node_model.eval()
    edge_model.eval()

    if edge_feature_len_model > 1:
        sample_fun_edges = lambda logits: sample_softmax(logits)
    else:
        sample_fun_edges = lambda logits: sample_bernoulli(torch.sigmoid(logits)).view(1)

    prev_adj_vec_input = torch.ones([1, 1, m_window_size, edge_feature_len_model], device=device)
    prev_node_attr_input = torch.zeros([1, 1, num_node_classes], device=device)

    list_s_i_scalar_class_vectors = []
    list_node_attrs_one_hot = []
    node_model.reset_hidden()
    actual_num_nodes_generated = 0

    for i in range(num_target_nodes):  # Generate node 0, 1, ..., num_target_nodes-1
        h_for_edge_model, node_attr_logits = node_model(prev_adj_vec_input, prev_node_attr_input)
        current_node_attr_onehot = sample_softmax(node_attr_logits.squeeze(0).squeeze(0))
        list_node_attrs_one_hot.append(current_node_attr_onehot.cpu().numpy())
        actual_num_nodes_generated += 1
        num_edges_to_predict_for_current_node = min(i, m_window_size)
        adj_vec_generated_current_one_hot = edge_gen_function(
            edge_model, h_for_edge_model,
            num_edges_to_predict_for_current_node, m_window_size,
            sample_fun_edges, edge_feature_len_model, attempts=edge_sample_attempts
        )
        adj_vec_slice_one_hot = adj_vec_generated_current_one_hot[0, 0, :num_edges_to_predict_for_current_node, :]
        if edge_feature_len_model > 1:
            processed_slice_scalar_classes = torch.argmax(adj_vec_slice_one_hot, dim=-1).cpu().int().numpy()
        else:
            processed_slice_scalar_classes = adj_vec_slice_one_hot.squeeze(-1).cpu().int().numpy()
        final_s_i_for_list = np.zeros(m_window_size, dtype=int)
        if num_edges_to_predict_for_current_node > 0:
            final_s_i_for_list[:num_edges_to_predict_for_current_node] = processed_slice_scalar_classes

        # --- MODIFICATION: EOS condition removed ---
        # if i > 0 and np.all(final_s_i_for_list[:num_edges_to_predict_for_current_node] == 0): # EOS check
        #     actual_num_nodes_generated -= 1
        #     list_node_attrs_one_hot.pop()
        #     break
        # --- End MODIFICATION ---

        list_s_i_scalar_class_vectors.append(final_s_i_for_list)

        prev_adj_vec_input = adj_vec_generated_current_one_hot
        prev_node_attr_input = current_node_attr_onehot.unsqueeze(0).unsqueeze(0)

    final_node_attrs_np = np.array(list_node_attrs_one_hot)

    # If loop completed, actual_num_nodes_generated will be num_target_nodes
    # and len(list_s_i_scalar_class_vectors) will be num_target_nodes.

    if not list_s_i_scalar_class_vectors or actual_num_nodes_generated == 0:
        return np.array([]), np.array([])

    # m_seq_to_adj_mat expects N-1 sequences (S_0 to S_{N-2}) to create an N x N matrix.
    # list_s_i_scalar_class_vectors has N elements (S_0 to S_{N-1}) if EOS was removed and loop completed.
    if actual_num_nodes_generated == 1:
        m_seq_for_conversion = np.array([])
    elif actual_num_nodes_generated > 1:
        # Use S_0 to S_{N-2} for an N-node graph.
        m_seq_for_conversion = np.array(list_s_i_scalar_class_vectors[:-1])
    else:  # actual_num_nodes_generated is 0
        m_seq_for_conversion = np.array([])

    if m_seq_for_conversion.ndim == 1 and m_seq_for_conversion.size > 0:
        m_seq_for_conversion = m_seq_for_conversion.reshape(1, -1)

    adj_matrix_scalar_classes = np.array([])
    if m_seq_for_conversion.shape[0] > 0:
        adj_matrix_scalar_classes = m_seq_to_adj_mat(m_seq_for_conversion, m_window_size)
    elif actual_num_nodes_generated > 0:
        adj_matrix_scalar_classes = np.zeros((actual_num_nodes_generated, actual_num_nodes_generated), dtype=int)

    # Ensure node attributes match the final number of nodes.
    # Since EOS is removed, actual_num_nodes_generated should match final_node_attrs_np.shape[0]
    # and adj_matrix_scalar_classes.shape[0] if actual_num_nodes_generated > 0.
    if final_node_attrs_np.shape[0] != actual_num_nodes_generated:
        print(
            f"Warning: Mismatch final_node_attrs ({final_node_attrs_np.shape[0]}) and actual_nodes_generated ({actual_num_nodes_generated}).")
        # This case should be less likely now without EOS modifying actual_num_nodes_generated mid-loop.
        if final_node_attrs_np.shape[0] > actual_num_nodes_generated:
            final_node_attrs_np = final_node_attrs_np[:actual_num_nodes_generated]
        # else: (padding case is less likely if EOS is removed)

    if adj_matrix_scalar_classes.shape[0] != actual_num_nodes_generated and actual_num_nodes_generated > 0:
        print(
            f"Warning: Mismatch adj_matrix_scalar_classes ({adj_matrix_scalar_classes.shape[0]}) and actual_nodes_generated ({actual_num_nodes_generated}). Creating zero matrix.")
        adj_matrix_scalar_classes = np.zeros((actual_num_nodes_generated, actual_num_nodes_generated), dtype=int)

    return adj_matrix_scalar_classes, final_node_attrs_np


def load_model_from_checkpoint(model_path, device):
    state = torch.load(model_path, map_location=device)
    config = state['config']
    m_window_size = config['data']['m']
    num_node_classes = config['model']['GraphRNN']['num_node_classes']
    edge_feature_len_model = config['model']['GraphRNN']['edge_feature_len']
    generation_mode = config['model']['mode']
    if config['model']['edge_model'] == 'rnn':
        node_model = GraphLevelRNN(input_size=m_window_size, output_size=config['model']['EdgeRNN']['hidden_size'],
                                   **config['model']['GraphRNN']).to(device)
        edge_model = EdgeLevelRNN(**config['model']['EdgeRNN']).to(device)
        edge_gen_function = rnn_edge_gen
    else:
        node_model = GraphLevelRNN(input_size=m_window_size, output_size=None, **config['model']['GraphRNN']).to(device)
        edge_model = EdgeLevelMLP(input_size_from_graph_rnn=config['model']['GraphRNN']['hidden_size'],
                                  mlp_hidden_size=config['model']['EdgeMLP']['hidden_size'],
                                  num_edges_to_predict=m_window_size, edge_feature_len=edge_feature_len_model).to(
            device)
        edge_gen_function = mlp_edge_gen
    node_model.load_state_dict(state['node_model'])
    edge_model.load_state_dict(state['edge_model'])
    print(
        f"Loaded model from {model_path} with mode: {generation_mode}, M: {m_window_size}, NodeClasses: {num_node_classes}, EdgeFeatLen: {edge_feature_len_model}")
    return node_model, edge_model, m_window_size, num_node_classes, edge_feature_len_model, edge_gen_function, generation_mode, config


# ==============================================================================
# Functions from/adapted from evaluate_aigs.py
# ==============================================================================

def get_node_type_str_from_attrs(attrs: dict) -> str:
    raw_type = attrs.get('type')
    if raw_type is None: return "UNKNOWN_MISSING_ATTR"
    if isinstance(raw_type, str): return raw_type
    if isinstance(raw_type, (list, np.ndarray)):
        try:
            type_tuple = tuple(float(x) for x in raw_type); return DECODING_NODE_TYPE_NX.get(type_tuple,
                                                                                             "UNKNOWN_ENCODING")
        except (ValueError, TypeError):
            return "UNKNOWN_VECTOR_CONVERSION_ERROR"
    return f"UNKNOWN_TYPE_FORMAT_{type(raw_type).__name__}"


def get_edge_type_str_from_attrs(attrs: dict) -> str:
    raw_type = attrs.get('type')
    if raw_type is None: return "UNKNOWN_MISSING_ATTR"
    if isinstance(raw_type, str): return raw_type
    if isinstance(raw_type, (list, np.ndarray)):
        try:
            type_tuple = tuple(float(x) for x in raw_type); return DECODING_EDGE_TYPE_NX.get(type_tuple,
                                                                                             "UNKNOWN_ENCODING")
        except (ValueError, TypeError):
            return "UNKNOWN_VECTOR_CONVERSION_ERROR"
    return f"UNKNOWN_TYPE_FORMAT_{type(raw_type).__name__}"


def calculate_structural_aig_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    metrics = defaultdict(float)
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['constraints_failed'] = []
    if not isinstance(G, nx.DiGraph) or metrics['num_nodes'] == 0:
        metrics['constraints_failed'].append("Empty or Invalid Graph Object");
        metrics['is_structurally_valid'] = False;
        return dict(metrics)
    try:
        metrics['is_dag'] = float(nx.is_directed_acyclic_graph(G))
        if not metrics['is_dag']: metrics['constraints_failed'].append("Not a DAG")
    except Exception:
        metrics['is_dag'] = 0.0; metrics['constraints_failed'].append("DAG Check Error")
    node_type_counts = Counter()
    for node, data in G.nodes(data=True):
        node_type_str = get_node_type_str_from_attrs(data);
        node_type_counts[node_type_str] += 1
        if node_type_str not in NODE_TYPE_KEYS:
            metrics['num_unknown_nodes'] += 1
            if "Unknown node types" not in metrics['constraints_failed']: metrics['constraints_failed'].append(
                "Unknown node types")
            continue
        try:
            in_deg, out_deg = G.in_degree(node), G.out_degree(node)
        except Exception:
            metrics['degree_check_errors'] += 1; continue
        if node_type_str == NODE_CONST0_EVAL:
            if in_deg != 0: metrics['const0_indegree_violations'] += 1
        elif node_type_str == NODE_PI_EVAL:
            if in_deg != 0: metrics['pi_indegree_violations'] += 1
        elif node_type_str == NODE_AND_EVAL:
            if in_deg != 2: metrics['and_indegree_violations'] += 1
        elif node_type_str == NODE_PO_EVAL:
            if out_deg != 0: metrics['po_outdegree_violations'] += 1
            if in_deg == 0: metrics['po_indegree_violations'] += 1
    metrics['num_pi'] = float(node_type_counts.get(NODE_PI_EVAL, 0));
    metrics['num_po'] = float(node_type_counts.get(NODE_PO_EVAL, 0));
    metrics['num_and'] = float(node_type_counts.get(NODE_AND_EVAL, 0))
    if metrics['const0_indegree_violations'] > 0: metrics['constraints_failed'].append("CONST0 in-degree != 0")
    if metrics['pi_indegree_violations'] > 0: metrics['constraints_failed'].append("PI in-degree != 0")
    if metrics['and_indegree_violations'] > 0: metrics['constraints_failed'].append("AND in-degree != 2")
    if metrics['po_outdegree_violations'] > 0: metrics['constraints_failed'].append("PO out-degree != 0")
    if metrics['po_indegree_violations'] > 0: metrics['constraints_failed'].append("PO in-degree == 0")
    for u, v, data in G.edges(data=True):
        edge_type_str = get_edge_type_str_from_attrs(data)
        if edge_type_str not in EDGE_TYPE_KEYS:
            metrics['num_unknown_edges'] += 1
            if "Unknown edge types" not in metrics['constraints_failed']: metrics['constraints_failed'].append(
                "Unknown edge types")
    if metrics['num_pi'] == 0 and node_type_counts.get(NODE_CONST0_EVAL, 0) == 0: metrics['constraints_failed'].append(
        "No PIs or Const0")
    if metrics['num_and'] < MIN_AND_COUNT: metrics['constraints_failed'].append(f"AND gates < {MIN_AND_COUNT}")
    if metrics['num_po'] < MIN_PO_COUNT: metrics['constraints_failed'].append(f"POs < {MIN_PO_COUNT}")
    is_valid = (metrics['is_dag'] > 0.5 and metrics['num_unknown_nodes'] == 0 and metrics['num_unknown_edges'] == 0 and
                metrics['const0_indegree_violations'] == 0 and metrics['pi_indegree_violations'] == 0 and metrics[
                    'and_indegree_violations'] == 0 and metrics['po_outdegree_violations'] == 0 and metrics[
                    'po_indegree_violations'] == 0 and (
                            metrics['num_pi'] > 0 or node_type_counts.get(NODE_CONST0_EVAL, 0) > 0) and metrics[
                    'num_and'] >= MIN_AND_COUNT and metrics['num_po'] >= MIN_PO_COUNT)
    metrics['is_structurally_valid'] = float(is_valid)
    if not is_valid and not metrics['constraints_failed']: metrics['constraints_failed'].append(
        "General Validity Rules Failed")
    return dict(metrics)


def node_match_aig_flexible(node1_attrs, node2_attrs):
    type1_str = get_node_type_str_from_attrs(node1_attrs);
    type2_str = get_node_type_str_from_attrs(node2_attrs)
    if "UNKNOWN" in type1_str and "UNKNOWN" not in type2_str: return False
    if "UNKNOWN" in type2_str and "UNKNOWN" not in type1_str: return False
    return type1_str == type2_str


def edge_match_aig_flexible(edge1_attrs, edge2_attrs):
    type1_str = get_edge_type_str_from_attrs(edge1_attrs);
    type2_str = get_edge_type_str_from_attrs(edge2_attrs)
    if "UNKNOWN" in type1_str and "UNKNOWN" not in type2_str: return False
    if "UNKNOWN" in type2_str and "UNKNOWN" not in type1_str: return False
    return type1_str == type2_str


def are_graphs_isomorphic(G1: nx.DiGraph, G2: nx.DiGraph) -> bool:
    if G1 is None or G2 is None: return False
    if G1.number_of_nodes() != G2.number_of_nodes() or G1.number_of_edges() != G2.number_of_edges(): return False
    try:
        return nx.is_isomorphic(G1, G2, node_match=node_match_aig_flexible, edge_match=edge_match_aig_flexible)
    except Exception:
        return False


def calculate_uniqueness(valid_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    num_valid = len(valid_graphs);
    unique_graph_representatives = []
    if num_valid <= 1: return (1.0, num_valid)
    for i in tqdm(range(num_valid), desc="Checking Uniqueness", leave=False, ncols=80):
        current_graph = valid_graphs[i];
        is_unique_to_list = True
        for representative_graph in unique_graph_representatives:
            if are_graphs_isomorphic(current_graph, representative_graph): is_unique_to_list = False; break
        if is_unique_to_list: unique_graph_representatives.append(current_graph)
    num_unique = len(unique_graph_representatives)
    return (num_unique / num_valid if num_valid > 0 else 0.0), num_unique


def calculate_novelty(valid_graphs: List[nx.DiGraph], train_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    num_valid = len(valid_graphs);
    num_novel = 0
    if num_valid == 0: return (0.0, 0)
    if not train_graphs: return (1.0, num_valid)
    for gen_graph in tqdm(valid_graphs, desc="Checking Novelty", leave=False, ncols=80):
        is_novel_to_train_set = True
        for train_graph in train_graphs:
            if are_graphs_isomorphic(gen_graph, train_graph): is_novel_to_train_set = False; break
        if is_novel_to_train_set: num_novel += 1
    return (num_novel / num_valid if num_valid > 0 else 0.0), num_novel


def load_training_graphs_from_pkls(train_pkl_dir: str, train_pkl_prefix: str, num_files_to_load: int = 0) -> Optional[
    List[nx.DiGraph]]:
    """Loads and combines graphs from multiple PKL files in a directory."""
    if not os.path.isdir(train_pkl_dir):
        print(f"Error: Training PKL directory not found: {train_pkl_dir}");
        return None
    try:
        all_pkl_files = sorted(
            [f for f in os.listdir(train_pkl_dir) if f.startswith(train_pkl_prefix) and f.endswith(".pkl")])
    except OSError as e:
        print(f"Error listing files in directory {train_pkl_dir}: {e}");
        return None

    if not all_pkl_files:
        print(f"No PKL files found in {train_pkl_dir} with prefix '{train_pkl_prefix}'.");
        return None

    files_to_process = all_pkl_files
    if num_files_to_load > 0:  # If 0, load all found files
        files_to_process = all_pkl_files[:num_files_to_load]
        if len(files_to_process) < num_files_to_load and num_files_to_load > 0:
            print(
                f"Warning: Found only {len(files_to_process)} files matching prefix, but {num_files_to_load} were requested.")

    if not files_to_process: print("No training PKL files selected to load."); return None

    print(f"Loading training graphs from {len(files_to_process)} PKL files in {train_pkl_dir}:")
    # for fname in files_to_process: print(f" - {fname}") # Can be verbose

    all_train_graphs = []
    for filename in tqdm(files_to_process, desc="Loading Training PKLs", ncols=100):
        file_path = os.path.join(train_pkl_dir, filename)
        try:
            with open(file_path, 'rb') as f:
                graphs_chunk = pickle.load(f, encoding='latin1')  # Added encoding
            if isinstance(graphs_chunk, list):
                valid_graphs_in_chunk = [g for g in graphs_chunk if isinstance(g, nx.DiGraph)]
                all_train_graphs.extend(valid_graphs_in_chunk)
                if len(valid_graphs_in_chunk) != len(graphs_chunk):
                    print(f"Warning: File {filename} contained non-DiGraph items that were skipped.")
            else:
                print(f"Warning: File {filename} did not contain a list. Skipping.")
        except Exception as e:
            print(f"Error loading or processing PKL file {filename}: {e}")
    print(f"Finished loading training graphs. Total graphs loaded: {len(all_train_graphs)}")
    return all_train_graphs


# ==============================================================================
# New Helper Function to Convert Generated Output to NetworkX AIG
# ==============================================================================
def generated_output_to_nx_aig(adj_matrix_scalar_classes: np.ndarray,
                               node_attrs_one_hot: np.ndarray) -> nx.DiGraph:
    G = nx.DiGraph()
    num_nodes = node_attrs_one_hot.shape[0]
    if num_nodes == 0: return G

    for i in range(num_nodes):
        G.add_node(i, type=list(node_attrs_one_hot[i, :]))

    if adj_matrix_scalar_classes.size > 0 and adj_matrix_scalar_classes.shape == (num_nodes, num_nodes):
        for s_idx in range(num_nodes):  # Source node
            for t_idx in range(num_nodes):  # Target node
                scalar_class = adj_matrix_scalar_classes[t_idx, s_idx]  # Class for edge s_idx -> t_idx
                if s_idx < t_idx:
                    if 0 <= scalar_class < NUM_EDGE_FEATURES:
                        try:
                            edge_type_key = EDGE_TYPE_KEYS[scalar_class]
                            one_hot_edge_type = EDGE_LABEL_ENCODING[edge_type_key]
                            G.add_edge(s_idx, t_idx, type=list(one_hot_edge_type))
                        except IndexError:
                            print(
                                f"Warning: Scalar class {scalar_class} out of bounds for EDGE_TYPE_KEYS. Skipping edge {s_idx}->{t_idx}")
                        except KeyError:
                            print(
                                f"Warning: Edge type key for scalar class {scalar_class} not in EDGE_LABEL_ENCODING. Skipping edge {s_idx}->{t_idx}")
    elif adj_matrix_scalar_classes.size > 0:
        print(f"Warning: adj_matrix_scalar_classes shape {adj_matrix_scalar_classes.shape} "
              f"does not match num_nodes {num_nodes}. Edges might not be correctly reconstructed.")
    return G


# ==============================================================================
# Main Evaluation Orchestration
# ==============================================================================
def run_evaluation_suite(
        generated_nx_aigs: List[nx.DiGraph],
        training_graphs_nx: Optional[List[nx.DiGraph]] = None,
        results_filename="evaluation_results.txt"
):
    num_total_generated = len(generated_nx_aigs)
    if num_total_generated == 0:
        print("No graphs were provided for evaluation.")
        with open(results_filename, 'w') as res_file: res_file.write(
            "No graphs generated or provided for evaluation.\n")
        return {}
    print(f"\n--- Starting AIG V.U.N. Evaluation for {num_total_generated} graphs ---")
    structurally_valid_aigs = []
    aggregate_metrics_all = defaultdict(list)
    failed_constraints_summary = Counter()
    for i, G_aig in enumerate(tqdm(generated_nx_aigs, desc="AIG Validity Check", ncols=80)):
        if not isinstance(G_aig, nx.DiGraph):
            print(f"Warning: Item {i} is not a NetworkX DiGraph. Skipping.");
            failed_constraints_summary["Invalid Graph Object Type"] += 1;
            continue
        metrics = calculate_structural_aig_metrics(G_aig)
        for k, v_val in metrics.items():
            if k == 'constraints_failed':
                if isinstance(v_val, list):
                    for reason in v_val: failed_constraints_summary[reason] += 1
            elif isinstance(v_val, (float, int)):
                aggregate_metrics_all[k].append(v_val)
        if metrics.get('is_structurally_valid', 0.0) > 0.5: structurally_valid_aigs.append(G_aig)
    uniqueness_score, num_unique = calculate_uniqueness(structurally_valid_aigs)
    novelty_score, num_novel = (-1.0, -1)
    if training_graphs_nx is not None:
        novelty_score, num_novel = calculate_novelty(structurally_valid_aigs, training_graphs_nx)
    else:
        print("Training graphs not provided. Novelty will be 100% if any valid graphs, or N/A.")
        if structurally_valid_aigs: novelty_score, num_novel = 1.0, len(structurally_valid_aigs)
    original_stdout = sys.stdout;
    string_buffer = io.StringIO();
    sys.stdout = string_buffer
    num_valid_structurally = len(structurally_valid_aigs)
    validity_fraction = (num_valid_structurally / num_total_generated) if num_total_generated > 0 else 0.0
    print("\n--- AIG V.U.N. Evaluation Summary ---")
    print(f"Total Graphs Generated & Evaluated: {num_total_generated}")
    print(f"Structurally Valid AIGs (V)     : {num_valid_structurally} ({validity_fraction * 100:.2f}%)")
    if num_valid_structurally > 0:
        print(f"Unique Valid AIGs             : {num_unique}")
        print(f"Uniqueness (U) among valid    : {uniqueness_score:.4f} ({uniqueness_score * 100:.2f}%)")
        if novelty_score != -1.0:
            print(f"Novel Valid AIGs vs Train Set : {num_novel}")
            print(f"Novelty (N) among valid       : {novelty_score:.4f} ({novelty_score * 100:.2f}%)")
        else:
            print(f"Novelty (N) among valid       : Not calculated")
    else:
        print(f"Uniqueness (U) among valid    : N/A (0 valid graphs)"); print(
            f"Novelty (N) among valid       : N/A (0 valid graphs)")
    print("\n--- Average Structural Metrics (All Processed Graphs) ---")
    if aggregate_metrics_all and num_total_generated > 0:
        for key, values in sorted(aggregate_metrics_all.items()):
            if key == 'is_structurally_valid': continue
            if not values: print(f"  - Avg {key:<27}: N/A (no data)"); continue
            avg_value = np.mean(values);
            std_value = np.std(values) if len(values) > 1 else 0.0
            if key == 'is_dag':
                print(f"  - Percentage {key:<22}: {avg_value * 100:.2f}%")
            else:
                print(f"  - Avg {key:<27}: {avg_value:.3f} (Std: {std_value:.3f})")
    else:
        print("  No structural metrics data collected or no graphs processed.")
    print("\n--- Constraint Violation Summary (Across Invalid Graphs) ---")
    num_structurally_invalid = num_total_generated - num_valid_structurally
    if not failed_constraints_summary and num_structurally_invalid == 0:
        print("  No structural violations detected.")
    elif not failed_constraints_summary and num_structurally_invalid > 0:
        print(f"  {num_structurally_invalid} graphs were invalid, but no specific reasons logged by metrics.")
    else:
        sorted_reasons = sorted(failed_constraints_summary.items(), key=lambda item: item[1], reverse=True)
        print(f"  (Violations summarized across {num_structurally_invalid} structurally invalid graphs)")
        for reason, count in sorted_reasons:
            reason_percentage = (count / num_structurally_invalid) * 100 if num_structurally_invalid > 0 else 0
            print(f"  - {reason:<45}: {count:<6} occurrences ({reason_percentage:.1f}% of invalid graphs)")
    print("------------------------------------")
    results_output_str = string_buffer.getvalue();
    sys.stdout = original_stdout;
    string_buffer.close()
    print(results_output_str)
    try:
        with open(results_filename, 'w') as res_file:
            res_file.write(results_output_str)
        print(f"Evaluation results also saved to {results_filename}")
    except IOError as e_io:
        print(f"Error: Failed to write results to file {results_filename}: {e_io}", file=sys.stderr)
    eval_summary = {'total_generated': num_total_generated, 'num_valid': num_valid_structurally,
                    'validity_fraction': validity_fraction, 'num_unique_valid': num_unique,
                    'uniqueness_score': uniqueness_score, 'num_novel_valid': num_novel, 'novelty_score': novelty_score,
                    'avg_metrics': {k: np.mean(v) if v else 0 for k, v in aggregate_metrics_all.items()},
                    'failed_constraints': dict(failed_constraints_summary)}
    return eval_summary


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and Evaluate AIGs from a GraphRNN checkpoint.")
    parser.add_argument('model_checkpoint_path', type=str,
                        help='Path to the trained GraphRNN model checkpoint (.pth file)')
    parser.add_argument('--num_graphs', type=int, default=100, help='Number of graphs to generate for evaluation.')
    parser.add_argument('--num_nodes', type=int, default=None,
                        help='Target number of nodes for each generated graph. If None, uses range.')
    parser.add_argument('--min_nodes_generate', type=int, default=5, help='Min nodes if --num_nodes is None.')
    parser.add_argument('--max_nodes_generate', type=int, default=MAX_NODE_COUNT,
                        help='Max nodes if --num_nodes is None.')

    # MODIFIED: Changed defaults for training data loading
    parser.add_argument('--train_set_pkl_dir', type=str, default="../raw_data/networkx_aigs/",
                        help='(Optional) Directory containing training PKL files.')
    parser.add_argument('--train_set_pkl_prefix', type=str, default="real_aigs_part_",
                        help='(Optional) Prefix of training PKL files.')
    parser.add_argument('--num_train_pkl_files', type=int, default=4,
                        help='(Optional) Number of training PKL files to load (0 for all found).')

    parser.add_argument('--results_file', type=str, default="generated_aig_eval_results.txt",
                        help="Filename to save the evaluation summary.")
    parser.add_argument('--output_generated_pkl', type=str, default=None,
                        help="(Optional) Path to save generated NetworkX AIGs.")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        (node_model, edge_model, m_param,
         num_node_classes_model, edge_feat_len_model,
         edge_gen_func, model_mode, model_config) = load_model_from_checkpoint(args.model_checkpoint_path, device)
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {args.model_checkpoint_path}"); sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}"); sys.exit(1)

    print(f"\nGenerating {args.num_graphs} graphs...")
    generated_nx_aigs_list = []
    raw_generated_outputs = []
    for i in tqdm(range(args.num_graphs), desc="Generating Graphs", ncols=100):
        nodes_to_generate_for_this_graph = args.num_nodes if args.num_nodes is not None else np.random.randint(
            args.min_nodes_generate, args.max_nodes_generate + 1)
        if nodes_to_generate_for_this_graph <= 0: raw_generated_outputs.append((np.array([]), np.array([]))); continue
        adj_matrix_scalar_classes, final_node_attrs_np = generate_single_graph(
            num_target_nodes=nodes_to_generate_for_this_graph, node_model=node_model, edge_model=edge_model,
            m_window_size=m_param, num_node_classes=num_node_classes_model, edge_feature_len_model=edge_feat_len_model,
            edge_gen_function=edge_gen_func, generation_mode=model_mode
        )
        raw_generated_outputs.append((adj_matrix_scalar_classes, final_node_attrs_np))
    if not raw_generated_outputs: print("No graphs were generated. Exiting evaluation."); sys.exit(0)
    print(f"Successfully generated {len(raw_generated_outputs)} raw graph outputs.")

    print("\nConverting generated outputs to NetworkX AIGs for evaluation...")
    for adj_matrix, node_attrs in tqdm(raw_generated_outputs, desc="Converting to NX", ncols=100):
        if adj_matrix.size > 0 and node_attrs.size > 0:  # Check if both are non-empty
            nx_aig = generated_output_to_nx_aig(adj_matrix, node_attrs)
            generated_nx_aigs_list.append(nx_aig)
        else:
            generated_nx_aigs_list.append(nx.DiGraph())  # Add empty graph if generation resulted in empty

    if args.output_generated_pkl:
        print(f"Saving {len(generated_nx_aigs_list)} generated NetworkX AIGs to {args.output_generated_pkl}...")
        try:
            with open(args.output_generated_pkl, 'wb') as f_out:
                pickle.dump(generated_nx_aigs_list, f_out)
            print("Saved generated graphs.")
        except Exception as e_save:
            print(f"Error saving generated graphs: {e_save}")

    training_aigs_for_novelty = None
    if args.train_set_pkl_dir and args.train_set_pkl_prefix:  # Check if dir and prefix are provided
        print(
            f"\nLoading training AIGs for novelty check from dir: {args.train_set_pkl_dir} with prefix: {args.train_set_pkl_prefix}")
        training_aigs_for_novelty = load_training_graphs_from_pkls(
            args.train_set_pkl_dir,
            args.train_set_pkl_prefix,
            args.num_train_pkl_files
        )
        if training_aigs_for_novelty:
            print(f"Loaded {len(training_aigs_for_novelty)} training AIGs.")
        else:
            print(
                f"Warning: Could not load training AIGs from the specified location. Novelty will be based on empty set (all unique).")
    else:
        print("\nTraining data directory or prefix not specified. Skipping novelty calculation against a training set.")

    evaluation_summary = run_evaluation_suite(generated_nx_aigs_list, training_graphs_nx=training_aigs_for_novelty,
                                              results_filename=args.results_file)
    print("\nGeneration and Evaluation finished.")

