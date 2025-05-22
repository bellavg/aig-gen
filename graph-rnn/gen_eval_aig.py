# generate_and_evaluate_aigs.py

import argparse
import numpy as np
import torch
import networkx as nx
import pickle
import os
# import logging # Keep logging for this script
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Tuple
import sys
import io
import warnings
from tqdm import tqdm

# --- Imports from GraphRNN project ---
from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP
from aig_config import (
    NODE_TYPE_KEYS, EDGE_TYPE_KEYS,  # EDGE_TYPE_KEYS[0] should be "EDGE_NO_EDGE"
    MIN_AND_COUNT, MIN_PO_COUNT, MAX_NODE_COUNT,
    DECODING_NODE_TYPE_NX, DECODING_EDGE_TYPE_NX,
    NODE_TYPE_ENCODING, EDGE_LABEL_ENCODING,
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES  # NUM_EDGE_FEATURES should be 3
)

# --- Logger Setup ---
# logger = logging.getLogger("generate_evaluate_aigs")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Using print for main feedback, but logger can be re-enabled if desired.

NODE_CONST0_EVAL = NODE_TYPE_KEYS[0]
NODE_PI_EVAL = NODE_TYPE_KEYS[1]
NODE_AND_EVAL = NODE_TYPE_KEYS[2]
NODE_PO_EVAL = NODE_TYPE_KEYS[3]

# Define the string name for the "no edge" type directly from your config
# This ensures consistency if EDGE_TYPE_KEYS order ever changes (though index 0 is assumed for NO_EDGE)
NO_EDGE_TYPE_NAME = EDGE_TYPE_KEYS[0]  # Should correspond to "EDGE_NO_EDGE"


# ==============================================================================
# Functions from/adapted from generate.py
# ==============================================================================

def m_seq_to_adj_mat(m_seq, m_window_size):
    # m_seq: (num_nodes - 1, m_window_size) - S_i for node i+1
    # adj_mat: (num_nodes, num_nodes)
    num_generated_sequences = m_seq.shape[0]
    n = num_generated_sequences + 1  # Total number of nodes
    adj_mat = np.zeros((n, n), dtype=int)
    for i in range(num_generated_sequences):  # i from 0 to N-2
        prev_nodes_connections = m_seq[i, :]  # S_i (for node i+1)
        current_node_idx = i + 1  # This is the target node index (1 to N-1)

        num_to_take_from_m_vec = min(current_node_idx, m_window_size)

        # Predecessor nodes are indexed from 0 to current_node_idx - 1
        # start_col_index is the index of the "oldest" predecessor considered in the M-window
        start_col_index = max(0, current_node_idx - num_to_take_from_m_vec)

        # assign_slice contains edge types from predecessors to current_node_idx
        # prev_nodes_connections[0] is connection from current_node_idx-1 to current_node_idx
        # prev_nodes_connections[m-1] is connection from current_node_idx-m to current_node_idx
        # We want adj_mat[source, target]
        # The S_i vector (prev_nodes_connections) is for incoming edges to current_node_idx
        # So, adj_mat[predecessor_idx, current_node_idx] = type

        # The original GraphRNN's m_seq_to_adj_mat:
        # adj_mat[current_node_idx, start_col_index : current_node_idx] = reversed_slice
        # This means adj_mat[target_node, source_node_in_window]
        # where reversed_slice[0] is connection from oldest considered predecessor.

        assign_slice = list(reversed(prev_nodes_connections[:num_to_take_from_m_vec]))
        adj_mat[current_node_idx, start_col_index:current_node_idx] = assign_slice
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
    # For "EDGE_NO_EDGE" as class 0, SOS should not be all ones if that implies NO_EDGE.
    # Let's use a distinct SOS pattern if needed, or rely on model learning.
    # Original GraphRNN used all ones. If [1,0,0] is NO_EDGE, then all ones is not it.
    # A common SOS for one-hot is a separate SOS vector or just zeros if the model handles it.
    # For simplicity, let's keep it as ones, assuming the model learns to interpret it.
    # Or, use a specific SOS encoding if you have one.
    # If EDGE_LABEL_ENCODING["EDGE_NO_EDGE"] is [1,0,0], then torch.ones might be okay if it's not that.
    # A safer SOS might be a vector that's not any of the valid edge types, e.g., all 1/N.
    # Or, if the first input is always ignored by GRU due to hidden state, it might not matter.
    # Let's stick to original GraphRNN's SOS of ones for now.
    current_edge_input_features = torch.ones([1, 1, edge_feature_len], device=device)
    # If "EDGE_NO_EDGE" is [1,0,0], then torch.ones([1,1,3]) is [1,1,1] which is not NO_EDGE.
    # This is probably fine as a generic SOS.

    for i in range(num_edges_to_predict):
        edge_pred_output = edge_rnn(current_edge_input_features,
                                    return_logits=True)  # Expects logits for sample_softmax
        sampled_edge_features = sample_fun_edges(edge_pred_output.squeeze(0).squeeze(0))
        adj_vec_output[0, 0, i, :] = sampled_edge_features
        current_edge_input_features[0, 0, :] = sampled_edge_features  # Teacher forcing with generated sample
    return adj_vec_output


def mlp_edge_gen(edge_mlp, h_for_edge_model, num_edges_to_predict,
                 m_window_size, sample_fun_edges, edge_feature_len, attempts=1):
    device = h_for_edge_model.device
    adj_vec_output = torch.zeros([1, 1, m_window_size, edge_feature_len], device=device)
    if num_edges_to_predict == 0:
        return adj_vec_output
    edge_predictions_all = edge_mlp(h_for_edge_model, return_logits=True)  # Expects logits
    for _ in range(attempts):  # Not standard for MLP, but kept from original
        for i in range(num_edges_to_predict):
            sampled_edge_features = sample_fun_edges(edge_predictions_all[0, 0, i, :])
            adj_vec_output[0, 0, i, :] = sampled_edge_features
        if (adj_vec_output[0, 0, :num_edges_to_predict,
            :] != 0).any():  # Check if any edge (non-zero feature vector) was sampled
            break
    return adj_vec_output


def generate_single_graph(num_target_nodes, node_model, edge_model,
                          m_window_size, num_node_classes, edge_feature_len_model,
                          # This is NUM_EDGE_FEATURES (e.g., 3)
                          edge_gen_function, generation_mode, edge_sample_attempts=1):
    device = next(node_model.parameters()).device
    node_model.eval()
    edge_model.eval()

    # sample_fun_edges expects logits and returns a one-hot tensor of shape [edge_feature_len_model]
    if edge_feature_len_model > 1:  # True for AIGs with 3 edge types
        sample_fun_edges = lambda logits: sample_softmax(logits)
    else:  # Binary edges
        sample_fun_edges = lambda logits: sample_bernoulli(torch.sigmoid(logits)).view(1)

    # SOS for GraphLevelRNN's adjacency input (S_{-1})
    # If "EDGE_NO_EDGE" is [1,0,0], then torch.ones is not it.
    # A better S_{-1} might be all "EDGE_NO_EDGE" encoding.
    # Let's use the encoding for "EDGE_NO_EDGE" as the SOS for adjacency.
    # This assumes the model learns that the first "adjacency" vector is special.
    # Or, the original GraphRNN used all ones, which is distinct from [1,0,0].
    # Let's stick to original GraphRNN's all ones for S_{-1} for now.
    prev_adj_vec_input = torch.ones([1, 1, m_window_size, edge_feature_len_model], device=device)

    # SOS for GraphLevelRNN's node attribute input (A_{-1})
    prev_node_attr_input = torch.zeros([1, 1, num_node_classes], device=device)  # All zeros is a common SOS

    list_s_i_scalar_class_vectors = []  # Stores S_i vectors (scalar class indices) for m_seq_to_adj_mat
    list_node_attrs_one_hot = []  # Stores one-hot A_i
    node_model.reset_hidden()
    actual_num_nodes_generated = 0

    for i in range(num_target_nodes):  # Generates node 0, 1, ..., num_target_nodes-1
        # node_model input: S_{i-1}, A_{i-1}
        # node_model output: h_i (for EdgeModel of node i), logits for A_i
        h_for_edge_model, node_attr_logits = node_model(prev_adj_vec_input, prev_node_attr_input)

        current_node_attr_onehot = sample_softmax(node_attr_logits.squeeze(0).squeeze(0))  # Sample A_i
        list_node_attrs_one_hot.append(current_node_attr_onehot.cpu().numpy())
        actual_num_nodes_generated += 1

        num_edges_to_predict_for_current_node = min(i, m_window_size)  # Node i connects to min(i, M) predecessors

        # adj_vec_generated_current_one_hot is S_i (one-hot edge types for connections to node i)
        # Shape: [1, 1, m_window_size, edge_feature_len_model]
        adj_vec_generated_current_one_hot = edge_gen_function(
            edge_model, h_for_edge_model,  # h_i
            num_edges_to_predict_for_current_node, m_window_size,
            sample_fun_edges, edge_feature_len_model, attempts=edge_sample_attempts
        )

        # Convert one-hot S_i to scalar class indices for storage
        adj_vec_slice_one_hot = adj_vec_generated_current_one_hot[0, 0, :num_edges_to_predict_for_current_node, :]
        if edge_feature_len_model > 1:  # True for AIG
            processed_slice_scalar_classes = torch.argmax(adj_vec_slice_one_hot, dim=-1).cpu().int().numpy()
        else:  # Binary
            processed_slice_scalar_classes = adj_vec_slice_one_hot.squeeze(-1).cpu().int().numpy()

        final_s_i_for_list = np.zeros(m_window_size, dtype=int)  # Pad to M
        if num_edges_to_predict_for_current_node > 0:
            final_s_i_for_list[:num_edges_to_predict_for_current_node] = processed_slice_scalar_classes

        list_s_i_scalar_class_vectors.append(final_s_i_for_list)  # Store S_i (scalar classes)

        # Update inputs for next iteration (node i+1)
        prev_adj_vec_input = adj_vec_generated_current_one_hot  # S_i
        prev_node_attr_input = current_node_attr_onehot.unsqueeze(0).unsqueeze(0)  # A_i

    final_node_attrs_np = np.array(list_node_attrs_one_hot)

    if not list_s_i_scalar_class_vectors or actual_num_nodes_generated == 0:
        return np.array([]), np.array([])  # Empty graph

    # m_seq_to_adj_mat expects N-1 sequences (S_0 to S_{N-2} if S_i is for node i)
    # list_s_i_scalar_class_vectors currently has N elements (S_0 to S_{N-1})
    if actual_num_nodes_generated == 1:
        # Only S_0 was generated, m_seq needs to be empty for a 1-node graph (no edges)
        m_seq_for_conversion = np.array([])
        adj_matrix_scalar_classes = np.zeros((1, 1), dtype=int)
    elif actual_num_nodes_generated > 1:
        # For N nodes, m_seq_to_adj_mat needs S_0, ..., S_{N-2} (N-1 vectors)
        # These are the M-window vectors for nodes 0 to N-2, defining connections for nodes 1 to N-1.
        # The vector list_s_i_scalar_class_vectors[k] is S_k.
        # So we take the first N-1 vectors from the list.
        m_seq_for_conversion = np.array(list_s_i_scalar_class_vectors[:-1])
        if m_seq_for_conversion.ndim == 1 and m_seq_for_conversion.size > 0:  # If N=2, m_seq is 1D
            m_seq_for_conversion = m_seq_for_conversion.reshape(1, -1)
        adj_matrix_scalar_classes = m_seq_to_adj_mat(m_seq_for_conversion, m_window_size)
    else:  # actual_num_nodes_generated == 0
        adj_matrix_scalar_classes = np.array([])

    # Ensure node attributes match the final number of nodes implied by adj_matrix
    # (which is actual_num_nodes_generated)
    if final_node_attrs_np.shape[0] != actual_num_nodes_generated:
        # This should ideally not happen if EOS is removed and logic is consistent
        print(
            f"Warning: Mismatch final_node_attrs ({final_node_attrs_np.shape[0]}) and actual_nodes_generated ({actual_num_nodes_generated}). "
            f"Trimming/padding node_attrs to {actual_num_nodes_generated}."
        )
        if final_node_attrs_np.shape[0] > actual_num_nodes_generated:
            final_node_attrs_np = final_node_attrs_np[:actual_num_nodes_generated]
        elif final_node_attrs_np.shape[0] < actual_num_nodes_generated and actual_num_nodes_generated > 0:
            # This case is less likely but handle by padding if necessary
            padding_needed = actual_num_nodes_generated - final_node_attrs_np.shape[0]
            # Default padding to a zero vector for node attributes
            attr_padding = np.zeros((padding_needed, num_node_classes))
            final_node_attrs_np = np.concatenate([final_node_attrs_np, attr_padding], axis=0)

    # Final check on adjacency matrix shape
    if adj_matrix_scalar_classes.shape != (actual_num_nodes_generated, actual_num_nodes_generated):
        if actual_num_nodes_generated > 0:
            print(
                f"Warning: Mismatch adj_matrix_scalar_classes shape ({adj_matrix_scalar_classes.shape}) "
                f"and actual_nodes_generated ({actual_num_nodes_generated}). Recreating zero matrix."
            )
            adj_matrix_scalar_classes = np.zeros((actual_num_nodes_generated, actual_num_nodes_generated), dtype=int)
        elif actual_num_nodes_generated == 0 and adj_matrix_scalar_classes.size > 0:
            adj_matrix_scalar_classes = np.array([])

    return adj_matrix_scalar_classes, final_node_attrs_np


def load_model_from_checkpoint(model_path, device):
    state = torch.load(model_path, map_location=device)
    config = state['config']
    m_window_size = config['data']['m']
    # Ensure these paths in config are correct for your saved checkpoint
    num_node_classes = config.get('model', {}).get('GraphRNN', {}).get('num_node_classes', 0)
    edge_feature_len_model = config.get('model', {}).get('GraphRNN', {}).get('edge_feature_len', 1)
    generation_mode = config.get('model', {}).get('mode', 'undirected')  # Default if not found

    if num_node_classes == 0:
        raise ValueError("'num_node_classes' not found or is 0 in loaded model config. Critical for generation.")
    if edge_feature_len_model != NUM_EDGE_FEATURES:  # NUM_EDGE_FEATURES from aig_config (should be 3)
        print(f"Warning: Model was trained with edge_feature_len={edge_feature_len_model}, "
              f"but current aig_config.NUM_EDGE_FEATURES is {NUM_EDGE_FEATURES}. Using model's value.")

    # Use edge_feature_len_model from the loaded checkpoint for model instantiation
    graph_rnn_config = config['model']['GraphRNN']
    graph_rnn_config['edge_feature_len'] = edge_feature_len_model  # Ensure loaded model uses its trained feature len

    if config['model']['edge_model'] == 'rnn':
        edge_rnn_config = config['model']['EdgeRNN']
        edge_rnn_config['edge_feature_len'] = edge_feature_len_model

        node_model = GraphLevelRNN(input_size=m_window_size,
                                   output_size=edge_rnn_config['hidden_size'],
                                   **graph_rnn_config).to(device)
        edge_model = EdgeLevelRNN(**edge_rnn_config).to(device)
        edge_gen_function = rnn_edge_gen
    else:  # MLP
        edge_mlp_config = config['model']['EdgeMLP']
        # EdgeLevelMLP's constructor needs edge_feature_len

        node_model = GraphLevelRNN(input_size=m_window_size,
                                   output_size=None,
                                   **graph_rnn_config).to(device)
        edge_model = EdgeLevelMLP(input_size_from_graph_rnn=graph_rnn_config['hidden_size'],
                                  mlp_hidden_size=edge_mlp_config['hidden_size'],
                                  num_edges_to_predict=m_window_size,
                                  edge_feature_len=edge_feature_len_model).to(device)  # Pass correct edge_feature_len
        edge_gen_function = mlp_edge_gen

    node_model.load_state_dict(state['node_model'])
    edge_model.load_state_dict(state['edge_model'])
    print(
        f"Loaded model from {model_path} with mode: {generation_mode}, M: {m_window_size}, "
        f"NodeClasses: {num_node_classes}, EdgeFeatLen (from model): {edge_feature_len_model}"
    )
    # Return edge_feature_len_model as this is what the loaded model expects/produces
    return node_model, edge_model, m_window_size, num_node_classes, edge_feature_len_model, edge_gen_function, generation_mode, config


# ==============================================================================
# Functions from/adapted from evaluate_aigs.py
# ==============================================================================

def get_node_type_str_from_attrs(attrs: dict) -> str:
    raw_type = attrs.get('type')
    if raw_type is None: return "UNKNOWN_MISSING_ATTR"
    if isinstance(raw_type, str): return raw_type  # Should not happen if type is list of floats
    if isinstance(raw_type, (list, np.ndarray)):
        try:
            # Convert to tuple of floats for dictionary key lookup
            type_tuple = tuple(float(f"{x:.6f}") for x in raw_type)  # Format to handle potential float precision issues
            return DECODING_NODE_TYPE_NX.get(type_tuple, f"UNKNOWN_ENCODING_{type_tuple}")
        except (ValueError, TypeError) as e:
            return f"UNKNOWN_VECTOR_CONVERSION_ERROR_{e}"
    return f"UNKNOWN_TYPE_FORMAT_{type(raw_type).__name__}"


def get_edge_type_str_from_attrs(attrs: dict) -> str:
    raw_type = attrs.get('type')
    if raw_type is None: return "UNKNOWN_MISSING_ATTR"
    if isinstance(raw_type, str): return raw_type
    if isinstance(raw_type, (list, np.ndarray)):
        try:
            type_tuple = tuple(float(f"{x:.6f}") for x in raw_type)
            return DECODING_EDGE_TYPE_NX.get(type_tuple, f"UNKNOWN_ENCODING_{type_tuple}")
        except (ValueError, TypeError) as e:
            return f"UNKNOWN_VECTOR_CONVERSION_ERROR_{e}"
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
        metrics['is_dag'] = 0.0;
        metrics['constraints_failed'].append("DAG Check Error")

    node_type_counts = Counter()
    for node_idx, data in G.nodes(data=True):
        node_type_str = get_node_type_str_from_attrs(data);
        node_type_counts[node_type_str] += 1

        if "UNKNOWN_ENCODING" in node_type_str or "UNKNOWN_TYPE_FORMAT" in node_type_str or "UNKNOWN_MISSING_ATTR" in node_type_str or "UNKNOWN_VECTOR_CONVERSION_ERROR" in node_type_str:
            metrics['num_unknown_nodes'] += 1
            if "Unknown node types" not in metrics['constraints_failed']: metrics['constraints_failed'].append(
                "Unknown node types")
            continue  # Skip degree checks for unknown node types

        if node_type_str not in NODE_TYPE_KEYS:  # Should be caught by UNKNOWN_ENCODING if DECODING_NODE_TYPE_NX is exhaustive
            metrics['num_unknown_nodes'] += 1
            if "Unknown node types" not in metrics['constraints_failed']: metrics['constraints_failed'].append(
                "Unknown node types")
            continue

        try:
            in_deg, out_deg = G.in_degree(node_idx), G.out_degree(node_idx)
        except Exception:  # Should not happen if node_idx is valid
            metrics['degree_check_errors'] += 1;
            continue

        if node_type_str == NODE_CONST0_EVAL:
            if in_deg != 0: metrics['const0_indegree_violations'] += 1
        elif node_type_str == NODE_PI_EVAL:
            if in_deg != 0: metrics['pi_indegree_violations'] += 1
        elif node_type_str == NODE_AND_EVAL:
            if in_deg != 2: metrics['and_indegree_violations'] += 1  # Strict check for final AIG
        elif node_type_str == NODE_PO_EVAL:
            if out_deg != 0: metrics['po_outdegree_violations'] += 1
            if in_deg == 0: metrics['po_indegree_violations'] += 1  # PO must have an input

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
        if "UNKNOWN_ENCODING" in edge_type_str or "UNKNOWN_TYPE_FORMAT" in edge_type_str or "UNKNOWN_MISSING_ATTR" in edge_type_str or "UNKNOWN_VECTOR_CONVERSION_ERROR" in edge_type_str:
            metrics['num_unknown_edges'] += 1
            if "Unknown edge types" not in metrics['constraints_failed']: metrics['constraints_failed'].append(
                "Unknown edge types")
        elif edge_type_str not in EDGE_TYPE_KEYS:  # Should be caught by UNKNOWN_ENCODING
            metrics['num_unknown_edges'] += 1
            if "Unknown edge types" not in metrics['constraints_failed']: metrics['constraints_failed'].append(
                "Unknown edge types")

    if metrics['num_pi'] == 0 and node_type_counts.get(NODE_CONST0_EVAL, 0) == 0:
        if metrics['num_nodes'] > 0:  # Only a constraint if graph is not empty
            metrics['constraints_failed'].append("No PIs or Const0")

    if metrics['num_and'] < MIN_AND_COUNT and metrics['num_nodes'] > 0:  # Only if not empty and ANDs are expected
        # This might be too strict if small non-logic graphs are possible
        pass  # metrics['constraints_failed'].append(f"AND gates < {MIN_AND_COUNT}")
    if metrics['num_po'] < MIN_PO_COUNT and metrics['num_nodes'] > 0:
        # This might be too strict
        pass  # metrics['constraints_failed'].append(f"POs < {MIN_PO_COUNT}")

    is_valid = (metrics['is_dag'] > 0.5 and
                metrics['num_unknown_nodes'] == 0 and
                metrics['num_unknown_edges'] == 0 and
                metrics['const0_indegree_violations'] == 0 and
                metrics['pi_indegree_violations'] == 0 and
                metrics['and_indegree_violations'] == 0 and
                metrics['po_outdegree_violations'] == 0 and
                metrics['po_indegree_violations'] == 0 and
                (metrics['num_pi'] > 0 or node_type_counts.get(NODE_CONST0_EVAL, 0) > 0 or metrics[
                    'num_nodes'] == 0) and  # Allow empty graph or graph with inputs
                (metrics['num_and'] >= MIN_AND_COUNT or metrics['num_nodes'] == 0 or (
                            metrics['num_pi'] > 0 and metrics['num_po'] > 0 and metrics[
                        'num_and'] == 0)) and  # Allow PI->PO without ANDs for simple cases
                (metrics['num_po'] >= MIN_PO_COUNT or metrics['num_nodes'] == 0)
                )
    metrics['is_structurally_valid'] = float(is_valid)
    if not is_valid and not metrics['constraints_failed'] and metrics['num_nodes'] > 0:
        metrics['constraints_failed'].append("General Validity Rules Failed")
    return dict(metrics)


def node_match_aig_flexible(node1_attrs, node2_attrs):
    type1_str = get_node_type_str_from_attrs(node1_attrs);
    type2_str = get_node_type_str_from_attrs(node2_attrs)
    # If one is unknown and the other is not, they don't match.
    # If both are unknown but different unknown strings, they don't match.
    is_type1_unknown = "UNKNOWN" in type1_str
    is_type2_unknown = "UNKNOWN" in type2_str
    if is_type1_unknown != is_type2_unknown: return False
    return type1_str == type2_str


def edge_match_aig_flexible(edge1_attrs, edge2_attrs):
    type1_str = get_edge_type_str_from_attrs(edge1_attrs);
    type2_str = get_edge_type_str_from_attrs(edge2_attrs)
    is_type1_unknown = "UNKNOWN" in type1_str
    is_type2_unknown = "UNKNOWN" in type2_str
    if is_type1_unknown != is_type2_unknown: return False
    return type1_str == type2_str


def are_graphs_isomorphic(G1: nx.DiGraph, G2: nx.DiGraph) -> bool:
    if G1 is None or G2 is None: return False
    if G1.number_of_nodes() != G2.number_of_nodes() or G1.number_of_edges() != G2.number_of_edges(): return False
    try:
        # Ensure node and edge match functions handle potential "UNKNOWN" types gracefully
        return nx.is_isomorphic(G1, G2, node_match=node_match_aig_flexible, edge_match=edge_match_aig_flexible)
    except Exception as e:
        # print(f"Isomorphism check failed: {e}") # Can be verbose
        return False


def calculate_uniqueness(valid_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    num_valid = len(valid_graphs);
    unique_graph_representatives = []
    if num_valid <= 1: return (1.0, num_valid)
    for i in tqdm(range(num_valid), desc="Checking Uniqueness", leave=False, ncols=80):
        current_graph = valid_graphs[i];
        is_unique_to_list = True
        for representative_graph in unique_graph_representatives:
            if are_graphs_isomorphic(current_graph, representative_graph):
                is_unique_to_list = False;
                break
        if is_unique_to_list: unique_graph_representatives.append(current_graph)
    num_unique = len(unique_graph_representatives)
    return (num_unique / num_valid if num_valid > 0 else 0.0), num_unique


def calculate_novelty(valid_graphs: List[nx.DiGraph], train_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    num_valid = len(valid_graphs);
    num_novel = 0
    if num_valid == 0: return (0.0, 0)
    if not train_graphs: return (1.0, num_valid)  # All are novel if no training set
    for gen_graph in tqdm(valid_graphs, desc="Checking Novelty", leave=False, ncols=80):
        is_novel_to_train_set = True
        for train_graph in train_graphs:
            if are_graphs_isomorphic(gen_graph, train_graph):
                is_novel_to_train_set = False;
                break
        if is_novel_to_train_set: num_novel += 1
    return (num_novel / num_valid if num_valid > 0 else 0.0), num_novel


def load_training_graphs_from_pkls(train_pkl_dir: str, train_pkl_prefix: str, num_files_to_load: int = 0) -> Optional[
    List[nx.DiGraph]]:
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
    if num_files_to_load > 0 and num_files_to_load < len(all_pkl_files):
        files_to_process = all_pkl_files[:num_files_to_load]
        print(f"Loading a subset of {len(files_to_process)} training PKL files.")
    elif num_files_to_load > 0:  # num_files_to_load >= len(all_pkl_files)
        print(f"Requested {num_files_to_load} files, found {len(all_pkl_files)}. Loading all found files.")
    else:  # num_files_to_load is 0, load all
        print(f"Loading all {len(all_pkl_files)} found training PKL files.")

    if not files_to_process: print("No training PKL files selected to load."); return None

    print(f"Loading training graphs from {len(files_to_process)} PKL files in {train_pkl_dir}:")

    all_train_graphs = []
    for filename in tqdm(files_to_process, desc="Loading Training PKLs", ncols=100):
        file_path = os.path.join(train_pkl_dir, filename)
        try:
            with open(file_path, 'rb') as f:
                # Specify encoding for compatibility, especially if PKLs were created with Python 2
                graphs_chunk = pickle.load(f, encoding='latin1')
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
# Helper Function to Convert Generated Output to NetworkX AIG (Corrected)
# ==============================================================================
def generated_output_to_nx_aig(adj_matrix_scalar_classes: np.ndarray,
                               node_attrs_one_hot: np.ndarray) -> nx.DiGraph:
    G = nx.DiGraph()
    num_nodes = node_attrs_one_hot.shape[0]
    if num_nodes == 0:
        return G

    # Add nodes with their one-hot 'type' attributes
    for i in range(num_nodes):
        G.add_node(i, type=list(node_attrs_one_hot[i, :]))

    # Add edges based on scalar classes, skipping "NO_EDGE" type
    if adj_matrix_scalar_classes.size > 0 and \
            adj_matrix_scalar_classes.shape == (num_nodes, num_nodes):
        for s_idx in range(num_nodes):  # Source node index
            for t_idx in range(num_nodes):  # Target node index
                # adj_matrix_scalar_classes[t_idx, s_idx] is the scalar class for edge s_idx -> t_idx
                # This indexing matches how m_seq_to_adj_mat populates: adj_mat[target, source]
                scalar_class_value = adj_matrix_scalar_classes[t_idx, s_idx]

                if 0 <= scalar_class_value < NUM_EDGE_FEATURES:  # Valid class index (0, 1, or 2)
                    try:
                        edge_type_key = EDGE_TYPE_KEYS[scalar_class_value]

                        # --- CORE CHANGE: Only add edge if it's not the "NO_EDGE" type ---
                        if edge_type_key != NO_EDGE_TYPE_NAME:
                            # For AIGs generated by GraphRNN, s_idx (source) < t_idx (target)
                            # should generally hold due to topological/BFS sequence.
                            if s_idx < t_idx:
                                one_hot_edge_type = EDGE_LABEL_ENCODING[edge_type_key]
                                G.add_edge(s_idx, t_idx, type=list(one_hot_edge_type))
                                # else:
                                # If s_idx >= t_idx, it implies a self-loop or backward edge in the sequence.
                                # GraphRNN's BFS-based generation typically produces adj[target, source]
                                # where source < target in the sequence.
                                # If m_seq_to_adj_mat can produce other structures, this might need review.
                                # For now, assume s_idx < t_idx for valid AIG edges from this generation process.
                                # print(f"Debug: Skipped edge {s_idx}->{t_idx} because s_idx not < t_idx for type {edge_type_key}")
                                pass


                    except IndexError:
                        print(
                            f"Warning: Scalar class {scalar_class_value} led to IndexError with EDGE_TYPE_KEYS. "
                            f"This should not happen if NUM_EDGE_FEATURES is correct. Skipping edge {s_idx}->{t_idx}."
                        )
                    except KeyError:
                        print(
                            f"Warning: Edge type key for scalar class {scalar_class_value} not in EDGE_LABEL_ENCODING. "
                            f"This indicates inconsistency between EDGE_TYPE_KEYS and EDGE_LABEL_ENCODING. Skipping edge {s_idx}->{t_idx}."
                        )
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
    aggregate_metrics_all = defaultdict(list)  # Store lists of metric values for averaging
    failed_constraints_summary = Counter()

    for i, G_aig in enumerate(tqdm(generated_nx_aigs, desc="AIG Validity Check", ncols=80)):
        if not isinstance(G_aig, nx.DiGraph):
            print(f"Warning: Item {i} is not a NetworkX DiGraph. Skipping.");
            failed_constraints_summary["Invalid Graph Object Type"] += 1;
            continue

        metrics = calculate_structural_aig_metrics(G_aig)
        for k_metric, v_metric_val in metrics.items():
            if k_metric == 'constraints_failed':
                if isinstance(v_metric_val, list):  # Ensure it's a list before iterating
                    for reason in v_metric_val: failed_constraints_summary[reason] += 1
            elif isinstance(v_metric_val, (float, int)):  # Only aggregate numeric metrics
                aggregate_metrics_all[k_metric].append(v_metric_val)

        if metrics.get('is_structurally_valid', 0.0) > 0.5:
            structurally_valid_aigs.append(G_aig)

    uniqueness_score, num_unique = calculate_uniqueness(structurally_valid_aigs)

    novelty_score, num_novel = (-1.0, -1)  # Default if no training data
    if training_graphs_nx is not None and len(training_graphs_nx) > 0:
        novelty_score, num_novel = calculate_novelty(structurally_valid_aigs, training_graphs_nx)
    elif structurally_valid_aigs:  # Has valid graphs, but no training data for comparison
        novelty_score, num_novel = 1.0, len(structurally_valid_aigs)
        print("Training graphs not provided or empty. Novelty is 100% relative to an empty set if valid graphs exist.")
    else:  # No valid graphs, novelty is 0
        novelty_score, num_novel = 0.0, 0

    # Capture print output to string buffer for saving to file
    original_stdout = sys.stdout;
    string_buffer = io.StringIO();
    sys.stdout = string_buffer  # Redirect print to buffer

    num_valid_structurally = len(structurally_valid_aigs)
    validity_fraction = (num_valid_structurally / num_total_generated) if num_total_generated > 0 else 0.0

    print("\n--- AIG V.U.N. Evaluation Summary ---")
    print(f"Total Graphs Generated & Evaluated: {num_total_generated}")
    print(f"Structurally Valid AIGs (V)     : {num_valid_structurally} ({validity_fraction * 100:.2f}%)")

    if num_valid_structurally > 0:
        print(f"Unique Valid AIGs             : {num_unique}")
        print(f"Uniqueness (U) among valid    : {uniqueness_score:.4f} ({uniqueness_score * 100:.2f}%)")
        if novelty_score != -1.0:  # Check if novelty was calculated
            print(f"Novel Valid AIGs vs Train Set : {num_novel}")
            print(f"Novelty (N) among valid       : {novelty_score:.4f} ({novelty_score * 100:.2f}%)")
        else:  # Novelty not calculated due to no training data
            print(
                f"Novelty (N) among valid       : Not calculated (no training data provided or training data was empty)")
    else:  # No structurally valid AIGs
        print(f"Uniqueness (U) among valid    : N/A (0 valid graphs)");
        print(f"Novelty (N) among valid       : N/A (0 valid graphs)")

    print("\n--- Average Structural Metrics (All Processed Graphs) ---")
    if num_total_generated > 0:  # Only print if graphs were processed
        # Calculate averages from collected lists
        avg_metrics_calculated = {}
        for key, values_list in aggregate_metrics_all.items():
            if values_list:  # If list is not empty
                avg_metrics_calculated[key] = np.mean(values_list)
                std_metrics_calculated = np.std(values_list) if len(values_list) > 1 else 0.0
                if key == 'is_dag':  # Specific formatting for percentages
                    print(f"  - Percentage {key:<22}: {avg_metrics_calculated[key] * 100:.2f}%")
                elif key != 'is_structurally_valid':  # Don't average the validity flag itself here
                    print(f"  - Avg {key:<27}: {avg_metrics_calculated[key]:.3f} (Std: {std_metrics_calculated:.3f})")
            else:  # Metric was defined but no values collected (e.g. all graphs empty)
                print(f"  - Avg {key:<27}: N/A (no data)")
        if not avg_metrics_calculated:
            print("  No numeric structural metrics data collected.")
    else:
        print("  No graphs processed, so no structural metrics to average.")

    print("\n--- Constraint Violation Summary (Across All Graphs Attempted) ---")
    num_graphs_with_any_constraint_failure = sum(
        1 for g in generated_nx_aigs if calculate_structural_aig_metrics(g)['constraints_failed'])

    if not failed_constraints_summary:
        print("  No structural constraint violations detected across all graphs.")
    else:
        sorted_reasons = sorted(failed_constraints_summary.items(), key=lambda item: item[1], reverse=True)
        print(f"  (Violations summarized across {num_total_generated} graphs attempted)")
        for reason, count in sorted_reasons:
            # Percentage relative to total generated, as a single graph can have multiple reasons
            reason_percentage_total = (count / num_total_generated) * 100 if num_total_generated > 0 else 0
            print(
                f"  - {reason:<45}: {count:<6} occurrences ({reason_percentage_total:.1f}% of total graphs had this issue at least once)")

    print("------------------------------------")

    results_output_str = string_buffer.getvalue();
    sys.stdout = original_stdout;  # Restore standard output
    string_buffer.close()

    print(results_output_str)  # Print to console

    try:
        with open(results_filename, 'w') as res_file:
            res_file.write(f"Model Checkpoint: {args.model_checkpoint_path if 'args' in locals() else 'N/A'}\n")
            res_file.write(f"Number of Graphs Generated for Eval: {num_total_generated}\n")
            if 'args' in locals() and args.num_nodes is not None:
                res_file.write(f"Target Nodes Per Graph: {args.num_nodes}\n")
            elif 'args' in locals():
                res_file.write(
                    f"Target Nodes Per Graph: Varied ({args.min_nodes_generate}-{args.max_nodes_generate})\n")
            if training_graphs_nx is not None:
                res_file.write(
                    f"Training AIGs for Novelty: {len(training_graphs_nx)} from {args.train_set_pkl_dir if 'args' in locals() else 'N/A'}\n")
            else:
                res_file.write("Training AIGs for Novelty: Not loaded or not used.\n")
            res_file.write("-------------------------------------\n")
            res_file.write(results_output_str)
        print(f"Evaluation results also saved to {results_filename}")
    except IOError as e_io:
        print(f"Error: Failed to write results to file {results_filename}: {e_io}", file=sys.stderr)

    # Prepare summary dictionary for return
    eval_summary_dict = {
        'total_generated': num_total_generated,
        'num_valid': num_valid_structurally,
        'validity_fraction': validity_fraction,
        'num_unique_valid': num_unique if num_valid_structurally > 0 else 0,
        'uniqueness_score': uniqueness_score if num_valid_structurally > 0 else 0.0,
        'num_novel_valid': num_novel if num_valid_structurally > 0 and novelty_score != -1.0 else 0,
        'novelty_score': novelty_score if num_valid_structurally > 0 and novelty_score != -1.0 else 0.0,
        'avg_metrics': avg_metrics_calculated if num_total_generated > 0 else {},
        'failed_constraints_summary': dict(failed_constraints_summary)
    }
    return eval_summary_dict


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
    parser.add_argument('--max_nodes_generate', type=int, default=MAX_NODE_COUNT,  # From aig_config
                        help='Max nodes if --num_nodes is None.')

    parser.add_argument('--train_set_pkl_dir', type=str, default=None,  # Default to None
                        help='(Optional) Directory containing training PKL files for novelty.')
    parser.add_argument('--train_set_pkl_prefix', type=str, default="real_aigs_part_",
                        help='(Optional) Prefix of training PKL files.')
    parser.add_argument('--num_train_pkl_files', type=int, default=0,  # Default to 0 (load all found)
                        help='(Optional) Number of training PKL files to load (0 for all found).')

    parser.add_argument('--results_file', type=str, default="generated_aig_eval_results.txt",
                        help="Filename to save the evaluation summary.")
    parser.add_argument('--output_generated_pkl', type=str, default=None,
                        help="(Optional) Path to save generated NetworkX AIGs.")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(args.model_checkpoint_path):
        print(f"Error: Model checkpoint not found at {args.model_checkpoint_path}");
        sys.exit(1)

    try:
        (node_model, edge_model, m_param,
         num_node_classes_model, edge_feat_len_from_model,  # This is what the model was trained with
         edge_gen_func, model_mode, model_config_loaded) = load_model_from_checkpoint(args.model_checkpoint_path,
                                                                                      device)
    except Exception as e:
        print(f"Error loading model: {e}");
        # import traceback; traceback.print_exc(); # For more detailed error
        sys.exit(1)

    # Critical check: The loaded model's edge feature length must match current config for 3 types
    if edge_feat_len_from_model != NUM_EDGE_FEATURES:
        print(f"FATAL ERROR: Model was trained with edge_feature_len = {edge_feat_len_from_model}, "
              f"but current aig_config.py expects NUM_EDGE_FEATURES = {NUM_EDGE_FEATURES} (for 3 edge types).")
        print("Please ensure the loaded model checkpoint was trained with the 3-edge-type system.")
        sys.exit(1)

    print(f"\nGenerating {args.num_graphs} graphs...")
    generated_nx_aigs_list = []
    raw_generated_outputs = []  # To store (adj_matrix_scalar_classes, final_node_attrs_np)

    for i in tqdm(range(args.num_graphs), desc="Generating Graphs", ncols=100):
        if args.num_nodes is not None:
            nodes_to_generate_for_this_graph = args.num_nodes
        else:
            # Ensure min_nodes is not greater than max_nodes
            min_gen_nodes = max(1, args.min_nodes_generate)  # At least 1 node
            max_gen_nodes = max(min_gen_nodes, args.max_nodes_generate)
            nodes_to_generate_for_this_graph = np.random.randint(min_gen_nodes, max_gen_nodes + 1)

        if nodes_to_generate_for_this_graph <= 0:
            # This case should ideally be prevented by min_gen_nodes=max(1,...)
            raw_generated_outputs.append((np.array([]), np.array([])));
            continue

        # generate_single_graph uses edge_feat_len_from_model for its internal logic
        adj_matrix_scalar_classes, final_node_attrs_np = generate_single_graph(
            num_target_nodes=nodes_to_generate_for_this_graph,
            node_model=node_model, edge_model=edge_model,
            m_window_size=m_param,
            num_node_classes=num_node_classes_model,
            edge_feature_len_model=edge_feat_len_from_model,  # Use model's trained feature length
            edge_gen_function=edge_gen_func,
            generation_mode=model_mode
        )
        raw_generated_outputs.append((adj_matrix_scalar_classes, final_node_attrs_np))

    if not raw_generated_outputs:
        print("No raw graph outputs were generated. Exiting evaluation.");
        sys.exit(0)
    print(f"Successfully generated {len(raw_generated_outputs)} raw graph outputs.")

    print("\nConverting generated outputs to NetworkX AIGs for evaluation...")
    for adj_matrix, node_attrs in tqdm(raw_generated_outputs, desc="Converting to NX", ncols=100):
        # Check if node_attrs is not empty before trying to get num_nodes from it
        if node_attrs.size > 0:
            # adj_matrix can be empty for 1-node graphs, handle this
            if adj_matrix.size == 0 and node_attrs.shape[0] == 1:  # 1-node graph, no edges
                nx_aig = nx.DiGraph()
                nx_aig.add_node(0, type=list(node_attrs[0, :]))
            elif adj_matrix.size > 0 and node_attrs.size > 0:
                nx_aig = generated_output_to_nx_aig(adj_matrix, node_attrs)
            else:  # Both empty or inconsistent
                nx_aig = nx.DiGraph()
        else:  # node_attrs is empty, so it's an empty graph
            nx_aig = nx.DiGraph()
        generated_nx_aigs_list.append(nx_aig)

    if args.output_generated_pkl:
        print(f"Saving {len(generated_nx_aigs_list)} generated NetworkX AIGs to {args.output_generated_pkl}...")
        try:
            with open(args.output_generated_pkl, 'wb') as f_out:
                pickle.dump(generated_nx_aigs_list, f_out)
            print("Saved generated graphs.")
        except Exception as e_save:
            print(f"Error saving generated graphs: {e_save}")

    training_aigs_for_novelty = None
    if args.train_set_pkl_dir and args.train_set_pkl_prefix:
        print(
            f"\nLoading training AIGs for novelty check from dir: {args.train_set_pkl_dir} with prefix: {args.train_set_pkl_prefix}"
        )
        training_aigs_for_novelty = load_training_graphs_from_pkls(
            args.train_set_pkl_dir,
            args.train_set_pkl_prefix,
            args.num_train_pkl_files
        )
        if training_aigs_for_novelty:
            print(f"Loaded {len(training_aigs_for_novelty)} training AIGs.")
        else:
            print(
                f"Warning: Could not load training AIGs from {args.train_set_pkl_dir}. Novelty will be 100% if valid graphs exist.")
    else:
        print(
            "\nTraining data directory or prefix not specified. Novelty will be 100% if valid graphs exist (relative to empty set).")

    evaluation_summary_dict = run_evaluation_suite(
        generated_nx_aigs_list,
        training_graphs_nx=training_aigs_for_novelty,
        results_filename=args.results_file
    )
    print("\nGeneration and Evaluation finished.")
