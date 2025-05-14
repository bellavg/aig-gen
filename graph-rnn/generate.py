'''Code to use trained GraphRNN to generate a new graph, now with node attributes.'''

import argparse
import numpy as np
import torch

# Ensure this is the model.py version that supports node attribute prediction
from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP
import evaluate # evaluate.draw_generated_graph is used

def m_seq_to_adj_mat(m_seq, m):
    """
    Converts a sequence of M-vectors (representing connections to M previous nodes)
    into a full adjacency matrix.
    Args:
        m_seq (np.array): Shape [num_generated_nodes - 1, M_window].
                          Each row is S_i, representing connections of node i+1
                          to its M predecessors. Values are typically binary or scalar class indices.
        m (int): The M-window size.
    Returns:
        np.array: An N x N adjacency matrix, where N = m_seq.shape[0] + 1.
    """
    num_generated_sequences = m_seq.shape[0]
    n = num_generated_sequences + 1 # Total number of nodes in the graph
    adj_mat = np.zeros((n, n))

    for i in range(num_generated_sequences): # i from 0 to N-2
        # prev_nodes is S_i from the paper, which defines connections for node i+1
        prev_nodes_connections = m_seq[i, :] # Shape [M]
        current_node_idx = i + 1 # Node being connected (from 1 to N-1)

        # Determine how many actual predecessors this current_node_idx can connect to
        # (up to M, and up to its own index)
        num_potential_connections = min(current_node_idx, m)

        # Connections are to nodes: current_node_idx-1, current_node_idx-2, ..., current_node_idx-M
        # The prev_nodes_connections[0] is for edge to current_node_idx-1
        # prev_nodes_connections[k] is for edge to current_node_idx-(k+1)
        # The adj_mat is filled such that adj_mat[row, col] = 1 if edge row -> col
        # Here, S_i (prev_nodes_connections) defines incoming edges to current_node_idx
        # So, adj_mat[predecessor, current_node_idx]

        # The original GraphRNN paper stores S_i such that S_i(j) is an edge between node i and node i-j.
        # The provided code's m_seq_to_adj_mat:
        # adj_mat[i+1, max(i+1-m, 0) : i+1] = list(reversed(prev_nodes[:i+1 - max(i+1-m, 0)]))
        # This means adj_mat[current_node_idx, start_pred_idx : current_node_idx]
        # The prev_nodes are connections from current_node_idx to its predecessors.
        # If S_i(j) is edge (node i+1, node i+1-j), then reversed list is S_i(1), S_i(2)...
        # adj_mat[current_node_idx, current_node_idx-1] = S_i(1) (from reversed list's 0th element)
        # adj_mat[current_node_idx, current_node_idx-2] = S_i(2) (from reversed list's 1st element)
        # This seems to represent S_i as outgoing edges from current_node_idx to its predecessors.
        # However, GraphRNN typically generates S_i as incoming edges to current_node_idx from predecessors.
        # Let's stick to the provided formula.

        # Number of elements to take from prev_nodes_connections for this row
        # This corresponds to connections from node (i+1) to (i), (i-1), ..., (i+1 - num_to_take_from_m_vec + 1)
        num_to_take_from_m_vec = min(current_node_idx, m)

        # The slice `prev_nodes_connections[:num_to_take_from_m_vec]` contains connections
        # to the `num_to_take_from_m_vec` most recent predecessors.
        # `reversed` means the connection to the immediate predecessor (i) is last in the reversed list.
        # `adj_mat[current_node_idx, col_idx]`
        # Example: m=3, current_node_idx = 2 (node S2). Connects to S1, S0. num_to_take=2.
        # prev_nodes_connections = [s2_to_s1, s2_to_s0, 0]. Slice is [s2_to_s1, s2_to_s0]. Reversed is [s2_to_s0, s2_to_s1]
        # adj_mat[2, 0] = s2_to_s0
        # adj_mat[2, 1] = s2_to_s1
        # This means adj_mat[current, predecessor_index_from_0]

        start_col_index = max(0, current_node_idx - num_to_take_from_m_vec) # Corrected start index calculation

        # Values to assign: connections from current_node_idx to its predecessors
        # The prev_nodes_connections are ordered from most recent predecessor to least recent
        # e.g. prev_nodes_connections[0] is connection to (current_node_idx-1)
        #      prev_nodes_connections[1] is connection to (current_node_idx-2)
        # We want adj_mat[current_node_idx, predecessor_original_index]
        # The slice `prev_nodes_connections[:num_to_take_from_m_vec]` gets the relevant M connections.
        # `list(reversed(prev_nodes_connections[:num_to_take_from_m_vec]))` means
        # the first element is connection to oldest considered predecessor.
        # Example: current_node_idx=2 (node S2). m=12. num_to_take_from_m_vec = 2.
        # prev_nodes_connections has S2's M connections. Slice gets first 2.
        # adj_mat[2, 0:2] = reversed([S2_to_S1, S2_to_S0]) = [S2_to_S0, S2_to_S1]
        # So adj_mat[2,0] = S2_to_S0, adj_mat[2,1] = S2_to_S1. This is correct.

        assign_slice = list(reversed(prev_nodes_connections[:num_to_take_from_m_vec]))
        adj_mat[current_node_idx, start_col_index : current_node_idx] = assign_slice

    return adj_mat


def sample_bernoulli(p_tensor):
    """ Samples from a Bernoulli distribution. p_tensor contains probabilities. """
    return torch.bernoulli(p_tensor).int()


def sample_softmax(logits_tensor):
    """ Samples a one-hot vector from a categorical distribution given logits. """
    probabilities = torch.softmax(logits_tensor, dim=-1)
    sampled_indices = torch.multinomial(probabilities, 1).squeeze(-1)
    one_hot = torch.nn.functional.one_hot(sampled_indices, num_classes=logits_tensor.shape[-1])
    return one_hot.float()


def rnn_edge_gen(edge_rnn, h_for_edge_model, num_edges_to_predict,
                 m_window_size, sample_fun_edges, edge_feature_len, attempts=None):
    """
    Generates the edges for the current node using the EdgeLevelRNN.
    Args:
        edge_rnn: The EdgeLevelRNN model.
        h_for_edge_model: Hidden state from GraphLevelRNN to initialize EdgeLevelRNN.
                          Shape [1, 1, hidden_size_for_edge_rnn_init].
        num_edges_to_predict: Actual number of edges to predict for this node (min(current_node_idx, m_window_size)).
        m_window_size: The M parameter, max number of potential edges. Output adj_vec will be padded to this.
        sample_fun_edges: Function to sample edge values (e.g., sample_bernoulli, sample_softmax then argmax).
        edge_feature_len: Dimensionality of edge features.
    Returns:
        torch.Tensor: Adjacency vector for the current node, padded to m_window_size.
                      Shape [1, 1, m_window_size, edge_feature_len].
    """
    device = h_for_edge_model.device
    # Output adjacency vector, padded to m_window_size
    adj_vec_output = torch.zeros([1, 1, m_window_size, edge_feature_len], device=device)

    if num_edges_to_predict == 0:
        return adj_vec_output

    edge_rnn.set_first_layer_hidden(h_for_edge_model.squeeze(1)) # Squeeze seq_len dim if h is [1,1,H]

    # SOS token for edge sequence
    current_edge_input_features = torch.ones([1, 1, edge_feature_len], device=device)

    for i in range(num_edges_to_predict):
        edge_pred_logits_or_probs = edge_rnn(current_edge_input_features) # Input [1,1,feat], Output [1,1,feat]

        # Sample the edge features for the current edge
        # sample_fun_edges should handle the output of edge_rnn (logits or probs)
        # and return a tensor of shape [edge_feature_len]
        sampled_edge_features = sample_fun_edges(edge_pred_logits_or_probs.squeeze(0).squeeze(0)) # Pass [feat]

        adj_vec_output[0, 0, i, :] = sampled_edge_features
        current_edge_input_features[0, 0, :] = sampled_edge_features # Teacher forcing with generated sample

    return adj_vec_output


def mlp_edge_gen(edge_mlp, h_for_edge_model, num_edges_to_predict,
                 m_window_size, sample_fun_edges, edge_feature_len, attempts=1):
    """
    Generates the edges for the current node using the EdgeLevelMLP.
    Args:
        h_for_edge_model: Hidden state from GraphLevelRNN. Shape [1, 1, hidden_size_for_mlp_input].
        num_edges_to_predict: Actual number of edges to predict.
        m_window_size: Max number of potential edges (M).
        sample_fun_edges: Function to sample edge values.
        edge_feature_len: Dimensionality of edge features.
    Returns:
        torch.Tensor: Adjacency vector. Shape [1, 1, m_window_size, edge_feature_len].
    """
    device = h_for_edge_model.device
    adj_vec_output = torch.zeros([1, 1, m_window_size, edge_feature_len], device=device)

    if num_edges_to_predict == 0:
        return adj_vec_output

    # MLP predicts all num_edges_to_predict edge features at once
    # edge_mlp output: [1, 1, num_edges_to_predict_in_mlp_config, edge_feature_len]
    # The num_edges_to_predict_in_mlp_config is usually m_window_size.
    edge_predictions_all = edge_mlp(h_for_edge_model)
    # Shape: [1, 1, m_window_size, edge_feature_len]

    for _ in range(attempts):
        for i in range(num_edges_to_predict):
            # Sample features for the i-th edge
            sampled_edge_features = sample_fun_edges(edge_predictions_all[0, 0, i, :]) # Pass [feat]
            adj_vec_output[0, 0, i, :] = sampled_edge_features

        if (adj_vec_output[0, 0, :num_edges_to_predict, :] != 0).any(): # Check if any edge was sampled
            break
    return adj_vec_output


def generate(num_total_nodes, node_model, edge_model,
             m_window_size, num_node_classes, edge_feature_len_model, # edge_feature_len_model is what the models were trained with
             edge_gen_function, mode, edge_sample_attempts=1):
    """
    Generates a graph with node attributes.
    Args:
        num_total_nodes (int): The desired number of nodes in the output graph.
        node_model (GraphLevelRNN): Trained graph-level RNN.
        edge_model (EdgeLevelRNN or EdgeLevelMLP): Trained edge-level model.
        m_window_size (int): The M parameter.
        num_node_classes (int): Number of possible node types.
        edge_feature_len_model (int): Dimensionality of edge features models were trained with.
        edge_gen_function: rnn_edge_gen or mlp_edge_gen.
        mode (str): Generation mode (e.g., 'undirected', 'directed-multiclass', 'aig-custom-topsort').
        edge_sample_attempts (int): For MLP edge generation.
    Returns:
        Tuple (np.array, np.array):
            - Adjacency matrix of the generated graph.
            - Array of generated node attributes (one-hot encoded).
    """
    device = next(node_model.parameters()).device
    node_model.eval()
    edge_model.eval()

    # Define sampling function for EDGES based on mode
    # This sample_fun_edges will be used by rnn_edge_gen/mlp_edge_gen
    # It should take logits/probabilities for a single edge slot and return sampled features (tensor of shape [edge_feature_len_model])
    if mode == 'directed-multiclass' or (mode == 'aig-custom-topsort' and edge_feature_len_model > 1) : # Assuming AIG uses softmax for its 2+ edge types
        # sample_softmax returns a one-hot vector
        sample_fun_edges = lambda logits: sample_softmax(logits) # expects logits
    else: # Undirected, binary directed-topsort, or AIG with edge_feature_len_model=1 (binary edge)
        # sample_bernoulli expects probabilities and returns a scalar (0 or 1)
        # Needs to be wrapped to return a tensor of shape [1] if edge_feature_len_model is 1
        sample_fun_edges = lambda probs: sample_bernoulli(probs).view(1)


    # Initialize the first "previous" adjacency vector (S_{-1}) - typically SOS (all ones)
    # This is input to GraphLevelRNN for generating the first node (node 0)
    # Shape: [1, 1, m_window_size, edge_feature_len_model]
    prev_adj_vec_input = torch.ones([1, 1, m_window_size, edge_feature_len_model], device=device)

    # Initialize the first "previous" node attribute (A_{-1}) - typically SOS (all zeros or specific class)
    # Shape: [1, 1, num_node_classes]
    prev_node_attr_input = torch.zeros([1, 1, num_node_classes], device=device)
    # If you have a specific SOS node class, one-hot encode it here:
    # prev_node_attr_input[0,0,SOS_CLASS_IDX] = 1.0

    list_adj_vecs_for_matrix = [] # Stores processed adjacency vectors (usually scalar class indices) for m_seq_to_adj_mat
    list_node_attrs_generated = []  # Stores generated one-hot node attributes

    node_model.reset_hidden()

    # Loop to generate num_total_nodes (from 0 to N-1)
    # The loop runs N times. In iteration `i`, we generate node `i`.
    # `list_adj_vecs_for_matrix` will store N S_i vectors if S_0 is also stored.
    # Original code loops `1` to `num_nodes` (exclusive of `num_nodes` if it means count).
    # If num_total_nodes is N, loop N times for node 0 to N-1.
    # The S_i sequence for m_seq_to_adj_mat should be N-1 long if N is total nodes.

    for i in range(num_total_nodes): # Generates node 0, 1, ..., num_total_nodes-1
        # Generate current node's attributes and hidden state for edge model
        # node_model input: S_{i-1}, A_{i-1}
        # node_model output: h_i (for EdgeModel of node i), logits for A_i
        h_for_edge_model, node_attr_logits = node_model(prev_adj_vec_input, prev_node_attr_input)
        # h_for_edge_model shape: [1, 1, H_graph]
        # node_attr_logits shape: [1, 1, num_node_classes]

        # Sample current node's attribute (A_i)
        current_node_attr_onehot = sample_softmax(node_attr_logits.squeeze(0).squeeze(0)) # Pass logits of shape [num_node_classes]
        list_node_attrs_generated.append(current_node_attr_onehot.cpu().numpy()) # Store one-hot A_i

        # Determine number of edges to predict for current node i
        # Node i can connect to min(i, m_window_size) previous nodes.
        num_edges_to_predict_for_current_node = min(i, m_window_size)

        # Generate edge connections for current node i (S_i)
        # adj_vec_generated_current is S_i, shape [1, 1, m_window_size, edge_feature_len_model]
        adj_vec_generated_current = edge_gen_function(
            edge_model,
            h_for_edge_model, # h_i
            num_edges_to_predict_for_current_node,
            m_window_size,
            sample_fun_edges, # Sampler for edge features
            edge_feature_len_model,
            attempts=edge_sample_attempts
        )

        # Prepare S_i for m_seq_to_adj_mat (needs to be scalar or 1D array of scalars per node)
        # This part needs to be mode-dependent, similar to original.
        # The slice [:num_edges_to_predict_for_current_node] is important.

        # We only add to list_adj_vecs_for_matrix if i > 0, because m_seq_to_adj_mat
        # expects N-1 sequences for N nodes. S_0 is not used by it.
        # However, GraphRNN generates S_0 (connections of node 0 to its M predecessors - none).
        # The original loop was `for i in range(1, num_nodes):` which means `list_adj_vecs` had N-1 elements.
        # Let's adjust: generate N nodes, but `list_adj_vecs_for_matrix` stores N-1 S_i vectors (S_1 to S_N-1 from paper's view)
        # or S_0 to S_{N-2} if S_i is adj for node i.
        # If S_i means "edges of node i", then we need N of them.
        # `m_seq_to_adj_mat` expects `m_seq.shape[0]` to be `num_generated_nodes - 1`.
        # So, `list_adj_vecs_for_matrix` should have `num_total_nodes - 1` entries.
        # This means we store the `adj_vec_generated_current` (which is S_i) starting from the generation of node 1.
        # The S_0 (adj_vec for node 0) is used as input for node 1, but not stored for m_seq_to_adj_mat.

        if i < num_total_nodes : # Store S_i for node i (0 to N-1) if we adapt m_seq_to_adj_mat
                                 # Or, only store if i > 0 for original m_seq_to_adj_mat.
                                 # Let's assume S_i is connections for node i.
                                 # The loop for m_seq_to_adj_mat is `num_generated_sequences` (N-1).
                                 # So `list_adj_vecs_for_matrix` should contain N-1 elements.
                                 # These are S_0, S_1, ..., S_{N-2} if S_i is for node i.
                                 # Or S_1, ..., S_{N-1} if S_i is for node i in paper.

            # Let's follow original: list_adj_vecs stores N-1 elements.
            # These are the "rows" of the BFS matrix in the paper (adj for node 1, node 2...).
            # So, we store adj_vec_generated_current (S_i) if i > 0.
            # No, the loop in m_seq_to_adj_mat `for i in range(num_generated_sequences)`
            # means `m_seq[i]` is the M-vector for node `i+1`.
            # So `list_adj_vecs_for_matrix` should store the M-vectors for nodes 1 to N.
            # This means it should have N-1 entries.
            # The M-vector for node 1 is generated when i=0 (current_node_attr is A0),
            # but it's based on h0, and represents S1 (connections for node 1).
            # This is confusing.

            # Simpler: `list_adj_vecs_for_matrix` stores the M-window connection vector for each generated node.
            # If we generate N nodes (0 to N-1), we have N such vectors.
            # Let `adj_vec_slice_to_store` be the connections for the current node `i`.
            # This slice should be `[m_window_size]` with scalar values.

            adj_vec_slice = adj_vec_generated_current[0, 0, :num_edges_to_predict_for_current_node, :] # Shape [num_pred_edges, edge_feat_len]

            if mode == 'undirected' or mode == 'directed-topsort': # Original binary modes
                # These modes expect scalar binary edge indicators.
                # sample_fun_edges for these returns [1] or [0]. We take .item() or [:,0]
                processed_slice_for_storage = adj_vec_slice[:,0].cpu().int().numpy()
            elif mode == 'directed-multiclass' or (mode == 'aig-custom-topsort' and edge_feature_len_model > 1):
                # These modes use softmax, output is one-hot. Convert to scalar class index.
                # adj_vec_slice is already one-hot from sample_fun_edges.
                processed_slice_for_storage = torch.argmax(adj_vec_slice, dim=-1).cpu().int().numpy()
            else: # Fallback or AIG with edge_feature_len_model=1
                processed_slice_for_storage = adj_vec_slice[:,0].cpu().int().numpy()

            # Pad this processed_slice_for_storage to full m_window_size for consistent shape in list_adj_vecs_for_matrix
            final_slice_for_storage = np.zeros(m_window_size, dtype=int)
            final_slice_for_storage[:num_edges_to_predict_for_current_node] = processed_slice_for_storage
            list_adj_vecs_for_matrix.append(final_slice_for_storage)


            # EOS condition: if the S_i generated for the current node `i` is all zeros (no edges)
            # Check the raw `adj_vec_generated_current` before processing for storage
            if torch.all(adj_vec_generated_current[0,0,:num_edges_to_predict_for_current_node,:] == 0):
                if i > 0 : # Avoid stopping after generating only node 0 if it has no edges
                    # print(f"EOS detected at node {i}. No edges generated.")
                    # Remove the all-zero S_i and corresponding A_i before breaking
                    # list_adj_vecs_for_matrix.pop()
                    # list_node_attrs_generated.pop() # This would shorten num_total_nodes
                    # The original GraphRNN paper stops if S_i is all zeros.
                    # This means the graph ends at node i-1.
                    # So we effectively generated i nodes (0 to i-1).
                    # The list_adj_vecs_for_matrix would have i entries (S0 to Si-1)
                    # list_node_attrs_generated would have i entries (A0 to Ai-1)
                    # We need to adjust num_total_nodes to `i` before m_seq_to_adj_mat
                    num_total_nodes = i # Graph has i nodes (0 to i-1)
                    list_adj_vecs_for_matrix.pop() # Remove the all-zero S_i
                    list_node_attrs_generated.pop() # Remove A_i for the non-existent node i
                    break

        # Update "previous" inputs for the next iteration (i+1)
        prev_adj_vec_input = adj_vec_generated_current # S_i becomes input for S_{i+1}
        prev_node_attr_input = current_node_attr_onehot.unsqueeze(0).unsqueeze(0) # A_i becomes input for A_{i+1}

    # Post-processing
    # list_adj_vecs_for_matrix should have num_total_nodes entries if EOS wasn't hit early.
    # m_seq_to_adj_mat expects N-1 sequences for N nodes.
    # If list_adj_vecs_for_matrix has N entries (S0 to SN-1), we pass S0 to S_{N-2} to it.
    if not list_adj_vecs_for_matrix: # Handle case where num_total_nodes was 0 or 1 and EOS hit.
        final_adj_matrix = np.zeros((num_total_nodes, num_total_nodes))
        final_node_attrs = np.array(list_node_attrs_generated) # Might be empty or have one entry
        if num_total_nodes > 0 and final_node_attrs.shape[0] == 0 and list_node_attrs_generated: # if num_total_nodes became 1 due to EOS
             final_node_attrs = np.array(list_node_attrs_generated)

        return final_adj_matrix, final_node_attrs


    # Adjust list_adj_vecs_for_matrix for m_seq_to_adj_mat if it contains S_0
    # m_seq_to_adj_mat(m_seq, m) where m_seq.shape[0] = (number of nodes in final matrix) - 1
    # If list_adj_vecs_for_matrix has `num_total_nodes` elements (S_0 to S_{N-1})
    # and we want an N x N matrix, then m_seq should be S_0 to S_{N-2} (length N-1).

    if len(list_adj_vecs_for_matrix) == num_total_nodes :
        m_seq_for_conversion = np.array(list_adj_vecs_for_matrix[:-1]) # Use S0..SN-2 for N node matrix
    elif len(list_adj_vecs_for_matrix) == num_total_nodes - 1:
        m_seq_for_conversion = np.array(list_adj_vecs_for_matrix) # Already N-1 elements
    else: # Mismatch due to EOS or other logic
        # This case needs robust handling. For now, use what we have.
        # This might lead to smaller matrix than num_total_nodes if EOS was aggressive.
        m_seq_for_conversion = np.array(list_adj_vecs_for_matrix)
        num_total_nodes = m_seq_for_conversion.shape[0] + 1


    if m_seq_for_conversion.ndim == 1: # if only one sequence was added (e.g. num_total_nodes = 2)
        m_seq_for_conversion = m_seq_for_conversion.reshape(1, -1)

    if m_seq_for_conversion.shape[0] == 0 and num_total_nodes > 0 : # e.g. num_total_nodes = 1
         adj_matrix_raw = np.zeros((num_total_nodes, num_total_nodes))
    elif m_seq_for_conversion.shape[0] == 0 and num_total_nodes == 0 :
         adj_matrix_raw = np.array([]) # empty graph
    else:
        adj_matrix_raw = m_seq_to_adj_mat(m_seq_for_conversion, m_window_size)


    # Final adjacency matrix construction based on mode (as in original)
    # This part assumes adj_matrix_raw contains scalar class indices for edges.
    if mode == 'undirected':
        adj_matrix_final = np.tril(adj_matrix_raw, k=-1) # Take lower triangle
        adj_matrix_final = adj_matrix_final + adj_matrix_final.T
    elif mode == 'directed-multiclass': # Original 4-class interpretation
        # adj_matrix_raw has classes 0,1,2,3. Reconstruct directed adj.
        # (0: No edge, 1: edge ->, 2: edge <-, 3: edge <>)
        # This reconstruction is specific to that 4-class encoding.
        # adj_fwd = (adj_matrix_raw == 1) | (adj_matrix_raw == 3)
        # adj_bwd = (adj_matrix_raw == 2) | (adj_matrix_raw == 3)
        # adj_matrix_final = adj_fwd.astype(int) - adj_bwd.astype(int) # Or some other interpretation
        # Original generate.py: adj = (adj % 2) + (adj // 2).T where adj was from m_seq_to_adj_mat
        # This implies the m_seq_to_adj_mat output was already suitable for this.
        # Let's assume adj_matrix_raw is N x N with these 0-3 classes.
        # The m_seq_to_adj_mat creates a matrix where adj[i,j] is the class of edge (i,j)
        # The formula (adj % 2) + (adj // 2).T is for the specific BFS ordering and S_i definition.
        # For a general matrix of edge classes, this might not be right.
        # Sticking to the original paper's final processing for undirected if not specified.
        # For now, let's assume for directed modes, adj_matrix_raw is the directed one.
        # The tril and symmetrization is for UNDIRECTED.
        adj_matrix_final = adj_matrix_raw # For directed modes, m_seq_to_adj_mat output is taken as is for now.
                                        # Specific reconstruction for 'directed-multiclass' or 'aig-custom-topsort'
                                        # would be needed here if adj_matrix_raw holds special class indices.
    elif mode == 'aig-custom-topsort':
        # adj_matrix_raw has edge classes (e.g. 0 for REG, 1 for INV, assuming a "no edge" class was handled
        # or that non-zero entries imply edges).
        # For AIGs, this matrix is already directed by the topsort generation.
        # We might want a binary matrix indicating existence, and attributes separately.
        # For now, return the matrix with class indices. User can post-process.
        adj_matrix_final = adj_matrix_raw
    else: # directed-topsort (original) or other directed
        adj_matrix_final = adj_matrix_raw

    # Remove isolated nodes (original logic)
    # This should be done carefully if num_total_nodes was adjusted by EOS.
    # The adj_matrix_final is N x N where N is the current num_total_nodes.
    if adj_matrix_final.size > 0: # Check if matrix is not empty
        if mode == 'undirected': # Only for undirected in original paper's GraphRNN
            # This part is tricky: if we remove nodes, node attributes also need to be filtered.
            # For now, let's keep this node removal part as is, and acknowledge attribute filtering is TODO.
            pass # Stanford code removed this, but GraphRNN paper has it.
                 # For simplicity and to avoid attribute mismatch, let's skip for now.
            # isolated_nodes = np.where(np.sum(adj_matrix_final, axis=0) + np.sum(adj_matrix_final, axis=1) == 0)[0]
            # adj_matrix_final = np.delete(adj_matrix_final, isolated_nodes, axis=0)
            # adj_matrix_final = np.delete(adj_matrix_final, isolated_nodes, axis=1)
            # list_node_attrs_generated would also need filtering:
            # if isolated_nodes.size > 0:
            #    print(f"Warning: Node attribute filtering for isolated nodes not fully implemented with node removal.")
            pass


    final_node_attrs_np = np.array(list_node_attrs_generated)
    # Ensure node attributes match the final number of nodes in adj_matrix_final
    if final_node_attrs_np.shape[0] != adj_matrix_final.shape[0] and adj_matrix_final.shape[0] > 0 :
        # This can happen if EOS adjusted num_total_nodes, and list_node_attrs_generated
        # wasn't trimmed exactly matching how list_adj_vecs_for_matrix was trimmed for m_seq_to_adj_mat.
        # The EOS logic above tries to trim both.
        # print(f"Warning: Mismatch in generated node attributes ({final_node_attrs_np.shape[0]}) and adj matrix nodes ({adj_matrix_final.shape[0]}). Trimming attributes.")
        final_node_attrs_np = final_node_attrs_np[:adj_matrix_final.shape[0]]
    elif adj_matrix_final.shape[0] == 0 and final_node_attrs_np.shape[0] > 0: # Empty adj but some attrs
        final_node_attrs_np = np.array([])


    return adj_matrix_final, final_node_attrs_np


def load_model_from_config(model_path):
    """Get model information from config and return models and model info."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state = torch.load(model_path, map_location=device)
    config = state['config']

    # M-window size
    m_window_size = config['data']['m']

    # Node attribute information (NEW)
    # Ensure this path exists in your config: config['model']['GraphRNN']['num_node_classes']
    num_node_classes = config.get('model', {}).get('GraphRNN', {}).get('num_node_classes', 0)
    if num_node_classes == 0:
        print("Warning: 'num_node_classes' is 0 or not found in config. Node attribute generation might not work as expected.")

    # Edge feature length the model was trained with
    edge_feature_len_model = config.get('model', {}).get('GraphRNN', {}).get('edge_feature_len', 1)


    if config['model']['edge_model'] == 'rnn':
        node_model = GraphLevelRNN(
            input_size=m_window_size, # Adj part of input to GraphLevelRNN
            output_size=config['model']['EdgeRNN']['hidden_size'], # For EdgeRNN init
            num_node_classes=num_node_classes, # NEW
            **config['model']['GraphRNN'] # Other GraphRNN params like embedding, hidden, layers, edge_feature_len
        ).to(device)
        edge_model = EdgeLevelRNN(
            **config['model']['EdgeRNN'] # EdgeRNN params like embedding, hidden, layers, edge_feature_len
        ).to(device)
        edge_gen_function = rnn_edge_gen
    else: # MLP
        node_model = GraphLevelRNN(
            input_size=m_window_size,
            output_size=None,  # MLP takes hidden state directly
            num_node_classes=num_node_classes, # NEW
            **config['model']['GraphRNN']
        ).to(device)
        # Ensure EdgeLevelMLP init matches its definition
        edge_model = EdgeLevelMLP(
            input_size_from_graph_rnn=config['model']['GraphRNN']['hidden_size'],
            mlp_hidden_size=config['model']['EdgeMLP']['hidden_size'],
            num_edges_to_predict=m_window_size, # MLP predicts M edge slots
            edge_feature_len=edge_feature_len_model # It should predict features of this length
        ).to(device)
        edge_gen_function = mlp_edge_gen

    node_model.load_state_dict(state['node_model'])
    edge_model.load_state_dict(state['edge_model'])

    mode = config.get('model', {}).get('mode', 'undirected')

    return node_model, edge_model, m_window_size, num_node_classes, edge_feature_len_model, edge_gen_function, mode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path of the model weights')
    parser.add_argument('-n', '--nodes', dest='num_output_nodes', required=False, default=10, type=int,
                        help='Number of nodes desired in the output graph')
    args = parser.parse_args()

    (node_model_loaded, edge_model_loaded, m_param,
     num_node_classes_loaded, edge_feat_len_loaded,
     edge_gen_func_loaded, mode_loaded) = load_model_from_config(args.model_path)

    print(f"Generating graph with mode: {mode_loaded}, target nodes: {args.num_output_nodes}")
    print(f"Model params: M={m_param}, NodeClasses={num_node_classes_loaded}, EdgeFeatLen={edge_feat_len_loaded}")

    adj_matrix_generated, node_attributes_generated = generate(
        args.num_output_nodes,
        node_model_loaded,
        edge_model_loaded,
        m_param,
        num_node_classes_loaded,
        edge_feat_len_loaded,
        edge_gen_func_loaded,
        mode_loaded
    )

    print(f"Generated Adjacency Matrix (shape: {adj_matrix_generated.shape}):")
    # print(adj_matrix_generated)
    print(f"Generated Node Attributes (shape: {node_attributes_generated.shape}):")
    # print(node_attributes_generated)

    # The evaluate.draw_generated_graph function might need to be updated
    # if you want to visualize node attributes (e.g., by coloring nodes).
    # For now, it just takes the adjacency matrix and a directed flag.
    is_directed_graph = (mode_loaded != 'undirected')

    # # Ensure graph is not empty before drawing
    # if adj_matrix_generated.shape[0] > 0:
    #     try:
    #         evaluate.draw_generated_graph(adj_matrix_generated, 'generated_graph_with_node_types', directed=is_directed_graph)
    #         print("Saved generated graph to generated_graph_with_node_types.png")
    #     except Exception as e:
    #         print(f"Error during graph drawing: {e}")
    #         print("Skipping graph drawing.")
    # else:
    #     print("Generated graph is empty, skipping drawing.")
    #
