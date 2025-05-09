### Code adapted from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow
import re
import numpy as np
import networkx as nx
from aig_config import *  # Imports NODE_TYPE_KEYS, NUM_NODE_FEATURES, VIRTUAL_EDGE_INDEX, NUM2EDGETYPE, check_validity, Fan_ins
import warnings  # Ensure warnings is imported
from collections import Counter  # <<<< ----- ADDED THIS IMPORT

# For detailed debugging of array contents
np.set_printoptions(threshold=50, edgeitems=5)  # Controls numpy array printing

# TODO figureout equivalent (This ATOM_VALENCY seems specific to chemistry, not AIGs)
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

# This bond_decoder_m seems specific to chemistry or a different edge type definition.
# For AIGs, we are using NUM2EDGETYPE from aig_config.py
# bond_decoder_m = {1: "EDGE_REG", 2: "EDGE_INV", 3: "NONE"}


# Local NODE_TYPE_ENCODING_PYG - for reference, but construct_aig uses NODE_TYPE_KEYS from aig_config
NODE_TYPE_ENCODING_PYG_LOCAL = {
    "NODE_CONST0": 0,
    "NODE_PI": 1,
    "NODE_AND": 2,
    "NODE_PO": 3,
    "VIRTUAL": 4  # This indicates the 5th channel in EBM output
}
print(f"DEBUG: generate.py: Loaded NODE_TYPE_KEYS from aig_config: {NODE_TYPE_KEYS}")
print(f"DEBUG: generate.py: Loaded NUM_NODE_FEATURES from aig_config: {NUM_NODE_FEATURES}")
print(f"DEBUG: generate.py: Loaded VIRTUAL_EDGE_INDEX from aig_config (for 'NONE' edge): {VIRTUAL_EDGE_INDEX}")
print(f"DEBUG: generate.py: Loaded NUM2EDGETYPE (for actual edges) from aig_config: {NUM2EDGETYPE}")
print(f"DEBUG: generate.py: Loaded Fan_ins from aig_config: {Fan_ins}")


def print_graph_summary(g, stage_name="Graph", sample_idx=None):
    """
    Prints a summary of the graph's properties, including node/edge counts, types,
    DAG status, and validity. Optimized cycle detection.
    """
    prefix = f"DEBUG: Sample {sample_idx}: " if sample_idx is not None else "DEBUG: "
    if g is None:
        print(f"{prefix}Graph at stage '{stage_name}' is None.")
        return

    print(f"{prefix}Graph summary at stage '{stage_name}':")
    print(f"{prefix}  Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")

    # Count and print node types
    node_types = Counter(d.get('type', 'UNTYPED') for _, d in g.nodes(data=True))
    print(f"{prefix}  Node Types: {dict(node_types)}")

    # Count and print edge types
    edge_types = Counter(d.get('type', 'UNTYPED') for _, _, d in g.edges(data=True))
    print(f"{prefix}  Edge Types: {dict(edge_types)}")

    # Check and print DAG status and cycles (if not a DAG and graph is small enough for quick check)
    if g.number_of_nodes() > 0:
        is_dag_val = nx.is_directed_acyclic_graph(g)
        print(f"{prefix}  Is DAG? {is_dag_val}")

        if not is_dag_val:  # If it's not a DAG, then find and print a few cycles
            print(f"{prefix}    Attempting to find a few cycles for non-DAG graph...")
            found_cycles_list = []
            try:
                # Iterate over the generator to get a few cycles
                # This is much more efficient than list(nx.simple_cycles(g))
                for cycle_count, cycle in enumerate(nx.simple_cycles(g)):
                    found_cycles_list.append(cycle)
                    if cycle_count >= 2:  # Stop after finding 3 cycles (0, 1, 2)
                        break

                if found_cycles_list:
                    print(f"{prefix}    Cycles found (first {len(found_cycles_list)}): {found_cycles_list}")
                else:
                    print(
                        f"{prefix}    No simple cycles were reported by nx.simple_cycles, despite not being a DAG (or graph too complex for quick check).")

            except Exception as e_cycle:
                # Catching specific NetworkX errors can be more informative if needed
                if isinstance(e_cycle, (nx.NetworkXUnfeasible, nx.NetworkXError, nx.NetworkXNotImplemented)):
                    print(
                        f"{prefix}    NetworkX error during cycle detection (possibly due to graph structure): {e_cycle}")
                else:
                    print(f"{prefix}    Error finding cycles: {e_cycle} (Type: {type(e_cycle)})")

        # Perform validity check using the function from aig_config
        validity_check_result = check_validity(g)
        print(f"{prefix}  Validity (check_validity result): {validity_check_result}")
    else:
        # Handle empty graph case
        print(f"{prefix}  Is DAG? N/A (empty graph)")
        print(
            f"{prefix}  Validity (check_validity result): True (empty graph typically considered valid by default in check_validity)")


def construct_aig(x_features_matrix, adj_channel_tensor, node_type_list=NODE_TYPE_KEYS,
                  edge_score_threshold=0.75, sample_idx_debug=None):  # Added edge_score_threshold
    """
    Constructs an AIG (And-Inverter Graph) from raw feature matrices.
    Args:
        x_features_matrix (np.ndarray): Node features, shape (max_nodes, num_node_channels).
        adj_channel_tensor (np.ndarray): Adjacency features, shape (num_edge_channels, max_nodes, max_nodes).
        node_type_list (list): List of node type strings for decoding.
        edge_score_threshold (float): Minimum score for an edge to be considered.
        sample_idx_debug (int, optional): Index for debugging prints.
    Returns:
        nx.DiGraph: The constructed AIG.
    """
    prefix = f"DEBUG: Sample {sample_idx_debug}: construct_aig: " if sample_idx_debug is not None else "DEBUG: construct_aig: "
    print(f"\n{prefix}Called.")
    print(f"{prefix}  x_features_matrix shape: {x_features_matrix.shape}, dtype: {x_features_matrix.dtype}")
    print(f"{prefix}  adj_channel_tensor shape: {adj_channel_tensor.shape}, dtype: {adj_channel_tensor.dtype}")
    print(f"{prefix}  node_type_list used for decoding: {node_type_list} (len: {len(node_type_list)})")
    print(f"{prefix}  Using edge_score_threshold: {edge_score_threshold}")  # Log the threshold

    # Print a small part of the input tensors to see their nature
    if x_features_matrix.size > 0:
        print(f"{prefix}  x_features_matrix (first 3 rows, all columns):\n{x_features_matrix[:3, :]}")
    if adj_channel_tensor.size > 0:
        print(f"{prefix}  adj_channel_tensor (channel 0, first 3x3 block norms):\n{adj_channel_tensor[0, :3, :3]}")
        if adj_channel_tensor.shape[0] > 1:
            print(f"{prefix}  adj_channel_tensor (channel 1, first 3x3 block norms):\n{adj_channel_tensor[1, :3, :3]}")

    aig = nx.DiGraph()
    # Determine node types by taking argmax over the feature channels for each node
    node_type_indices_all = np.argmax(x_features_matrix, axis=1)
    print(
        f"{prefix}  node_type_indices_all (argmax over last dim of x_features_matrix, first 10 values): {node_type_indices_all[:10]}")

    # Index corresponding to the "virtual node" type in the EBM's output node feature matrix
    # This is NUM_NODE_FEATURES because it's the channel *after* all actual node features.
    virtual_node_channel_matrix_idx = NUM_NODE_FEATURES
    print(
        f"{prefix}  virtual_node_channel_matrix_idx (from aig_config.NUM_NODE_FEATURES): {virtual_node_channel_matrix_idx}")

    # Create a mask for active nodes (nodes that are not virtual padding nodes)
    active_node_mask = node_type_indices_all != virtual_node_channel_matrix_idx
    print(f"{prefix}  active_node_mask (first 10 values): {active_node_mask[:10]}")
    num_active_nodes_calc = active_node_mask.sum()
    print(f"{prefix}  Number of active nodes identified by mask: {num_active_nodes_calc}")

    # Get the assigned type indices only for the active nodes
    active_node_assigned_type_indices = node_type_indices_all[active_node_mask]
    print(
        f"{prefix}  active_node_assigned_type_indices (indices for active nodes, first 10 if any): {active_node_assigned_type_indices[:10] if active_node_assigned_type_indices.size > 0 else 'None'}")

    # Add active nodes to the graph with their types
    nodes_added_count = 0
    for compact_idx, assigned_type_idx in enumerate(active_node_assigned_type_indices):
        if assigned_type_idx >= len(node_type_list):  # Should not happen if virtual_node_channel_matrix_idx is correct
            warnings.warn(
                f"{prefix}Internal Error: assigned_type_idx ({assigned_type_idx}) is out of bounds for "
                f"node_type_list (len {len(node_type_list)}) even after filtering. "
            )
            print(
                f"{prefix}SKIPPING node add for compact_idx {compact_idx} due to out-of-bounds assigned_type_idx {assigned_type_idx}")
            continue

        actual_node_type_str = node_type_list[assigned_type_idx]
        aig.add_node(compact_idx, type=actual_node_type_str)  # Use compact_idx (0 to num_active_nodes-1)
        nodes_added_count += 1
        if nodes_added_count <= 5: print(
            f"{prefix}  Added node {compact_idx} with type '{actual_node_type_str}' (original type index from argmax: {assigned_type_idx})")
    if nodes_added_count > 5: print(f"{prefix}  ... and {nodes_added_count - 5} more nodes.")

    # Determine edge types and scores
    # adj_matrix_edge_indices: for each (src, tgt) pair, the channel index with the max score (i.e., predicted edge type)
    adj_matrix_edge_indices = np.argmax(adj_channel_tensor, axis=0)  # Shape (max_nodes, max_nodes)
    # adj_matrix_edge_scores: for each (src, tgt) pair, the max score itself across channels
    adj_matrix_edge_scores = np.max(adj_channel_tensor, axis=0)  # Shape (max_nodes, max_nodes)

    print(
        f"{prefix}  adj_matrix_edge_indices (argmax over channel dim of adj_tensor, first 5x5 block):\n{adj_matrix_edge_indices[:5, :5]}")
    print(f"{prefix}  VIRTUAL_EDGE_INDEX (for 'NONE' edge type) from aig_config: {VIRTUAL_EDGE_INDEX}")

    edges_added_count = 0
    edges_skipped_due_to_threshold_count = 0  # Counter for skipped edges

    if num_active_nodes_calc > 0:  # Proceed only if there are active nodes
        # Get original indices (in the full 0-63 range) of nodes that are active
        original_indices_of_active_nodes = np.where(active_node_mask)[0]
        # Map these original indices to compact indices (0 to num_active_nodes-1)
        map_original_to_compact = {orig_idx: compact_idx for compact_idx, orig_idx in
                                   enumerate(original_indices_of_active_nodes)}
        print(
            f"{prefix}  original_indices_of_active_nodes (indices in x_features_matrix that are active, first 10): {original_indices_of_active_nodes[:10]}")

        # Iterate over pairs of *original indices* of active nodes to check for edges
        for start_node_orig_idx in original_indices_of_active_nodes:
            for end_node_orig_idx in original_indices_of_active_nodes:
                if start_node_orig_idx == end_node_orig_idx:  # No self-loops
                    continue

                # Get the predicted edge type index from the full adjacency matrix using original indices
                edge_type_idx_in_adj = adj_matrix_edge_indices[start_node_orig_idx, end_node_orig_idx]

                # Check if the predicted edge type is NOT the virtual/none edge
                if edge_type_idx_in_adj != VIRTUAL_EDGE_INDEX:
                    # Get the score for this predicted edge
                    edge_score = adj_matrix_edge_scores[start_node_orig_idx, end_node_orig_idx]

                    # --- NEW THRESHOLDING STEP ---
                    if edge_score >= edge_score_threshold:
                        # Convert original indices to compact indices for adding to the graph
                        start_node_compact_idx = map_original_to_compact[start_node_orig_idx]
                        end_node_compact_idx = map_original_to_compact[end_node_orig_idx]

                        # Check if the determined edge type index is valid (e.g., 0 for REG, 1 for INV)
                        if edge_type_idx_in_adj in NUM2EDGETYPE:
                            actual_edge_type_str = NUM2EDGETYPE[edge_type_idx_in_adj]
                            aig.add_edge(
                                int(start_node_compact_idx),  # Source
                                int(end_node_compact_idx),  # Target
                                type=actual_edge_type_str,
                                score=float(edge_score)  # Store score for potential use in correct_fanins
                            )
                            edges_added_count += 1
                            if edges_added_count <= 5:
                                print(
                                    f"{prefix}  Added edge ({start_node_compact_idx}-{end_node_compact_idx}) type '{actual_edge_type_str}', score {edge_score:.2f} (orig_indices: {start_node_orig_idx}-{end_node_orig_idx}, edge_type_idx_from_adj: {edge_type_idx_in_adj})")
                        else:
                            # This case should be rare if VIRTUAL_EDGE_INDEX is handled correctly and NUM2EDGETYPE is comprehensive
                            warnings.warn(
                                f"{prefix}Unexpected edge_type_idx_in_adj: {edge_type_idx_in_adj} for an active edge between original nodes {start_node_orig_idx}-{end_node_orig_idx}. NUM2EDGETYPE: {NUM2EDGETYPE}")
                            print(
                                f"{prefix}SKIPPING edge add for orig_indices {start_node_orig_idx}-{end_node_orig_idx} due to unexpected edge_type_idx_in_adj {edge_type_idx_in_adj}")
                    else:  # Edge score is below threshold
                        edges_skipped_due_to_threshold_count += 1
                        # Optional: print if you want to see skipped edges, can be very verbose
                        # if edges_skipped_due_to_threshold_count <= 5:
                        #     print(f"{prefix}  SKIPPED edge (score {edge_score:.2f} < {edge_score_threshold}) between orig_indices {start_node_orig_idx}-{end_node_orig_idx}")

        if edges_added_count > 5: print(f"{prefix}  ... and {edges_added_count - 5} more edges.")
        print(f"{prefix}  Total edges added in construct_aig: {edges_added_count}")
        print(
            f"{prefix}  Total edges skipped due to threshold ({edge_score_threshold}): {edges_skipped_due_to_threshold_count}")
    else:
        print(f"{prefix}No active nodes, so no edges were considered or added in construct_aig.")

    print_graph_summary(aig, "construct_aig_end", sample_idx=sample_idx_debug)
    return aig


# Fan_ins is already imported from aig_config via *

def correct_fanins(aig, sample_idx_debug=None):
    """
    Corrects fan-ins for nodes in the AIG based on predefined rules.
    Also corrects PO node out-degrees to be 0.
    Args:
        aig (nx.DiGraph): The input AIG.
        sample_idx_debug (int, optional): Index for debugging prints.
    Returns:
        Tuple[nx.DiGraph or None, int]: The corrected AIG (or None if it becomes invalid) and a purity flag.
    """
    prefix = f"DEBUG: Sample {sample_idx_debug}: correct_fanins: " if sample_idx_debug is not None else "DEBUG: correct_fanins: "
    print(f"\n{prefix}Called.")
    if aig is None:
        print(f"{prefix}Input AIG is None. Returning None.")
        return None, 0  # 0 indicates not pure/valid from start or corrections failed

    print_graph_summary(aig, "correct_fanins_start", sample_idx=sample_idx_debug)
    is_initially_valid = check_validity(aig)
    print(f"{prefix}  Initial validity (check_validity before corrections): {is_initially_valid}")

    if is_initially_valid:
        print(f"{prefix}  AIG already valid before fan-in correction. Returning as is.")
        return aig, 1  # 1 indicates it was pure/valid

    # Work on a copy to avoid modifying the original graph passed to this function
    # if it's used elsewhere before this function returns.
    # However, the common pattern here is to modify and return.
    # For clarity of changes, let's assume modification in place is fine for the `aig` var.
    # corrected_graph = aig.copy() # If you need to preserve `aig` in its pre-correction state for the caller.
    # For now, we'll modify `aig` directly.

    print(f"{prefix}  Attempting fan-in corrections...")
    # Iterate over a list of nodes because we might modify the graph (remove edges)
    for node_id, node_data in list(aig.nodes(data=True)):
        node_type = node_data.get('type')
        if node_type is None:
            print(f"{prefix}  Skipping node {node_id} in fan-in correction: no 'type' attribute.")
            continue
        if node_type not in Fan_ins:  # Fan_ins is from aig_config
            print(
                f"{prefix}  Skipping node {node_id} (type: {node_type}) in fan-in correction: type not in Fan_ins dict ({Fan_ins}).")
            continue

        current_in_degree = aig.in_degree(node_id)
        target_fan_in = Fan_ins[node_type]

        # Correcting In-degrees
        if current_in_degree > target_fan_in:
            print(
                f"{prefix}    Correcting fan-in for node {node_id} (type: {node_type}). In-degree {current_in_degree} > Target {target_fan_in}")
            # Get all incoming edges with their scores
            incoming_edges_with_scores = []
            for u, v, edge_attributes in aig.in_edges(node_id, data=True):  # u is source, v is target (node_id)
                score = edge_attributes.get('score', float('-inf'))  # Default to very low score if not present
                incoming_edges_with_scores.append(((u, v), score))

            # Sort edges by score in descending order (keep highest scores)
            incoming_edges_with_scores.sort(key=lambda x: x[1], reverse=True)

            if len(incoming_edges_with_scores) > 3:
                print(
                    f"{prefix}      Sorted incoming edges (first 3 of {len(incoming_edges_with_scores)}): {incoming_edges_with_scores[:3]}...")
            else:
                print(f"{prefix}      Sorted incoming edges: {incoming_edges_with_scores}")

            # Edges to remove are those beyond the target_fan_in count
            edges_to_remove = incoming_edges_with_scores[target_fan_in:]
            print(
                f"{prefix}      Edges to remove (to meet target fan-in {target_fan_in}): {len(edges_to_remove)} edges.")
            for i_rem, ((u_rem, v_rem), score_rem) in enumerate(edges_to_remove):
                if aig.has_edge(u_rem, v_rem):
                    aig.remove_edge(u_rem, v_rem)
                    if i_rem < 3: print(f"{prefix}      Removed edge ({u_rem},{v_rem}) with score {score_rem:.2f}")
            if len(edges_to_remove) > 3: print(f"{prefix}      ... and {len(edges_to_remove) - 3} more edges removed.")

        # Correcting Out-degrees for PO nodes (should be 0)
        if node_type == NODE_TYPE_KEYS[3]:  # NODE_PO (ensure NODE_TYPE_KEYS[3] is 'NODE_PO')
            current_out_degree = aig.out_degree(node_id)
            if current_out_degree > 0:
                print(
                    f"{prefix}    Correcting out-degree for PO node {node_id}. Out-degree is {current_out_degree}, should be 0.")
                # Get list of successors to remove edges to.
                # Iterate over a copy as we are modifying the graph.
                successors_to_remove = list(aig.successors(node_id))
                if len(successors_to_remove) > 3:
                    print(
                        f"{prefix}      Successors to remove edges to (first 3 of {len(successors_to_remove)}): {successors_to_remove[:3]}...")
                else:
                    print(f"{prefix}      Successors to remove edges to: {successors_to_remove}")

                removed_count = 0
                for target_node_succ in successors_to_remove:
                    if aig.has_edge(node_id, target_node_succ):
                        aig.remove_edge(node_id, target_node_succ)
                        removed_count += 1
                        if removed_count <= 3: print(
                            f"{prefix}      Removed outgoing edge ({node_id} -> {target_node_succ}) from PO node.")
                if removed_count > 3: print(
                    f"{prefix}      ... and {removed_count - 3} more outgoing edges removed from PO node.")

    print_graph_summary(aig, "correct_fanins_intermediate_after_loops", sample_idx=sample_idx_debug)
    is_valid_after_correction = check_validity(aig)
    print(f"{prefix}  Validity after fan-in corrections (check_validity result): {is_valid_after_correction}")

    if is_valid_after_correction:
        print(f"{prefix}  AIG valid after fan-in corrections.")
        return aig, 0  # 0 because corrections were made (not initially pure)
    else:
        print(f"{prefix}Warning: issues everywhere still wtf")  # Your original warning
        print_graph_summary(aig, "correct_fanins_fail_validity_check", sample_idx=sample_idx_debug)
        print(f"{prefix}  AIG still invalid after fan-in corrections. Returning None.")
        return None, 0  # 0 because corrections failed to make it valid


def valid_aig_can_with_seg(aig: nx.DiGraph, sample_idx_debug=None):
    """
    Performs canonicalization (relabeling based on topological sort) and
    removes isolated nodes from a valid AIG.
    Args:
        aig (nx.DiGraph): The input AIG, assumed to be valid and DAG.
        sample_idx_debug (int, optional): Index for debugging prints.
    Returns:
        nx.DiGraph or None: The canonicalized AIG, or None if input is None or not DAG.
    """
    prefix = f"DEBUG: Sample {sample_idx_debug}: valid_aig_can_with_seg: " if sample_idx_debug is not None else "DEBUG: valid_aig_can_with_seg: "
    print(f"\n{prefix}Called.")
    if aig is None:
        print(f"{prefix}Input AIG is None. Returning None.")
        return None
    print_graph_summary(aig, "valid_aig_can_with_seg_start", sample_idx=sample_idx_debug)

    if not isinstance(aig, nx.DiGraph):
        print(f"{prefix}Input is not a NetworkX DiGraph (type: {type(aig)}). Returning as is.")
        return aig  # Or None, depending on desired strictness

    if aig.number_of_nodes() == 0:  # Handle empty graph explicitly
        print(f"{prefix}Input graph is empty. Returning as is.")
        return aig

    is_dag = nx.is_directed_acyclic_graph(aig)
    print(f"{prefix}  Is DAG? {is_dag}")
    if not is_dag:
        print(f"{prefix}  Graph is not a DAG. Returning as is (no canonicalization or isolation removal).")
        # Depending on strictness, you might want to return None here
        # if subsequent steps absolutely require a DAG.
        return aig

    print(f"{prefix}  Graph is a DAG. Relabeling nodes based on topological sort.")
    try:
        topo_sorted_nodes = list(nx.topological_sort(aig))
        # Create mapping from old labels (original node IDs in `aig`) to new sequential labels (0, 1, 2...)
        mapping = {old_label: new_label for new_label, old_label in enumerate(topo_sorted_nodes)}
        # print(f"{prefix}    Topological sort mapping (first 5 if many): {list(mapping.items())[:5]}") # Can be long
        aig = nx.relabel_nodes(aig, mapping, copy=True)  # Use copy=True to ensure a new graph object
        print_graph_summary(aig, "valid_aig_can_with_seg_after_relabel", sample_idx=sample_idx_debug)
    except Exception as e:
        print(f"{prefix}  Error during topological sort or relabeling: {e}. Returning graph before this attempt.")
        return aig  # Return graph state before this failed block

    # Remove isolated nodes (nodes with in-degree 0 and out-degree 0)
    # This is often done for "segmentation" or cleaning up disconnected parts.
    if aig.number_of_nodes() > 1:  # Isolate removal makes sense for graphs with more than one node
        isolates = list(nx.isolates(aig))
        if isolates:
            print(
                f"{prefix}  Removing {len(isolates)} isolated nodes: {isolates[:5] if len(isolates) > 5 else isolates}...")
            aig.remove_nodes_from(isolates)
            # Summary will be printed at the end
        else:
            print(f"{prefix}  No isolated nodes found to remove.")
    else:
        print(f"{prefix}  Graph has <= 1 node, skipping isolate removal.")

    print_graph_summary(aig, "valid_aig_can_with_seg_end", sample_idx=sample_idx_debug)
    return aig


def gen_mol_from_one_shot_tensor(adj, x, largest_connected_comp=True,
                                 edge_score_thresh_for_construct=0.5):  # largest_connected_comp seems unused
    """
    Generates AIGs from batched tensor representations.
    Args:
        adj (torch.Tensor): Adjacency tensor from EBM, shape (B, E_channels, N, N).
        x (torch.Tensor): Node feature tensor from EBM, shape (B, N_channels, N).
        largest_connected_comp (bool): (Currently unused) Flag for post-processing.
        edge_score_thresh_for_construct (float): Threshold passed to construct_aig.
    Returns:
        Tuple[List[nx.DiGraph or None], int]: List of generated graphs and count of initially pure/valid graphs.
    """
    prefix = "DEBUG: gen_mol_from_one_shot_tensor: "
    print(f"\n{prefix}Called.")
    print(f"{prefix}  Input x shape: {x.shape}, Input adj shape: {adj.shape}")
    print(f"{prefix}  Using edge_score_threshold for construct_aig: {edge_score_thresh_for_construct}")

    # Permute x from (B, N_channels, N) to (B, N, N_channels) to match expected input for zipping
    x = x.permute(0, 2, 1)
    print(f"{prefix}  x shape after permute (B, N, NodeChannels): {x.shape}")

    # Move to CPU and convert to NumPy arrays
    adj = adj.cpu().detach().numpy()
    x = x.cpu().detach().numpy()

    gen_graphs = []  # List to store generated NetworkX graphs (or None if generation fails for a sample)
    uncorrected_graphs_count = 0  # Counter for graphs that were valid before/without needing fan-in correction
    num_samples_to_process = x.shape[0]
    print(f"{prefix}  Processing {num_samples_to_process} samples...")

    for i_sample, (x_elem, adj_elem) in enumerate(zip(x, adj)):  # Iterate through each sample in the batch
        print(f"\n--- {prefix}Processing sample {i_sample + 1}/{num_samples_to_process} ---")
        print(
            f"{prefix}  x_elem shape (N, NodeChannels): {x_elem.shape}, adj_elem shape (EdgeChannels, N, N): {adj_elem.shape}")

        # Step 1: Construct initial AIG from raw EBM output
        # Pass the edge_score_thresh_for_construct to construct_aig
        aig_constructed = construct_aig(x_elem, adj_elem,
                                        node_type_list=NODE_TYPE_KEYS,
                                        edge_score_threshold=edge_score_thresh_for_construct,
                                        sample_idx_debug=i_sample + 1)

        # Step 2: Correct fan-ins and PO out-degrees
        aig_corrected, pure_flag = correct_fanins(aig_constructed, sample_idx_debug=i_sample + 1)

        uncorrected_graphs_count += pure_flag  # pure_flag is 1 if valid before/no changes, 0 otherwise
        print(
            f"{prefix}  Sample {i_sample + 1}: pure_flag from correct_fanins: {pure_flag} (1 means was valid before/no changes, 0 means corrections attempted or failed)")

        # Step 3: Canonicalize (relabel, remove isolates) if graph is valid and a DAG
        # valid_aig_can_with_seg will also check for DAG internally.
        aig_canonicalized = valid_aig_can_with_seg(aig_corrected, sample_idx_debug=i_sample + 1)

        if aig_canonicalized is None:
            print(f"{prefix}  Sample {i_sample + 1}: Resulting AIG is None after all processing steps.")
        else:
            print(
                f"{prefix}  Sample {i_sample + 1}: Final AIG for this sample has {aig_canonicalized.number_of_nodes()} nodes, {aig_canonicalized.number_of_edges()} edges.")

        gen_graphs.append(aig_canonicalized)  # Add the final graph (or None) to the list

        # Limit full debug output for brevity if processing many samples
        if i_sample >= 1 and num_samples_to_process > 2:
            print(f"\n{prefix}--- Further sample processing will be less verbose (summary prints only) ---")
            # Note: True reduction of verbosity would require passing a flag to sub-functions.
            # For now, we just break after 2 full samples for this debugging session.
            # Remove 'break' for full processing in production.
            # break
            pass  # Keep processing all samples

    print(f"\n{prefix}Finished processing. Generated {len(gen_graphs)} graph objects (some might be None).")
    print(
        f"{prefix}  Count of graphs initially 'pure' and valid before/during correct_fanins: {uncorrected_graphs_count}")
    return gen_graphs, uncorrected_graphs_count
