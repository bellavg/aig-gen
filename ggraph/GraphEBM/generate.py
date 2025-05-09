### Code adapted from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow
import re
import numpy as np
import networkx as nx
from aig_config import *  # Imports NODE_TYPE_KEYS, NUM_NODE_FEATURES, VIRTUAL_EDGE_INDEX, NUM2EDGETYPE, check_validity
import warnings  # Ensure warnings is imported

# For detailed debugging of array contents
np.set_printoptions(threshold=50, edgeitems=5)

# TODO figureout equivalent
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

# bond_decoder_m = {1: "EDGE_REG", 2: "EDGE_INV", 3: "NONE"} # This seems unused if NUM2EDGETYPE is used

# Local NODE_TYPE_ENCODING_PYG - for reference or if used locally, but construct_aig uses NODE_TYPE_KEYS from aig_config
NODE_TYPE_ENCODING_PYG_LOCAL = {
    "NODE_CONST0": 0,
    "NODE_PI": 1,
    "NODE_AND": 2,
    "NODE_PO": 3,
    "VIRTUAL": 4
}
print(f"DEBUG: generate.py: NODE_TYPE_KEYS from aig_config: {NODE_TYPE_KEYS}")
print(f"DEBUG: generate.py: NUM_NODE_FEATURES from aig_config: {NUM_NODE_FEATURES}")
print(f"DEBUG: generate.py: VIRTUAL_EDGE_INDEX from aig_config: {VIRTUAL_EDGE_INDEX}")
print(f"DEBUG: generate.py: NUM2EDGETYPE from aig_config: {NUM2EDGETYPE}")


def print_graph_summary(g, stage_name):
    if g is None:
        print(f"DEBUG: Graph at stage '{stage_name}' is None.")
        return
    print(f"DEBUG: Graph summary at stage '{stage_name}':")
    print(f"  Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
    if g.number_of_nodes() > 0:
        print(f"  Node data (first few):")
        for i, (node_id, data) in enumerate(g.nodes(data=True)):
            if i >= 3 and g.number_of_nodes() > 5:  # Print more if graph is small
                print("    ...")
                break
            print(f"    Node {node_id}: {data}")
    if g.number_of_edges() > 0:
        print(f"  Edge data (first few):")
        for i, (u, v, data) in enumerate(g.edges(data=True)):
            if i >= 3 and g.number_of_edges() > 5:
                print("    ...")
                break
            print(f"    Edge ({u}-{v}): {data}")
    print(f"  Is DAG? {nx.is_directed_acyclic_graph(g) if g.number_of_nodes() > 0 else 'N/A (empty)'}")
    print(f"  Validity (check_validity): {check_validity(g) if g.number_of_nodes() > 0 else 'N/A (empty)'}")


def construct_aig(x_features_matrix, adj_channel_tensor, node_type_list=NODE_TYPE_KEYS):
    print(f"\nDEBUG: construct_aig called.")
    print(f"  x_features_matrix shape: {x_features_matrix.shape}")
    print(f"  adj_channel_tensor shape: {adj_channel_tensor.shape}")
    print(f"  node_type_list: {node_type_list}")
    # print(f"  x_features_matrix (first 5 rows norms): {[np.linalg.norm(row) for row in x_features_matrix[:5]]}")
    # print(f"  x_features_matrix (first 5 rows argmax): {np.argmax(x_features_matrix[:5], axis=1)}")
    # print(f"  adj_channel_tensor (norm per channel): {[np.linalg.norm(channel) for channel in adj_channel_tensor]}")
    # print(f"  adj_channel_tensor (max val per channel): {[np.max(channel) for channel in adj_channel_tensor]}")

    aig = nx.DiGraph()
    node_type_indices_all = np.argmax(x_features_matrix, axis=1)
    print(f"  node_type_indices_all (first 10): {node_type_indices_all[:10]}")

    virtual_node_channel_matrix_idx = NUM_NODE_FEATURES
    print(f"  virtual_node_channel_matrix_idx: {virtual_node_channel_matrix_idx}")

    active_node_mask = node_type_indices_all != virtual_node_channel_matrix_idx
    print(f"  active_node_mask (first 10): {active_node_mask[:10]}")
    print(f"  Number of active nodes identified: {active_node_mask.sum()}")

    active_node_assigned_type_indices = node_type_indices_all[active_node_mask]
    print(
        f"  active_node_assigned_type_indices (first 10 if any): {active_node_assigned_type_indices[:10] if active_node_assigned_type_indices.size > 0 else 'None'}")

    for compact_idx, assigned_type_idx in enumerate(active_node_assigned_type_indices):
        if assigned_type_idx >= len(node_type_list):
            warnings.warn(
                f"Internal Error: construct_aig: assigned_type_idx ({assigned_type_idx}) is out of bounds for "
                f"node_type_list (len {len(node_type_list)}) even after filtering."
            )
            print(
                f"DEBUG: construct_aig: SKIPPING node add for compact_idx {compact_idx} due to out-of-bounds assigned_type_idx {assigned_type_idx}")
            continue

        actual_node_type_str = node_type_list[assigned_type_idx]
        aig.add_node(compact_idx, type=actual_node_type_str)
        if compact_idx < 5: print(
            f"  Added node {compact_idx} with type '{actual_node_type_str}' (original index from argmax: {assigned_type_idx})")

    adj_matrix_edge_indices = np.argmax(adj_channel_tensor, axis=0)  # Shape (max_nodes, max_nodes)
    adj_matrix_edge_scores = np.max(adj_channel_tensor, axis=0)
    # print(f"  adj_matrix_edge_indices (first 5x5 block):\n{adj_matrix_edge_indices[:5,:5]}")

    if active_node_mask.any():
        original_indices_of_active_nodes = np.where(active_node_mask)[0]
        map_original_to_compact = {orig_idx: compact_idx for compact_idx, orig_idx in
                                   enumerate(original_indices_of_active_nodes)}
        print(f"  original_indices_of_active_nodes (first 10): {original_indices_of_active_nodes[:10]}")

        edges_added_count = 0
        for start_node_orig_idx in original_indices_of_active_nodes:
            for end_node_orig_idx in original_indices_of_active_nodes:
                if start_node_orig_idx == end_node_orig_idx:  # No self-loops
                    continue

                edge_type_idx_in_adj = adj_matrix_edge_indices[start_node_orig_idx, end_node_orig_idx]

                if edge_type_idx_in_adj != VIRTUAL_EDGE_INDEX:  # VIRTUAL_EDGE_INDEX is 2 (for "NONE" type)
                    edge_score = adj_matrix_edge_scores[start_node_orig_idx, end_node_orig_idx]
                    start_node_compact_idx = map_original_to_compact[start_node_orig_idx]
                    end_node_compact_idx = map_original_to_compact[end_node_orig_idx]

                    if edge_type_idx_in_adj in NUM2EDGETYPE:  # NUM2EDGETYPE is {0: "EDGE_REG", 1: "EDGE_INV"}
                        actual_edge_type_str = NUM2EDGETYPE[edge_type_idx_in_adj]
                        aig.add_edge(
                            int(start_node_compact_idx),
                            int(end_node_compact_idx),
                            type=actual_edge_type_str,
                            score=float(edge_score)
                        )
                        edges_added_count += 1
                        if edges_added_count <= 5:
                            print(
                                f"  Added edge ({start_node_compact_idx}-{end_node_compact_idx}) type '{actual_edge_type_str}', score {edge_score:.2f} (orig_indices: {start_node_orig_idx}-{end_node_orig_idx}, edge_type_idx_from_adj: {edge_type_idx_in_adj})")
                    else:
                        warnings.warn(
                            f"construct_aig: Unexpected edge_type_idx_in_adj: {edge_type_idx_in_adj} for an active edge between original nodes {start_node_orig_idx}-{end_node_orig_idx}.")
                        print(
                            f"DEBUG: construct_aig: SKIPPING edge add for orig_indices {start_node_orig_idx}-{end_node_orig_idx} due to unexpected edge_type_idx_in_adj {edge_type_idx_in_adj}")

        print(f"  Total edges added in construct_aig: {edges_added_count}")
    else:
        print("  No active nodes, so no edges will be added.")

    print_graph_summary(aig, "construct_aig_end")
    return aig


Fan_ins = {"NODE_PO": 1, "NODE_AND": 2, "NODE_PI": 0, "NODE_CONST0": 0}


def correct_fanins(aig):
    print(f"\nDEBUG: correct_fanins called.")
    if aig is None:
        print("  Input AIG is None. Returning None.")
        return None, 0

    print_graph_summary(aig, "correct_fanins_start")
    is_initially_valid = check_validity(aig)
    print(f"  Initial validity (check_validity before corrections): {is_initially_valid}")
    if is_initially_valid:
        print("  AIG already valid before fan-in correction. Returning as is.")
        return aig, 1  # pure = 1 (no corrections needed, implies it was "born" pure)

    # Create a copy to avoid modifying the graph if only checks are done or if it returns early
    # However, the logic modifies in place and returns the same object or None.
    # For debugging, let's see node types before loop
    for node_id_debug, node_data_debug in list(aig.nodes(data=True)):  # list() for safe iteration if nodes are removed
        node_type_debug = node_data_debug.get('type')
        if node_type_debug is None:
            print(f"  WARNING: Node {node_id_debug} has no 'type' attribute before fan-in correction loop.")
            continue
        if node_type_debug not in Fan_ins:
            print(
                f"  WARNING: Node {node_id_debug} has type '{node_type_debug}' not in Fan_ins dict. Fan-in correction might fail or be incorrect for this node.")

    for node_id, node_data in list(
            aig.nodes(data=True)):  # Use list() if nodes can be removed by other operations (not in this loop)
        node_type = node_data.get('type')
        if node_type is None:
            print(f"  Skipping node {node_id} in fan-in correction: no 'type' attribute.")
            continue
        if node_type not in Fan_ins:
            print(f"  Skipping node {node_id} (type: {node_type}) in fan-in correction: type not in Fan_ins dict.")
            continue

        current_in_degree = aig.in_degree(node_id)
        target_fan_in = Fan_ins[node_type]
        print(
            f"  Node {node_id} (type: {node_type}): Current In-degree: {current_in_degree}, Target Fan-in: {target_fan_in}")

        if current_in_degree > target_fan_in:
            print(f"    Correcting fan-in for node {node_id}. In-degree {current_in_degree} > Target {target_fan_in}")
            incoming_edges_with_scores = []
            for u, v, edge_attributes in aig.in_edges(node_id, data=True):
                score = edge_attributes.get('score', float('-inf'))
                incoming_edges_with_scores.append(((u, v), score))
                print(f"      Incoming edge ({u},{v}) with score: {score}")

            incoming_edges_with_scores.sort(key=lambda x: x[1], reverse=True)  # Keep highest scores
            print(f"      Sorted incoming edges (by score, descending): {incoming_edges_with_scores}")

            edges_to_remove = incoming_edges_with_scores[target_fan_in:]
            print(f"      Edges to remove (to meet target fan-in {target_fan_in}): {edges_to_remove}")
            for (u_rem, v_rem), score_rem in edges_to_remove:
                if aig.has_edge(u_rem, v_rem):
                    aig.remove_edge(u_rem, v_rem)
                    print(f"      Removed edge ({u_rem},{v_rem}) with score {score_rem}")
                else:
                    print(f"      Attempted to remove edge ({u_rem},{v_rem}) but it was already removed.")

        if node_type == "NODE_PO":  # NODE_PO_STR from aig_config.NODE_TYPE_KEYS[3]
            current_out_degree = aig.out_degree(node_id)
            if current_out_degree > 0:
                print(
                    f"    Correcting out-degree for PO node {node_id}. Out-degree is {current_out_degree}, should be 0.")
                successors_to_remove = list(aig.successors(node_id))  # list() for safe iteration
                print(f"      Successors to remove edges to: {successors_to_remove}")
                for target_node_succ in successors_to_remove:
                    if aig.has_edge(node_id, target_node_succ):
                        aig.remove_edge(node_id, target_node_succ)
                        print(f"      Removed outgoing edge ({node_id} -> {target_node_succ}) from PO node.")
                    else:
                        print(
                            f"      Attempted to remove outgoing edge ({node_id} -> {target_node_succ}) but it was already removed.")

    print_graph_summary(aig, "correct_fanins_intermediate_after_loops")
    is_valid_after_correction = check_validity(aig)
    print(f"  Validity after fan-in corrections: {is_valid_after_correction}")

    if is_valid_after_correction:
        print("  AIG valid after fan-in corrections.")
        return aig, 0  # pure = 0 (corrections were made)
    else:
        print("Warning: issues everywhere still wtf")  # This is your original warning
        print_graph_summary(aig, "correct_fanins_fail_validity_check")
        print("  AIG still invalid after fan-in corrections. Returning None.")
        return None, 0


def valid_aig_can_with_seg(aig: nx.DiGraph):
    print(f"\nDEBUG: valid_aig_can_with_seg called.")
    if aig is None:
        print("  Input AIG is None. Returning None.")
        return None
    print_graph_summary(aig, "valid_aig_can_with_seg_start")

    if not isinstance(aig, nx.DiGraph):
        print(f"  Input is not a NetworkX DiGraph (type: {type(aig)}). This function expects DiGraph. Returning as is.")
        return aig  # Or handle error appropriately

    # 1. Validity Check: Must be a Directed Acyclic Graph (DAG)
    is_dag = nx.is_directed_acyclic_graph(aig)
    print(f"  Is DAG? {is_dag}")
    if not is_dag:
        print("  Graph is not a DAG. Returning as is (no canonicalization or isolation removal).")
        return aig  # Return original graph if not DAG, as per original logic

    # Relabel nodes according to topological sort if it's a DAG
    print("  Graph is a DAG. Relabeling nodes based on topological sort.")
    try:
        topo_sorted_nodes = list(nx.topological_sort(aig))
        mapping = {old_label: new_label for new_label, old_label in enumerate(topo_sorted_nodes)}
        print(f"    Topological sort mapping (first 5 if many): {list(mapping.items())[:5]}")
        aig = nx.relabel_nodes(aig, mapping, copy=True)  # copy=True is important
        print_graph_summary(aig, "valid_aig_can_with_seg_after_relabel")
    except Exception as e:
        print(f"  Error during topological sort or relabeling: {e}. Returning graph before relabeling attempt.")
        # Revert to graph before relabeling attempt if an error occurred, or handle as preferred
        # For now, this means the 'aig' before this block is returned if 'is_dag' was true but relabeling failed.
        # This state might be inconsistent with subsequent steps, so careful consideration is needed.
        # Let's return the 'aig' that was passed into this 'else' block if relabeling fails.
        # This part is tricky; ideally, we'd return the graph state just before the failing nx.relabel_nodes.
        # The current 'aig' variable *is* the original if copy was not done or if relabel_nodes failed early.
        # If relabel_nodes was in-place (copy=False), 'aig' is already modified.
        # Since `copy=True` is used, if `relabel_nodes` fails, `aig` refers to the graph before this statement.
        # So, returning `aig` here would be the graph *before* the failed relabeling.
        # If nx.topological_sort fails, `aig` is unchanged.
        return aig  # Return the graph as it was before the error in this block.

    # Remove isolated nodes (except CONST0, though this function doesn't check type for isolation)
    if aig.number_of_nodes() > 1:  # Original check
        isolates = list(nx.isolates(aig))
        if isolates:
            print(f"  Removing isolated nodes: {isolates}")
            aig.remove_nodes_from(isolates)
            print_graph_summary(aig, "valid_aig_can_with_seg_after_isolate_removal")
        else:
            print("  No isolated nodes found to remove.")
    else:
        print("  Graph has <= 1 node, skipping isolate removal.")

    print_graph_summary(aig, "valid_aig_can_with_seg_end")
    return aig


def gen_mol_from_one_shot_tensor(adj, x, largest_connected_comp=True):  # largest_connected_comp seems unused
    print(f"\nDEBUG: gen_mol_from_one_shot_tensor called.")
    print(f"  Input x shape: {x.shape}, Input adj shape: {adj.shape}")

    x = x.permute(0, 2, 1)  # Now B, N, (Atom Types + 1)
    print(f"  x shape after permute: {x.shape}")
    adj = adj.cpu().detach().numpy()
    x = x.cpu().detach().numpy()

    gen_graphs = []
    uncorrected_graphs_count = 0  # Changed name for clarity
    num_samples_processed = 0

    for i_sample, (x_elem, adj_elem) in enumerate(zip(x, adj)):
        print(f"\n--- Processing sample {i_sample + 1}/{len(x)} ---")
        print(f"  x_elem shape: {x_elem.shape}, adj_elem shape: {adj_elem.shape}")
        num_samples_processed += 1

        aig_constructed = construct_aig(x_elem, adj_elem)
        # print_graph_summary(aig_constructed, f"Sample {i_sample}_construct_aig") # Covered by construct_aig's end print

        aig_corrected, pure_flag = correct_fanins(aig_constructed)
        # print_graph_summary(aig_corrected, f"Sample {i_sample}_correct_fanins") # Covered by correct_fanins's end print

        uncorrected_graphs_count += pure_flag  # pure_flag is 1 if no corrections, 0 if corrections or failed
        print(f"  Sample {i_sample}: pure_flag from correct_fanins: {pure_flag}")

        aig_canonicalized = valid_aig_can_with_seg(aig_corrected)
        # print_graph_summary(aig_canonicalized, f"Sample {i_sample}_valid_aig_can_with_seg") # Covered

        if aig_canonicalized is None:
            print(f"  Sample {i_sample}: Resulting AIG is None after all processing steps.")
        else:
            print(
                f"  Sample {i_sample}: Final AIG has {aig_canonicalized.number_of_nodes()} nodes, {aig_canonicalized.number_of_edges()} edges.")

        gen_graphs.append(aig_canonicalized)

        if num_samples_processed >= 2 and len(x) > 2:  # Limit debug output for many samples
            print("\nDEBUG: Further sample processing will be less verbose...")
            # Potentially reduce verbosity of called functions for subsequent iterations if needed
            # For now, this only stops the per-sample header in this loop.
            if num_samples_processed == 2:  # one more full print
                pass  # allow next one to print fully
            elif num_samples_processed > 3:  # from the 4th sample onwards only this message
                print(f"--- Processing sample {i_sample + 1}/{len(x)} (verbose output reduced) ---")

    print(f"\nDEBUG: gen_mol_from_one_shot_tensor finished processing {num_samples_processed} samples.")
    print(f"  Total graphs generated (some might be None): {len(gen_graphs)}")
    print(f"  Count of graphs 'pure' after correct_fanins (uncorrected_graphs_count): {uncorrected_graphs_count}")
    return gen_graphs, uncorrected_graphs_count