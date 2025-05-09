### Code adapted from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow
import re
import numpy as np
import networkx as nx
from aig_config import *  # Imports NODE_TYPE_KEYS, NUM_NODE_FEATURES, VIRTUAL_EDGE_INDEX, NUM2EDGETYPE, check_validity, Fan_ins
import warnings  # Ensure warnings is imported

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
    prefix = f"DEBUG: Sample {sample_idx}: " if sample_idx is not None else "DEBUG: "
    if g is None:
        print(f"{prefix}Graph at stage '{stage_name}' is None.")
        return

    print(f"{prefix}Graph summary at stage '{stage_name}':")
    print(f"{prefix}  Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")

    node_types = Counter(d.get('type', 'UNTYPED') for _, d in g.nodes(data=True))
    print(f"{prefix}  Node Types: {dict(node_types)}")

    edge_types = Counter(d.get('type', 'UNTYPED') for _, _, d in g.edges(data=True))
    print(f"{prefix}  Edge Types: {dict(edge_types)}")

    if g.number_of_nodes() > 0:
        is_dag_val = nx.is_directed_acyclic_graph(g)
        print(f"{prefix}  Is DAG? {is_dag_val}")
        if not is_dag_val:
            try:
                cycles = list(nx.simple_cycles(g))
                print(f"{prefix}    Cycles found (first few): {cycles[:min(3, len(cycles))]}")
            except Exception as e_cycle:
                print(f"{prefix}    Error finding cycles: {e_cycle}")

        validity_check_result = check_validity(g)  # Uses aig_config.check_validity
        print(f"{prefix}  Validity (check_validity result): {validity_check_result}")
    else:
        print(f"{prefix}  Is DAG? N/A (empty graph)")
        print(
            f"{prefix}  Validity (check_validity result): True (empty graph typically considered valid by default in check_validity)")


def construct_aig(x_features_matrix, adj_channel_tensor, node_type_list=NODE_TYPE_KEYS, sample_idx_debug=None):
    prefix = f"DEBUG: Sample {sample_idx_debug}: construct_aig: " if sample_idx_debug is not None else "DEBUG: construct_aig: "
    print(f"\n{prefix}Called.")
    print(f"{prefix}  x_features_matrix shape: {x_features_matrix.shape}, dtype: {x_features_matrix.dtype}")
    print(f"{prefix}  adj_channel_tensor shape: {adj_channel_tensor.shape}, dtype: {adj_channel_tensor.dtype}")
    print(f"{prefix}  node_type_list used for decoding: {node_type_list} (len: {len(node_type_list)})")

    # Print a small part of the input tensors to see their nature
    if x_features_matrix.size > 0:
        print(f"{prefix}  x_features_matrix (first 3 rows, all columns):\n{x_features_matrix[:3, :]}")
    if adj_channel_tensor.size > 0:
        print(f"{prefix}  adj_channel_tensor (channel 0, first 3x3 block norms):\n{adj_channel_tensor[0, :3, :3]}")
        if adj_channel_tensor.shape[0] > 1:
            print(f"{prefix}  adj_channel_tensor (channel 1, first 3x3 block norms):\n{adj_channel_tensor[1, :3, :3]}")

    aig = nx.DiGraph()
    node_type_indices_all = np.argmax(x_features_matrix, axis=1)
    print(
        f"{prefix}  node_type_indices_all (argmax over last dim of x_features_matrix, first 10 values): {node_type_indices_all[:10]}")

    virtual_node_channel_matrix_idx = NUM_NODE_FEATURES
    print(
        f"{prefix}  virtual_node_channel_matrix_idx (from aig_config.NUM_NODE_FEATURES): {virtual_node_channel_matrix_idx}")

    active_node_mask = node_type_indices_all != virtual_node_channel_matrix_idx
    print(f"{prefix}  active_node_mask (first 10 values): {active_node_mask[:10]}")
    num_active_nodes_calc = active_node_mask.sum()
    print(f"{prefix}  Number of active nodes identified by mask: {num_active_nodes_calc}")

    active_node_assigned_type_indices = node_type_indices_all[active_node_mask]
    print(
        f"{prefix}  active_node_assigned_type_indices (indices for active nodes, first 10 if any): {active_node_assigned_type_indices[:10] if active_node_assigned_type_indices.size > 0 else 'None'}")

    nodes_added_count = 0
    for compact_idx, assigned_type_idx in enumerate(active_node_assigned_type_indices):
        if assigned_type_idx >= len(node_type_list):
            warnings.warn(
                f"{prefix}Internal Error: assigned_type_idx ({assigned_type_idx}) is out of bounds for "
                f"node_type_list (len {len(node_type_list)}) even after filtering. "
            )
            print(
                f"{prefix}SKIPPING node add for compact_idx {compact_idx} due to out-of-bounds assigned_type_idx {assigned_type_idx}")
            continue

        actual_node_type_str = node_type_list[assigned_type_idx]
        aig.add_node(compact_idx, type=actual_node_type_str)
        nodes_added_count += 1
        if nodes_added_count <= 5: print(
            f"{prefix}  Added node {compact_idx} with type '{actual_node_type_str}' (original index from argmax: {assigned_type_idx})")
    if nodes_added_count > 5: print(f"{prefix}  ... and {nodes_added_count - 5} more nodes.")

    adj_matrix_edge_indices = np.argmax(adj_channel_tensor, axis=0)  # Shape (max_nodes, max_nodes)
    adj_matrix_edge_scores = np.max(adj_channel_tensor, axis=0)
    print(
        f"{prefix}  adj_matrix_edge_indices (argmax over channel dim of adj_tensor, first 5x5 block):\n{adj_matrix_edge_indices[:5, :5]}")
    print(f"{prefix}  VIRTUAL_EDGE_INDEX (for 'NONE' edge type) from aig_config: {VIRTUAL_EDGE_INDEX}")

    edges_added_count = 0
    if num_active_nodes_calc > 0:  # Proceed only if there are active nodes
        original_indices_of_active_nodes = np.where(active_node_mask)[0]
        map_original_to_compact = {orig_idx: compact_idx for compact_idx, orig_idx in
                                   enumerate(original_indices_of_active_nodes)}
        print(
            f"{prefix}  original_indices_of_active_nodes (indices in x_features_matrix that are active, first 10): {original_indices_of_active_nodes[:10]}")

        for start_node_orig_idx in original_indices_of_active_nodes:
            for end_node_orig_idx in original_indices_of_active_nodes:
                if start_node_orig_idx == end_node_orig_idx:
                    continue

                edge_type_idx_in_adj = adj_matrix_edge_indices[start_node_orig_idx, end_node_orig_idx]

                if edge_type_idx_in_adj != VIRTUAL_EDGE_INDEX:
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
                                f"{prefix}  Added edge ({start_node_compact_idx}-{end_node_compact_idx}) type '{actual_edge_type_str}', score {edge_score:.2f} (orig_indices: {start_node_orig_idx}-{end_node_orig_idx}, edge_type_idx_from_adj: {edge_type_idx_in_adj})")
                    else:
                        # This warning implies that edge_type_idx_in_adj was NOT VIRTUAL_EDGE_INDEX,
                        # but also not in NUM2EDGETYPE (e.g. if it was an unexpected index).
                        # Given VIRTUAL_EDGE_INDEX=2 and NUM2EDGETYPE covers 0 and 1, this condition should ideally not be met if adj_channel_tensor has 3 channels.
                        warnings.warn(
                            f"{prefix}Unexpected edge_type_idx_in_adj: {edge_type_idx_in_adj} for an active edge between original nodes {start_node_orig_idx}-{end_node_orig_idx}. NUM2EDGETYPE: {NUM2EDGETYPE}")
                        print(
                            f"{prefix}SKIPPING edge add for orig_indices {start_node_orig_idx}-{end_node_orig_idx} due to unexpected edge_type_idx_in_adj {edge_type_idx_in_adj}")

        if edges_added_count > 5: print(f"{prefix}  ... and {edges_added_count - 5} more edges.")
        print(f"{prefix}  Total edges added in construct_aig: {edges_added_count}")
    else:
        print(f"{prefix}No active nodes, so no edges were added in construct_aig.")

    print_graph_summary(aig, "construct_aig_end", sample_idx=sample_idx_debug)
    return aig


# Fan_ins is already imported from aig_config via *

def correct_fanins(aig, sample_idx_debug=None):
    prefix = f"DEBUG: Sample {sample_idx_debug}: correct_fanins: " if sample_idx_debug is not None else "DEBUG: correct_fanins: "
    print(f"\n{prefix}Called.")
    if aig is None:
        print(f"{prefix}Input AIG is None. Returning None.")
        return None, 0

    print_graph_summary(aig, "correct_fanins_start", sample_idx=sample_idx_debug)
    is_initially_valid = check_validity(aig)
    print(f"{prefix}  Initial validity (check_validity before corrections): {is_initially_valid}")

    if is_initially_valid:
        print(f"{prefix}  AIG already valid before fan-in correction. Returning as is.")
        return aig, 1

    corrected_graph = aig.copy()  # Work on a copy to observe changes more clearly if needed, though original modifies in-place. Let's stick to in-place for now.

    print(f"{prefix}  Attempting fan-in corrections...")
    for node_id, node_data in list(corrected_graph.nodes(data=True)):
        node_type = node_data.get('type')
        if node_type is None:
            print(f"{prefix}  Skipping node {node_id} in fan-in correction: no 'type' attribute.")
            continue
        if node_type not in Fan_ins:  # Fan_ins is from aig_config
            print(
                f"{prefix}  Skipping node {node_id} (type: {node_type}) in fan-in correction: type not in Fan_ins dict ({Fan_ins}).")
            continue

        current_in_degree = corrected_graph.in_degree(node_id)
        target_fan_in = Fan_ins[node_type]
        # print(f"{prefix}  Node {node_id} (type: {node_type}): Current In-degree: {current_in_degree}, Target Fan-in: {target_fan_in}")

        if current_in_degree > target_fan_in:
            print(
                f"{prefix}    Correcting fan-in for node {node_id} (type: {node_type}). In-degree {current_in_degree} > Target {target_fan_in}")
            incoming_edges_with_scores = []
            for u, v, edge_attributes in corrected_graph.in_edges(node_id, data=True):
                score = edge_attributes.get('score', float('-inf'))
                incoming_edges_with_scores.append(((u, v), score))

            incoming_edges_with_scores.sort(key=lambda x: x[1], reverse=True)
            if len(incoming_edges_with_scores) > 3:
                print(
                    f"{prefix}      Sorted incoming edges (first 3 of {len(incoming_edges_with_scores)}): {incoming_edges_with_scores[:3]}...")
            else:
                print(f"{prefix}      Sorted incoming edges: {incoming_edges_with_scores}")

            edges_to_remove = incoming_edges_with_scores[target_fan_in:]
            print(
                f"{prefix}      Edges to remove (to meet target fan-in {target_fan_in}): {len(edges_to_remove)} edges.")
            for i_rem, ((u_rem, v_rem), score_rem) in enumerate(edges_to_remove):
                if corrected_graph.has_edge(u_rem, v_rem):
                    corrected_graph.remove_edge(u_rem, v_rem)
                    if i_rem < 3: print(f"{prefix}      Removed edge ({u_rem},{v_rem}) with score {score_rem:.2f}")
                # else: # This case might be too verbose if many edges
                #     print(f"{prefix}      Attempted to remove edge ({u_rem},{v_rem}) but it was already removed.")
            if len(edges_to_remove) > 3: print(f"{prefix}      ... and {len(edges_to_remove) - 3} more edges removed.")

        # PO nodes should have out-degree 0.
        if node_type == NODE_TYPE_KEYS[3]:  # NODE_PO
            current_out_degree = corrected_graph.out_degree(node_id)
            if current_out_degree > 0:
                print(
                    f"{prefix}    Correcting out-degree for PO node {node_id}. Out-degree is {current_out_degree}, should be 0.")
                successors_to_remove = list(corrected_graph.successors(node_id))
                if len(successors_to_remove) > 3:
                    print(
                        f"{prefix}      Successors to remove edges to (first 3 of {len(successors_to_remove)}): {successors_to_remove[:3]}...")
                else:
                    print(f"{prefix}      Successors to remove edges to: {successors_to_remove}")

                removed_count = 0
                for target_node_succ in successors_to_remove:
                    if corrected_graph.has_edge(node_id, target_node_succ):
                        corrected_graph.remove_edge(node_id, target_node_succ)
                        removed_count += 1
                        if removed_count <= 3: print(
                            f"{prefix}      Removed outgoing edge ({node_id} -> {target_node_succ}) from PO node.")
                if removed_count > 3: print(
                    f"{prefix}      ... and {removed_count - 3} more outgoing edges removed from PO node.")

    print_graph_summary(corrected_graph, "correct_fanins_intermediate_after_loops", sample_idx=sample_idx_debug)
    is_valid_after_correction = check_validity(corrected_graph)
    print(f"{prefix}  Validity after fan-in corrections (check_validity result): {is_valid_after_correction}")

    if is_valid_after_correction:
        print(f"{prefix}  AIG valid after fan-in corrections.")
        return corrected_graph, 0
    else:
        print(f"{prefix}Warning: issues everywhere still wtf")  # Your original warning
        print_graph_summary(corrected_graph, "correct_fanins_fail_validity_check", sample_idx=sample_idx_debug)
        print(f"{prefix}  AIG still invalid after fan-in corrections. Returning None.")
        return None, 0


def valid_aig_can_with_seg(aig: nx.DiGraph, sample_idx_debug=None):
    prefix = f"DEBUG: Sample {sample_idx_debug}: valid_aig_can_with_seg: " if sample_idx_debug is not None else "DEBUG: valid_aig_can_with_seg: "
    print(f"\n{prefix}Called.")
    if aig is None:
        print(f"{prefix}Input AIG is None. Returning None.")
        return None
    print_graph_summary(aig, "valid_aig_can_with_seg_start", sample_idx=sample_idx_debug)

    if not isinstance(aig, nx.DiGraph):
        print(f"{prefix}Input is not a NetworkX DiGraph (type: {type(aig)}). Returning as is.")
        return aig

    if aig.number_of_nodes() == 0:  # Handle empty graph explicitly before DAG check
        print(f"{prefix}Input graph is empty. Returning as is.")
        return aig

    is_dag = nx.is_directed_acyclic_graph(aig)
    print(f"{prefix}  Is DAG? {is_dag}")
    if not is_dag:
        print(f"{prefix}  Graph is not a DAG. Returning as is (no canonicalization or isolation removal).")
        return aig

    print(f"{prefix}  Graph is a DAG. Relabeling nodes based on topological sort.")
    try:
        topo_sorted_nodes = list(nx.topological_sort(aig))
        mapping = {old_label: new_label for new_label, old_label in enumerate(topo_sorted_nodes)}
        # print(f"{prefix}    Topological sort mapping (first 5 if many): {list(mapping.items())[:5]}") # Can be long
        aig = nx.relabel_nodes(aig, mapping, copy=True)
        print_graph_summary(aig, "valid_aig_can_with_seg_after_relabel", sample_idx=sample_idx_debug)
    except Exception as e:
        print(f"{prefix}  Error during topological sort or relabeling: {e}. Returning graph before this attempt.")
        return aig  # Return graph state before this failed block

    if aig.number_of_nodes() > 1:
        isolates = list(nx.isolates(aig))
        if isolates:
            print(
                f"{prefix}  Removing {len(isolates)} isolated nodes: {isolates[:5] if len(isolates) > 5 else isolates}...")
            aig.remove_nodes_from(isolates)
            # print_graph_summary(aig, "valid_aig_can_with_seg_after_isolate_removal", sample_idx=sample_idx_debug) # Can be verbose, summary at end
        else:
            print(f"{prefix}  No isolated nodes found to remove.")
    else:
        print(f"{prefix}  Graph has <= 1 node, skipping isolate removal.")

    print_graph_summary(aig, "valid_aig_can_with_seg_end", sample_idx=sample_idx_debug)
    return aig


def gen_mol_from_one_shot_tensor(adj, x, largest_connected_comp=True):  # largest_connected_comp seems unused
    prefix = "DEBUG: gen_mol_from_one_shot_tensor: "
    print(f"\n{prefix}Called.")
    print(f"{prefix}  Input x shape: {x.shape}, Input adj shape: {adj.shape}")

    x = x.permute(0, 2, 1)
    print(f"{prefix}  x shape after permute (B, N, NodeChannels): {x.shape}")
    adj = adj.cpu().detach().numpy()
    x = x.cpu().detach().numpy()

    gen_graphs = []
    uncorrected_graphs_count = 0
    num_samples_to_process = x.shape[0]
    print(f"{prefix}  Processing {num_samples_to_process} samples...")

    for i_sample, (x_elem, adj_elem) in enumerate(zip(x, adj)):
        print(f"\n--- {prefix}Processing sample {i_sample + 1}/{num_samples_to_process} ---")
        print(
            f"{prefix}  x_elem shape (N, NodeChannels): {x_elem.shape}, adj_elem shape (EdgeChannels, N, N): {adj_elem.shape}")

        aig_constructed = construct_aig(x_elem, adj_elem, sample_idx_debug=i_sample + 1)

        aig_corrected, pure_flag = correct_fanins(aig_constructed, sample_idx_debug=i_sample + 1)

        uncorrected_graphs_count += pure_flag
        print(
            f"{prefix}  Sample {i_sample + 1}: pure_flag from correct_fanins: {pure_flag} (1 means was valid before/no changes, 0 means corrections attempted or failed)")

        aig_canonicalized = valid_aig_can_with_seg(aig_corrected, sample_idx_debug=i_sample + 1)

        if aig_canonicalized is None:
            print(f"{prefix}  Sample {i_sample + 1}: Resulting AIG is None after all processing steps.")
        else:
            print(
                f"{prefix}  Sample {i_sample + 1}: Final AIG for this sample has {aig_canonicalized.number_of_nodes()} nodes, {aig_canonicalized.number_of_edges()} edges.")

        gen_graphs.append(aig_canonicalized)

        if i_sample >= 1 and num_samples_to_process > 2:  # Limit full debug output to first 2 samples if many
            print(f"\n{prefix}--- Further sample processing will be less verbose (summary prints only) ---")
            # This is a placeholder; true reduction of verbosity would require passing a flag
            # to the sub-functions or globally changing a debug level.
            # For now, all debug prints within sub-functions will still occur.
            break  # For now, let's just process 2 samples fully for debugging if more are present.

    print(f"\n{prefix}Finished processing. Generated {len(gen_graphs)} graph objects (some might be None).")
    print(
        f"{prefix}  Count of graphs initially 'pure' and valid before/during correct_fanins: {uncorrected_graphs_count}")
    return gen_graphs, uncorrected_graphs_count