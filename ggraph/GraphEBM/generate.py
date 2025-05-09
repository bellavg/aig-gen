### Code adapted from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow
import re
import numpy as np
import networkx as nx
from aig_config import *  # Imports NODE_TYPE_KEYS, NUM_NODE_FEATURES, VIRTUAL_EDGE_INDEX, NUM2EDGETYPE, check_validity, Fan_ins
import warnings
from collections import Counter

# For detailed debugging of array contents (can be commented out for production)
# np.set_printoptions(threshold=50, edgeitems=5)

# These are likely not needed for AIGs
# ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
# bond_decoder_m = {1: "EDGE_REG", 2: "EDGE_INV", 3: "NONE"}

# Initial print statements for config loading (can be removed if too verbose)
print(f"INFO: generate.py: Loaded NODE_TYPE_KEYS from aig_config: {NODE_TYPE_KEYS}")
print(f"INFO: generate.py: Loaded NUM_NODE_FEATURES from aig_config: {NUM_NODE_FEATURES}")
print(f"INFO: generate.py: Loaded VIRTUAL_EDGE_INDEX from aig_config (for 'NONE' edge): {VIRTUAL_EDGE_INDEX}")
print(f"INFO: generate.py: Loaded NUM2EDGETYPE (for actual edges) from aig_config: {NUM2EDGETYPE}")
print(f"INFO: generate.py: Loaded Fan_ins from aig_config: {Fan_ins}")


def print_graph_summary(g, stage_name="Graph", sample_idx=None, verbose=False):
    """
    Prints a summary of the graph's properties.
    Set verbose=True for more detailed output including cycle examples.
    """
    prefix = f"INFO: Sample {sample_idx}: " if sample_idx is not None else "INFO: "
    if g is None:
        print(f"{prefix}Graph at stage '{stage_name}' is None.")
        return

    print(f"{prefix}Graph summary at stage '{stage_name}': Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")

    if verbose:
        node_types = Counter(d.get('type', 'UNTYPED') for _, d in g.nodes(data=True))
        print(f"{prefix}  Node Types: {dict(node_types)}")
        edge_types = Counter(d.get('type', 'UNTYPED') for _, _, d in g.edges(data=True))
        print(f"{prefix}  Edge Types: {dict(edge_types)}")

    if g.number_of_nodes() > 0:
        is_dag_val = nx.is_directed_acyclic_graph(g)
        print(f"{prefix}  Is DAG? {is_dag_val}")

        if not is_dag_val and verbose:
            print(f"{prefix}    Attempting to find a few cycles for non-DAG graph...")
            found_cycles_list = []
            try:
                for cycle_count, cycle in enumerate(nx.simple_cycles(g)):
                    found_cycles_list.append(cycle)
                    if cycle_count >= 2: break
                if found_cycles_list:
                    print(f"{prefix}    Cycles found (first {len(found_cycles_list)}): {found_cycles_list}")
                else:
                    print(
                        f"{prefix}    No simple cycles reported by nx.simple_cycles (or graph too complex for quick check).")
            except Exception as e_cycle:
                print(f"{prefix}    Error finding cycles: {e_cycle}")

        validity_check_result = check_validity(g)
        print(f"{prefix}  Validity (check_validity): {validity_check_result}")
    else:
        print(f"{prefix}  Is DAG? N/A (empty graph)")
        print(f"{prefix}  Validity (check_validity): True (empty graph)")


def construct_aig(x_features_matrix, adj_channel_tensor, node_type_list=NODE_TYPE_KEYS,
                  edge_score_threshold=0.5, sample_idx_debug=None, verbose_construct=False):
    """
    Constructs an AIG from raw feature matrices with edge score thresholding.
    """
    prefix = f"INFO: Sample {sample_idx_debug}: construct_aig: " if sample_idx_debug is not None else "INFO: construct_aig: "
    if verbose_construct:
        print(f"\n{prefix}Called.")
        print(
            f"{prefix}  x_features_matrix shape: {x_features_matrix.shape}, adj_channel_tensor shape: {adj_channel_tensor.shape}")
        print(f"{prefix}  Using edge_score_threshold: {edge_score_threshold}")

    aig = nx.DiGraph()
    node_type_indices_all = np.argmax(x_features_matrix, axis=1)
    virtual_node_channel_matrix_idx = NUM_NODE_FEATURES
    active_node_mask = node_type_indices_all != virtual_node_channel_matrix_idx
    num_active_nodes_calc = active_node_mask.sum()

    if verbose_construct:
        print(f"{prefix}  Number of active nodes identified by mask: {num_active_nodes_calc}")

    active_node_assigned_type_indices = node_type_indices_all[active_node_mask]
    for compact_idx, assigned_type_idx in enumerate(active_node_assigned_type_indices):
        if assigned_type_idx >= len(node_type_list):
            warnings.warn(f"{prefix}Internal Error: assigned_type_idx ({assigned_type_idx}) out of bounds.")
            continue
        aig.add_node(compact_idx, type=node_type_list[assigned_type_idx])

    adj_matrix_edge_indices = np.argmax(adj_channel_tensor, axis=0)
    adj_matrix_edge_scores = np.max(adj_channel_tensor, axis=0)

    edges_added_count = 0
    edges_skipped_due_to_threshold_count = 0

    if num_active_nodes_calc > 0:
        original_indices_of_active_nodes = np.where(active_node_mask)[0]
        map_original_to_compact = {orig_idx: compact_idx for compact_idx, orig_idx in
                                   enumerate(original_indices_of_active_nodes)}

        for start_node_orig_idx in original_indices_of_active_nodes:
            for end_node_orig_idx in original_indices_of_active_nodes:
                if start_node_orig_idx == end_node_orig_idx:
                    continue
                edge_type_idx_in_adj = adj_matrix_edge_indices[start_node_orig_idx, end_node_orig_idx]
                if edge_type_idx_in_adj != VIRTUAL_EDGE_INDEX:
                    edge_score = adj_matrix_edge_scores[start_node_orig_idx, end_node_orig_idx]
                    if edge_score >= edge_score_threshold:
                        start_node_compact_idx = map_original_to_compact[start_node_orig_idx]
                        end_node_compact_idx = map_original_to_compact[end_node_orig_idx]
                        if edge_type_idx_in_adj in NUM2EDGETYPE:
                            actual_edge_type_str = NUM2EDGETYPE[edge_type_idx_in_adj]
                            aig.add_edge(int(start_node_compact_idx), int(end_node_compact_idx),
                                         type=actual_edge_type_str, score=float(edge_score))
                            edges_added_count += 1
                        else:
                            warnings.warn(f"{prefix}Unexpected edge_type_idx_in_adj: {edge_type_idx_in_adj}")
                    else:
                        edges_skipped_due_to_threshold_count += 1

        if verbose_construct:
            print(f"{prefix}  Total edges added: {edges_added_count}")
            print(
                f"{prefix}  Total edges skipped (threshold {edge_score_threshold}): {edges_skipped_due_to_threshold_count}")

    if verbose_construct or sample_idx_debug is not None:  # Always print summary if sample_idx_debug is provided
        print_graph_summary(aig, "construct_aig_end", sample_idx_debug, verbose=verbose_construct)
    return aig


def correct_fanins(aig, sample_idx_debug=None, verbose_correct=False, max_cycle_break_attempts=5):
    """
    Corrects fan-ins, PO out-degrees, and attempts to break a limited number of cycles.
    """
    prefix = f"INFO: Sample {sample_idx_debug}: correct_fanins: " if sample_idx_debug is not None else "INFO: correct_fanins: "

    if aig is None:
        if verbose_correct: print(f"{prefix}Input AIG is None. Returning None.")
        return None, 0

    if verbose_correct:
        print(f"\n{prefix}Called.")
        print_graph_summary(aig, "correct_fanins_start", sample_idx_debug, verbose=True)

    is_initially_valid = check_validity(aig)
    if verbose_correct: print(f"{prefix}  Initial validity: {is_initially_valid}")

    if is_initially_valid:
        if verbose_correct: print(f"{prefix}  AIG already valid. Returning as is.")
        return aig, 1

        # --- Fan-in and PO Out-degree corrections ---
    graph_changed_by_fanin = False
    for node_id, node_data in list(aig.nodes(data=True)):  # Iterate over a copy of node list
        node_type = node_data.get('type')
        if node_type is None or node_type not in Fan_ins:
            continue

        current_in_degree = aig.in_degree(node_id)
        target_fan_in = Fan_ins[node_type]

        if current_in_degree > target_fan_in:
            graph_changed_by_fanin = True
            if verbose_correct: print(
                f"{prefix}    Correcting fan-in for node {node_id} ({node_type}): {current_in_degree} > {target_fan_in}")
            incoming_edges = sorted(aig.in_edges(node_id, data=True), key=lambda x: x[2].get('score', float('-inf')),
                                    reverse=True)
            for u, v, data in incoming_edges[target_fan_in:]:
                if aig.has_edge(u, v): aig.remove_edge(u, v)

        if node_type == NODE_TYPE_KEYS[3]:  # NODE_PO
            if aig.out_degree(node_id) > 0:
                graph_changed_by_fanin = True
                if verbose_correct: print(f"{prefix}    Correcting out-degree for PO node {node_id}.")
                edges_to_remove = list(aig.out_edges(node_id))
                aig.remove_edges_from(edges_to_remove)

    # --- Cycle Breaking (if still not a DAG) ---
    graph_changed_by_cycle_breaking = False
    if not nx.is_directed_acyclic_graph(aig):
        if verbose_correct: print(
            f"{prefix}  Graph not a DAG after fan-in. Attempting cycle breaking (max {max_cycle_break_attempts} attempts)...")
        for attempt in range(max_cycle_break_attempts):
            if nx.is_directed_acyclic_graph(aig):
                if verbose_correct: print(f"{prefix}    Graph became a DAG after {attempt} cycle(s) broken.")
                break
            try:
                cycle_nodes = next(nx.simple_cycles(aig), None)
                if not cycle_nodes:
                    if verbose_correct: print(f"{prefix}    No more cycles found by simple_cycles.")
                    break

                if verbose_correct: print(f"{prefix}    Found cycle: {cycle_nodes}. Breaking...")
                edges_in_cycle_with_scores = []
                for i in range(len(cycle_nodes)):
                    u, v = cycle_nodes[i], cycle_nodes[(i + 1) % len(cycle_nodes)]
                    if aig.has_edge(u, v):
                        edges_in_cycle_with_scores.append(((u, v), aig.edges[u, v].get('score', float('-inf'))))

                if not edges_in_cycle_with_scores:
                    if verbose_correct: print(
                        f"{prefix}    Could not identify scored edges in cycle {cycle_nodes}. Stopping cycle break attempt.")
                    break

                edges_in_cycle_with_scores.sort(key=lambda x: x[1])  # Sort by score, ascending
                edge_to_remove, score_removed = edges_in_cycle_with_scores[0]

                if aig.has_edge(edge_to_remove[0], edge_to_remove[1]):
                    aig.remove_edge(edge_to_remove[0], edge_to_remove[1])
                    graph_changed_by_cycle_breaking = True
                    if verbose_correct: print(
                        f"{prefix}      Removed edge {edge_to_remove} (score {score_removed:.2f})")
                else:  # Should not happen if logic is correct
                    if verbose_correct: print(
                        f"{prefix}      Edge {edge_to_remove} to break cycle not found (unexpected).")
                    break
            except nx.NetworkXNoCycle:
                if verbose_correct: print(f"{prefix}    No cycles found, graph is now a DAG.")
                break
            except Exception as e_cycle:
                if verbose_correct: print(f"{prefix}    Error during cycle breaking: {e_cycle}")
                break
        else:  # Loop finished without break (i.e., max_cycle_break_attempts reached)
            if not nx.is_directed_acyclic_graph(aig) and verbose_correct:
                print(f"{prefix}  Graph still not a DAG after {max_cycle_break_attempts} cycle breaking attempts.")

    if verbose_correct or sample_idx_debug is not None:
        print_graph_summary(aig, "correct_fanins_end", sample_idx_debug, verbose=verbose_correct)

    is_valid_after_all_corrections = check_validity(aig)
    if verbose_correct: print(f"{prefix}  Final validity: {is_valid_after_all_corrections}")

    if is_valid_after_all_corrections:
        return aig, 0  # 0 indicates corrections were made or attempted
    else:
        if verbose_correct: print(f"{prefix}  AIG still invalid after all corrections. Returning None.")
        return None, 0


def valid_aig_can_with_seg(aig: nx.DiGraph, sample_idx_debug=None, verbose_canonicalize=False):
    """
    Canonicalizes a valid AIG by relabeling and removing isolates.
    """
    prefix = f"INFO: Sample {sample_idx_debug}: valid_aig_can_with_seg: " if sample_idx_debug is not None else "INFO: valid_aig_can_with_seg: "
    if aig is None:
        if verbose_canonicalize: print(f"{prefix}Input AIG is None.")
        return None

    if verbose_canonicalize:
        print(f"\n{prefix}Called.")
        print_graph_summary(aig, "valid_aig_can_with_seg_start", sample_idx_debug, verbose=True)

    if not isinstance(aig, nx.DiGraph):
        warnings.warn(f"{prefix}Input is not a NetworkX DiGraph (type: {type(aig)}).")
        return aig
    if aig.number_of_nodes() == 0:
        if verbose_canonicalize: print(f"{prefix}Input graph is empty.")
        return aig

    if not nx.is_directed_acyclic_graph(aig):
        if verbose_canonicalize: print(f"{prefix}  Graph is not a DAG. Skipping canonicalization.")
        return aig

    try:
        if verbose_canonicalize: print(f"{prefix}  Graph is a DAG. Relabeling nodes...")
        topo_sorted_nodes = list(nx.topological_sort(aig))
        mapping = {old_label: new_label for new_label, old_label in enumerate(topo_sorted_nodes)}
        aig = nx.relabel_nodes(aig, mapping, copy=True)
    except Exception as e:
        warnings.warn(f"{prefix}  Error during topological sort or relabeling: {e}.")
        return aig

    if aig.number_of_nodes() > 1:
        isolates = list(nx.isolates(aig))
        if isolates:
            if verbose_canonicalize: print(f"{prefix}  Removing {len(isolates)} isolated nodes...")
            aig.remove_nodes_from(isolates)

    if verbose_canonicalize or sample_idx_debug is not None:
        print_graph_summary(aig, "valid_aig_can_with_seg_end", sample_idx_debug, verbose=verbose_canonicalize)
    return aig


def gen_mol_from_one_shot_tensor(adj, x,
                                 edge_score_thresh_for_construct=0.5,
                                 max_cycle_break_attempts_in_correct=5,
                                 verbose_generation=False):
    """
    Generates AIGs from batched tensor representations.
    """
    prefix = "INFO: gen_mol_from_one_shot_tensor: "
    print(f"\n{prefix}Called.")
    if verbose_generation:
        print(f"{prefix}  Input x shape: {x.shape}, Input adj shape: {adj.shape}")
        print(f"{prefix}  Using edge_score_threshold for construct_aig: {edge_score_thresh_for_construct}")
        print(f"{prefix}  Using max_cycle_break_attempts in correct_fanins: {max_cycle_break_attempts_in_correct}")

    x = x.permute(0, 2, 1)
    adj_np = adj.cpu().detach().numpy()
    x_np = x.cpu().detach().numpy()

    gen_graphs = []
    pure_graphs_count = 0
    num_samples_to_process = x_np.shape[0]
    print(f"{prefix}Processing {num_samples_to_process} samples...")

    for i_sample in range(num_samples_to_process):
        x_elem, adj_elem = x_np[i_sample], adj_np[i_sample]
        current_sample_prefix = f"INFO: Sample {i_sample + 1}/{num_samples_to_process}: "

        if verbose_generation:
            print(f"\n--- {current_sample_prefix}Starting processing ---")

        aig_constructed = construct_aig(x_elem, adj_elem,
                                        node_type_list=NODE_TYPE_KEYS,
                                        edge_score_threshold=edge_score_thresh_for_construct,
                                        sample_idx_debug=i_sample + 1 if verbose_generation else None,
                                        verbose_construct=verbose_generation)

        aig_corrected, pure_flag = correct_fanins(aig_constructed,
                                                  sample_idx_debug=i_sample + 1 if verbose_generation else None,
                                                  verbose_correct=verbose_generation,
                                                  max_cycle_break_attempts=max_cycle_break_attempts_in_correct)

        pure_graphs_count += pure_flag
        if verbose_generation:
            print(f"{current_sample_prefix}pure_flag from correct_fanins: {pure_flag}")

        aig_canonicalized = valid_aig_can_with_seg(aig_corrected,
                                                   sample_idx_debug=i_sample + 1 if verbose_generation else None,
                                                   verbose_canonicalize=verbose_generation)

        if aig_canonicalized is None:
            print(f"{current_sample_prefix}Resulting AIG is None after all processing steps.")
        else:
            print(
                f"{current_sample_prefix}Final AIG: {aig_canonicalized.number_of_nodes()} nodes, {aig_canonicalized.number_of_edges()} edges. Valid: {check_validity(aig_canonicalized)}")

        gen_graphs.append(aig_canonicalized)

    print(f"\n{prefix}Finished processing. Generated {len(gen_graphs)} graph objects (some might be None).")
    print(f"{prefix}  Count of graphs initially 'pure' (valid before corrections): {pure_graphs_count}")
    return gen_graphs, pure_graphs_count
