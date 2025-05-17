### Code adapted from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow
import re
import numpy as np
import networkx as nx
from aig_config import *  # Imports NODE_TYPE_KEYS, NUM_NODE_FEATURES, VIRTUAL_EDGE_INDEX, NUM2EDGETYPE, check_validity, Fan_ins
import warnings
from collections import Counter



def construct_mol(x, A, node_type_list=ALL_NODE_KEYS):
    aig = nx.Graph()
    atoms = np.argmax(x, axis=1)
    active_node_mask = atoms != len(node_type_list) - 1

    atoms = atoms[active_node_mask]

    for compact_idx, assigned_type_idx in enumerate(atoms):
        aig.add_node(compact_idx, type=node_type_list[assigned_type_idx])


    # A (edge_type, num_node, num_node)
    adj = np.argmax(A, axis=0)
    adj = np.array(adj)
    adj = adj[active_node_mask, :][:, active_node_mask]
    adj[adj == NO_EDGE_CHANNEL] = -1
    adj += 1
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            aig.add_edge(int(start), int(end), type=NUM2EDGETYPE[adj[start, end]])

    return aig



def correct_mol(x):

    aig = x
    aig = to_directed_aig(aig)

    if check_validity(aig):
        return aig, 1


    graph_copy = aig.copy()  # Work on a copy
    # Remove isolated
    isolates_step1 = [
        n for n in nx.isolates(graph_copy)
        if graph_copy.nodes[n].get('type') != NODE_CONST0_STR
    ]
    if isolates_step1:
        graph_copy.remove_nodes_from(isolates_step1)

    # remove incoming for pi and const 0
    for node_id in list(graph_copy.nodes()):  # Iterate on copy of node list
        if node_id not in graph_copy: continue
        node_type = graph_copy.nodes[node_id].get('type')
        if node_type == NODE_PI_STR or node_type == NODE_CONST0_STR:
            if graph_copy.in_degree(node_id) > 0:
                in_edges = list(graph_copy.in_edges(node_id))
                graph_copy.remove_edges_from(in_edges)

    # remove outgoing for po
    for node_id in list(graph_copy.nodes()):
        if node_id not in graph_copy: continue
        if graph_copy.nodes[node_id].get('type') == NODE_PO_STR:
            if graph_copy.out_degree(node_id) > 0:
                out_edges = list(graph_copy.out_edges(node_id))
                graph_copy.remove_edges_from(out_edges)

    # Step 4: Remove isolated nodes again (excluding CONST0)
    nodes_before_step4 = graph_copy.number_of_nodes()
    isolates_step4 = [
        n for n in nx.isolates(graph_copy)
        if graph_copy.nodes[n].get('type') != NODE_CONST0_STR
    ]
    if isolates_step4:
        graph_copy.remove_nodes_from(isolates_step4)

    if check_validity(aig):
        return graph_copy, 0

    made_changes_in_current_pass = True  # Assume changes might be made, to enter loop at least once for iterative potential
    iteration_count = 0
    max_iterations = 20  # Heuristic limit for iterations
    while made_changes_in_current_pass and iteration_count < max_iterations:  # Loop for stability
        made_changes_in_current_pass = False
        iteration_count += 1
        is_dag = nx.is_directed_acyclic_graph(graph_copy)
        processing_order = []
        if is_dag:
            try:
                # Process from outputs towards inputs for dangling AND removal logic
                processing_order = reversed(list(nx.topological_sort(graph_copy)))
            except nx.NetworkXUnfeasible:
                is_dag = False  # Fallback
        if not is_dag:  # Handles initial non-DAG or fallback
            processing_order = sorted(list(graph_copy.nodes()), reverse=True)

        nodes_to_iterate = list(processing_order)

        for node_id in nodes_to_iterate:
            if node_id not in graph_copy:  # Node might have been removed
                continue
            node_attributes = graph_copy.nodes[node_id]
            node_type = node_attributes.get('type')
            action_taken_on_node = False

            if node_type == NODE_PO_STR:
                target_fan_in_po = Fan_ins.get(NODE_PO_STR, 1)
                current_in_degree = graph_copy.in_degree(node_id)
                if current_in_degree > target_fan_in_po:
                    in_edges_po = list(graph_copy.in_edges(node_id))
                    edges_to_remove_po = in_edges_po[target_fan_in_po:]  # Keep first, remove rest
                    if edges_to_remove_po:
                        graph_copy.remove_edges_from(edges_to_remove_po)
                        corrected_anything_overall = True
                        made_changes_in_current_pass = True
                        action_taken_on_node = True

            elif node_type == NODE_AND_STR:
                if graph_copy.out_degree(node_id) == 0:  # Dangling AND
                    in_edges_and_dangling = list(graph_copy.in_edges(node_id))
                    if in_edges_and_dangling:
                        graph_copy.remove_edges_from(in_edges_and_dangling)

                    graph_copy.remove_node(node_id)  # Remove the node itself
                    corrected_anything_overall = True
                    made_changes_in_current_pass = True
                    action_taken_on_node = True
                else:  # AND node has outgoing edges, check its fan-in
                    target_fan_in_and = Fan_ins.get(NODE_AND_STR, 2)
                    current_in_degree_and = graph_copy.in_degree(node_id)
                    if current_in_degree_and > target_fan_in_and:
                        in_edges_and = list(graph_copy.in_edges(node_id))
                        edges_to_remove_and = in_edges_and[target_fan_in_and:]  # Keep first two, remove rest
                        if edges_to_remove_and:
                            graph_copy.remove_edges_from(edges_to_remove_and)
                            corrected_anything_overall = True
                            made_changes_in_current_pass = True
                            action_taken_on_node = True

    return graph_copy, 0






def gen_mol_from_one_shot_tensor(adj, x):
    r"""Construct molecules from the node tensors and adjacency tensors generated by one-shot molecular graph generation methods.

    Args:
        adj (Tensor): The adjacency tensor with shape [:obj:`number of samples`, :obj:`number of possible bond types`, :obj:`maximum number of atoms`, :obj:`maximum number of atoms`].
        x (Tensor): The node tensor with shape [:obj:`number of samples`, :obj:`number of possible atom types`, :obj:`maximum number of atoms`].
        atomic_num_list (list): A list to specify what atom each channel of the 2nd dimension of :obj: `x` corresponds to.
        correct_validity (bool, optional): Whether to use the validity correction introduced by the paper `MoFlow: an invertible flow model for generating molecular graphs <https://arxiv.org/pdf/2006.10137.pdf>`_. (default: :obj:`True`)
        largest_connected_comp (bool, optional): Whether to use the largest connected component as the final molecule in the validity correction.(default: :obj:`True`)

    :rtype: A list of rdkit mol object. The length of the list is :obj:`number of samples`.

    Examples
    --------
    [<rdkit.Chem.rdchem.Mol>, <rdkit.Chem.rdchem.Mol>]
    """
    x = x.permute(0, 2, 1)
    adj = adj.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    gen_mols = []
    pure_valids = 0
    for x_elem, adj_elem in zip(x, adj):
        mol = construct_mol(x_elem, adj_elem,)
        cmol, pure = correct_mol(mol)
        pure_valids += pure
        gen_mols.append(cmol)

    return gen_mols, pure_valids