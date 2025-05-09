
### Code adapted from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow
import re
import numpy as np
import networkx as nx
from aig_config import *
#
# NUM2EDGETYPE = {
#     0: "EDGE_REG",  # Index 0 feature
#     1: "EDGE_INV",   # Index 1 feature
# }

#TODO figureout equivalent
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

bond_decoder_m = {1: "EDGE_REG", 2: "EDGE_INV", 3: "NONE"}
#bond_decoder_m = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}

# Define one-hot encodings (Keep these hardcoded as they define the feature representation)
NODE_TYPE_ENCODING_PYG = {
    # Map "NODE_CONST0" to the first encoding vector (index 0), etc.
    "NODE_CONST0": 0, # Index 0 feature
    "NODE_PI":     1, # Index 1 feature
    "NODE_AND":    2, # Index 2 feature
    "NODE_PO":     3,  # Index 3 feature,
    "VIRTUAL": 4
}


def construct_aig(x_features_matrix, adj_channel_tensor, node_type_list=NODE_TYPE_KEYS):
    """
    Constructs an AIG from node features and adjacency tensor.
    Stores edge scores for later use in correction.
    """
    aig = nx.DiGraph()
    # x_features_matrix has shape (max_nodes, num_actual_features + 1_virtual_channel)
    # e.g., (max_nodes, 5), where channel 4 is the virtual/padding type.

    node_type_indices_all = np.argmax(x_features_matrix, axis=1)

    # The index for the virtual/padding channel in the x_features_matrix.
    # This corresponds to GraphEBM's self.virtual_node_channel_idx, which is NUM_NODE_FEATURES.
    virtual_node_channel_matrix_idx = NUM_NODE_FEATURES  # This should be 4

    # Active nodes are those whose type is NOT the virtual/padding channel.
    active_node_mask = node_type_indices_all != virtual_node_channel_matrix_idx

    # Get the type indices only for the active nodes.
    # These indices should now be in the range [0, NUM_NODE_FEATURES - 1]
    active_node_assigned_type_indices = node_type_indices_all[active_node_mask]

    for compact_idx, assigned_type_idx in enumerate(active_node_assigned_type_indices):
        # Ensure assigned_type_idx is within bounds for node_type_list (NODE_TYPE_KEYS)
        if assigned_type_idx >= len(node_type_list):
            warnings.warn(
                f"Internal Error: assigned_type_idx ({assigned_type_idx}) is out of bounds for "
                f"node_type_list (len {len(node_type_list)}) even after filtering. "
                f"This indicates an issue with virtual channel handling or input tensor structure."
            )
            continue  # Skip adding this problematic node

        actual_node_type_str = node_type_list[assigned_type_idx]
        aig.add_node(compact_idx, type=actual_node_type_str)

    # --- The rest of the adjacency matrix processing logic remains the same ---
    # adj_channel_tensor shape: (num_edge_type_channels, max_nodes, max_nodes)
    # Get the index of the chosen edge type for each (source, target) pair
    adj_matrix_edge_indices = np.argmax(adj_channel_tensor, axis=0)
    # Get the actual score (e.g., logit or probability) for the chosen edge type
    adj_matrix_edge_scores = np.max(adj_channel_tensor, axis=0)

    # Filter for active nodes (using the same active_node_mask)
    # Ensure adj_matrix views are correctly indexed by active_node_mask if it's boolean
    if active_node_mask.any():  # Proceed only if there are active nodes
        num_active_nodes = active_node_mask.sum()
        # Create a mapping from original index (0 to max_nodes-1) to compact index (0 to num_active_nodes-1)
        original_indices_of_active_nodes = np.where(active_node_mask)[0]
        map_original_to_compact = {orig_idx: compact_idx for compact_idx, orig_idx in
                                   enumerate(original_indices_of_active_nodes)}

        for start_node_orig_idx in original_indices_of_active_nodes:
            for end_node_orig_idx in original_indices_of_active_nodes:
                edge_type_idx_in_adj = adj_matrix_edge_indices[start_node_orig_idx, end_node_orig_idx]

                # Check if it's an actual edge (not "NONE" if "NONE" is a channel in adj_channel_tensor)
                # bond_decoder_m is {1: "EDGE_REG", 2: "EDGE_INV", 3: "NONE"}
                # Assuming adj_matrix_edge_indices gives 0, 1, 2 corresponding to these.
                # And that adj_channel_tensor has 3 channels.
                # VIRTUAL_EDGE_INDEX in aig_config is 2 (for "NONE")

                # If your adj_matrix_edge_indices directly gives 0 for REG, 1 for INV, and 2 for NONE (from argmax over 3 channels)
                if edge_type_idx_in_adj != VIRTUAL_EDGE_INDEX:  # VIRTUAL_EDGE_INDEX is 2
                    edge_score = adj_matrix_edge_scores[start_node_orig_idx, end_node_orig_idx]

                    # Map original indices to compact indices for adding edges to 'aig'
                    start_node_compact_idx = map_original_to_compact[start_node_orig_idx]
                    end_node_compact_idx = map_original_to_compact[end_node_orig_idx]

                    # Assuming edge_type_idx_in_adj (0 or 1) can be used directly with NUM2EDGETYPE from aig_config
                    # NUM2EDGETYPE = {0: "EDGE_REG", 1: "EDGE_INV"}
                    # However, bond_decoder_m is {1: "EDGE_REG", 2: "EDGE_INV", 3: "NONE"}
                    # This part needs careful alignment.
                    # If adj_matrix_edge_indices is 0-indexed for actual edge types (REG=0, INV=1)
                    if edge_type_idx_in_adj in NUM2EDGETYPE:
                        actual_edge_type_str = NUM2EDGETYPE[edge_type_idx_in_adj]
                    else:
                        warnings.warn(f"Unexpected edge_type_idx_in_adj: {edge_type_idx_in_adj} for an active edge.")
                        continue

                    aig.add_edge(
                        int(start_node_compact_idx),
                        int(end_node_compact_idx),
                        type=actual_edge_type_str,
                        score=float(edge_score)
                    )
    return aig


Fan_ins = { "NODE_PO":1, "NODE_AND":2, "NODE_PI":0, "NODE_CONST0": 0 }

# prev correct_mol
def correct_fanins(aig):
    if check_validity(aig):
        return aig, 1
    for node_id, node_data in aig.nodes(data=True):
        node_type = node_data.get('type')
        if len(list(aig.predecessors(node_id))) > Fan_ins[node_type]:
            incoming_edges_with_scores = []
            for u, v, edge_attributes in aig.in_edges(node_id, data=True):
                score = edge_attributes.get('score', float('-inf'))  # float('inf') for missing scores (least preferred)
                incoming_edges_with_scores.append(((u, v), score))
            incoming_edges_with_scores.sort(key=lambda x: x[1], reverse=True)

            for incoming_edge, _ in incoming_edges_with_scores[Fan_ins[node_type]:]:
                aig.remove_edge(incoming_edge[0], incoming_edge[1])
        if node_type == "NODE_PO":
            successors_to_remove = list(aig.successors(node_id))
            if successors_to_remove:
                for target in successors_to_remove:
                    aig.remove_edge(node_id, target)

    if check_validity(aig):
        return aig, 0
    else:
        print("Warning: issues everywhere still wtf")
    return None, 0


def valid_aig_can_with_seg(aig: nx.DiGraph):
    """
    Validates an AIG (checks for DAG property), optionally canonicalizes node labels
    if it's a DAG, and extracts the largest connected component.

    Args:
        aig (AIG): The AIG object to process.
        largest_connected_comp (bool, optional): Whether to only keep the largest
                                                 weakly connected component. Defaults to True.

    Returns:
        Optional[AIG]: The processed AIG object, or None if the input is None or
                       becomes invalid (e.g., not a DAG and handling chooses to return None).
    """
    if aig is None:
        return None

    if not isinstance(aig, nx.DiGraph):  # Ensure it's our AIG class for the description attribute
        pass

    # 1. Validity Check: Must be a Directed Acyclic Graph (DAG) for combinational AIGs
    if not nx.is_directed_acyclic_graph(aig):
        return aig

    else:
        # This re-labels nodes 0 to N-1 according to topological sort.
        # Note: This creates a new graph object. We need to wrap it in AIG again.
        topo_sorted_nodes = list(nx.topological_sort(aig))
        mapping = {old_label: new_label for new_label, old_label in enumerate(topo_sorted_nodes)}

        # Create a new AIG instance for the relabeled graph
        aig = nx.relabel_nodes(aig, mapping, copy=True)


    if aig.number_of_nodes() > 1:
        aig.remove_nodes_from(list(nx.isolates(aig)))

    return aig




def gen_mol_from_one_shot_tensor(adj, x, largest_connected_comp=True):
    r"""Construct molecules from the node tensors and adjacency tensors generated by one-shot molecular graph generation methods.

    Args:
        adj (Tensor): The adjacency tensor with shape [:obj:`number of samples`, :obj:`number of possible bond types`, :obj:`maximum number of atoms`, :obj:`maximum number of atoms`].
        x (Tensor): The node tensor with shape [:obj:`number of samples`, :obj:`number of possible atom types`, :obj:`maximum number of atoms`].
        atomic_num_list (list): A list to specify what atom each channel of the 2nd dimension of :obj: `x` corresponds to.
        correct_validity (bool, optional): Whether to use the validity correction introduced by the paper `MoFlow: an invertible flow model for generating molecular graphs <https://arxiv.org/pdf/2006.10137.pdf>`_. (default: :obj:`True`)
        largest_connected_comp (bool, optional): Whether to use the largest connected component as the final molecule in the validity correction.(default: :obj:`True`)

    :rtype: A list of rdkit mol object. The length of the list is :obj:`number of samples`.

    Examples
    # --------
    # >>> adj = torch.rand(2, 4, 38, 38)
    # >>> x = torch.rand(2, 10, 38)
    # >>> atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0] [0,1,2,3]? for my case?
    # >>> gen_mols = gen_mol_from_one_shot_tensor(adj, x, atomic_num_list)
    # >>> gen_mols
    [<rdkit.Chem.rdchem.Mol>, <rdkit.Chem.rdchem.Mol>]
    """
    x = x.permute(0, 2, 1) # Now B, N, Atom Types + 1
    adj = adj.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    # if not correct_validity:
    #     gen_mols = [construct_mol(x_elem, adj_elem, atomic_num_list) for x_elem, adj_elem in zip(x, adj)]
    # else:
    gen_graphs = []
    # TODO I think atomic num list should be 0, 1, 2, 3, 4 where 4 is virtual?
    uncorrected_graphs = 0
    for x_elem, adj_elem in zip(x, adj):
        aig = construct_aig(x_elem, adj_elem)
        aig, pure = correct_fanins(aig)
        uncorrected_graphs += pure
        aig = valid_aig_can_with_seg(aig)
        gen_graphs.append(aig)

    return gen_graphs, uncorrected_graphs

