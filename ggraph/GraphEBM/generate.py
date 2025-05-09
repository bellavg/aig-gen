
### Code adapted from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow
import re
import numpy as np
from rdkit import Chem
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
    node_type_indices_all = np.argmax(x_features_matrix, axis=1)
    pad_node_type_index = len(node_type_list) - 1
    active_node_mask = node_type_indices_all != pad_node_type_index
    active_node_assigned_type_indices = node_type_indices_all[active_node_mask]

    for compact_idx, assigned_type_idx in enumerate(active_node_assigned_type_indices):
        actual_node_type_str = node_type_list[assigned_type_idx]
        aig.add_node(compact_idx, type=actual_node_type_str)

    # adj_channel_tensor shape: (num_edge_type_channels, max_nodes, max_nodes)
    # Get the index of the chosen edge type for each (source, target) pair
    adj_matrix_edge_indices = np.argmax(adj_channel_tensor, axis=0)
    # Get the actual score (e.g., logit or probability) for the chosen edge type
    adj_matrix_edge_scores = np.max(adj_channel_tensor, axis=0)

    # Filter for active nodes
    filtered_adj_matrix_edge_indices = adj_matrix_edge_indices[active_node_mask, :][:, active_node_mask]
    filtered_adj_matrix_edge_scores = adj_matrix_edge_scores[active_node_mask, :][:, active_node_mask]

    for start_node_compact_idx, end_node_compact_idx in zip(*np.nonzero(filtered_adj_matrix_edge_indices)):
        edge_type_idx = filtered_adj_matrix_edge_indices[start_node_compact_idx, end_node_compact_idx]
        edge_score = filtered_adj_matrix_edge_scores[start_node_compact_idx, end_node_compact_idx]

        actual_edge_type_str = bond_decoder_m[edge_type_idx]
        aig.add_edge(
            int(start_node_compact_idx),
            int(end_node_compact_idx),
            type=actual_edge_type_str,
            score=float(edge_score)  # Store the score as an edge attribute
        )
    return aig


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence
Fan_ins = { "NODE_PO":1, "NODE_AND":2}

# prev correct_mol
def correct_fanins(aig):
    if check_valency(aig):
        return aig
    for node_id, node_data in aig.nodes(data=True):
        node_type = node_data.get('type')
        if node_type == "NODE_CONST0" or node_type == "NODE_PI":
            if any(aig.predecessors(node_id)):
                for source in list(aig.predecessors(node_id)):
                    aig.remove_edge(source, node_id)
        elif node_type == "NODE_PO":
            successors_to_remove = list(aig.successors(node_id))
            if successors_to_remove:
                for target in successors_to_remove:
                    aig.remove_edge(node_id, target)

        if len(list(aig.predecessors(node_id))) > Fan_ins[node_type]:
                incoming_edges_with_scores = []
                for source, data in aig.predecessors(node_id):
                    score = data.get('score', float('-inf'))  # float('inf') for missing scores (least preferred)
                    incoming_edges_with_scores.append(((source, node_id), score))
                incoming_edges_with_scores.sort(key=lambda x: x[1])
                for incoming_edge, score in incoming_edges_with_scores[Fan_ins[node_type]:]:
                    aig.remove_edge(incoming_edge[0], incoming_edge[1])

    return aig






def valid_mol_can_with_seg(x, largest_connected_comp=True):
    # mol = None
    if x is None:
        return None
    sm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol


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
    for x_elem, adj_elem in zip(x, adj):
        aig = construct_aig(x_elem, adj_elem)
        cmol = correct_fanins(aig)
        vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=largest_connected_comp)
        gen_graphs.append(vcmol)

    return gen_graphs, 0
