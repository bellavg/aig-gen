import torch
import unittest
from torch_geometric.data import Data


def create_masked_batch(batch, node_mask_prob=0.15):
    masked_batch = batch.clone()

    # Get node types (AND gates)
    is_and_gate = (batch.x[:, 0] == 0) & (batch.x[:, 1] == 1) & (batch.x[:, 2] == 0)

    # Initialize node mask (only for AND gates)
    node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=batch.x.device)

    # Get batch assignment for each node
    batch_idx = batch.batch if hasattr(batch, 'batch') else torch.zeros(batch.x.size(0), dtype=torch.long)

    # Iterate through each graph in the batch
    for b in torch.unique(batch_idx):
        # Get nodes for this graph
        graph_mask = batch_idx == b
        graph_nodes = torch.nonzero(graph_mask & is_and_gate).squeeze(-1)

        # Randomly select nodes to mask
        num_to_mask = max(1, int(len(graph_nodes) * node_mask_prob))
        if len(graph_nodes) > 0:
            masked_indices = graph_nodes[torch.randperm(len(graph_nodes))[:num_to_mask]]
            node_mask[masked_indices] = True

    # Store original values as targets
    masked_batch.x_target = batch.x.clone()
    masked_batch.edge_index_target = batch.edge_index.clone()
    masked_batch.edge_attr_target = batch.edge_attr.clone() if hasattr(batch, 'edge_attr') else None

    # Apply node masking
    if node_mask.sum() > 0:
        masked_batch.x[node_mask] = 0.0  # Zero out masked nodes

    # Store masks for loss computation
    masked_batch.node_mask = node_mask

    return masked_batch





def create_masked_aig_with_edges(aig, node_mask_prob=0.15):
    """
    Create a masked version of an AIG where masked AND gates have all their edges removed too

    Args:
        aig: PyG Data object representing an AIG
        node_mask_prob: Probability of masking each AND gate

    Returns:
        masked_aig: PyG Data object with masked nodes and their edges
    """
    # Create a copy to avoid modifying the original
    masked_aig = aig.clone()

    # Get node types (input, AND gate, output)
    # A mask that’s True exactly where the node is "AND":
    is_and_gate = (aig.x[:, 0] == 0) & (aig.x[:, 1] == 1) & (aig.x[:, 2] == 0)

    # Initialize node mask (only for AND gates)
    node_mask = torch.zeros(aig.x.size(0), dtype=torch.bool, device=aig.x.device)
    for node in range(aig.x.size(0)):
        if is_and_gate[node] and torch.rand(1).item() < node_mask_prob:
            node_mask[node] = True

    # Initialize edge mask based on node mask
    edge_mask = torch.zeros(aig.edge_index.size(1), dtype=torch.bool, device=aig.edge_index.device)

    # Mask edges connected to masked nodes
    for i in range(aig.edge_index.size(1)):
        src, dst = aig.edge_index[0, i], aig.edge_index[1, i]
        # If either source or destination is masked, mask the edge
        if node_mask[src] or node_mask[dst]:
            edge_mask[i] = True

    # Store original values as targets
    masked_aig.x_target = aig.x.clone()
    masked_aig.edge_index_target = aig.edge_index.clone()
    masked_aig.edge_attr_target = aig.edge_attr.clone() if hasattr(aig, 'edge_attr') else None

    # Apply node masking
    if node_mask.sum() > 0:
        masked_aig.x[node_mask] = 0.0  # Zero out masked nodes

    # Store masks for loss computation
    masked_aig.node_mask = node_mask
    masked_aig.edge_mask = edge_mask

    # Create edge subset without masked edges (for the forward pass)
    # We'll keep the original edges in edge_index_target for the loss computation
    if edge_mask.sum() > 0:
        kept_edges = ~edge_mask
        masked_aig.edge_index = aig.edge_index[:, kept_edges]

        if hasattr(aig, 'edge_attr') and aig.edge_attr is not None:
            masked_aig.edge_attr = aig.edge_attr[kept_edges]

    return masked_aig

