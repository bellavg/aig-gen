import os
import torch
import warnings
import os.path as osp
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.separate import separate


# Add this function to GraphDF/aig_dataset.py
import random
from collections import deque
import networkx as nx
import numpy as np
import torch
import warnings

def custom_randomized_topological_sort(G, random_generator):
    """
    Performs a topological sort, randomizing the order of nodes
    that have the same in-degree at each step.
    Uses the provided random_generator instance.
    Raises NetworkXUnfeasible if a cycle is detected.
    """
    if not G.is_directed():
        raise nx.NetworkXError("Topological sort not defined for undirected graphs.")

    in_degree_map = {node: degree for node, degree in G.in_degree()}
    # Nodes with in-degree 0 are the starting points
    zero_in_degree_nodes = [node for node, degree in in_degree_map.items() if degree == 0]

    # Shuffle the initial zero-degree nodes if augmentation is desired
    if len(zero_in_degree_nodes) > 1:
        random_generator.shuffle(zero_in_degree_nodes) # Use the passed generator

    queue = deque(zero_in_degree_nodes)
    result_order = []

    while queue:
        u = queue.popleft()
        result_order.append(u)

        # Find successors whose in-degree will become zero
        newly_zero_in_degree = []
        # Sort successors for deterministic iteration before potential shuffle
        for v in sorted(list(G.successors(u))):
            in_degree_map[v] -= 1
            if in_degree_map[v] == 0:
                newly_zero_in_degree.append(v)
            elif in_degree_map[v] < 0:
                # This indicates an issue in the graph structure or algorithm
                raise RuntimeError(f"In-degree became negative for node {v} during topological sort.")

        # Shuffle the newly zero-in-degree nodes for augmentation
        if len(newly_zero_in_degree) > 1:
            random_generator.shuffle(newly_zero_in_degree) # Use the passed generator

        # Add newly discovered zero-in-degree nodes to the queue
        for node in newly_zero_in_degree:
            queue.append(node)

    # Check if all nodes were included (if not, there's a cycle)
    if len(result_order) != G.number_of_nodes():
        # Raise the specific error NetworkX uses for cycles in topological sort
        raise nx.NetworkXUnfeasible(f"Graph contains a cycle. Topological sort cannot proceed.")

    return result_order

