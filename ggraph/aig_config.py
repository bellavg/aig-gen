# G2PT/configs/aig.py
import math
import os  # Added for potential path joining if needed
import networkx as nx
from typing import Union, Optional, List, Tuple  # Added Tuple for type hinting
import warnings

# --- Primary Configuration Constants ---
dataset = 'aig'

# Consider using os.path.join for better path handling
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Gets G2PT directory
# data_dir = os.path.join(base_dir, 'datasets', 'aig')
# tokenizer_path = os.path.join(base_dir, 'tokenizers', 'aig')


# AIG constraint constants
MAX_NODE_COUNT = 64
MIN_PI_COUNT = 2
MAX_PI_COUNT = 8
MIN_PO_COUNT = 1
MAX_PO_COUNT = 8
MIN_AND_COUNT = 1  # Assuming at least one AND gate needed

PADDING_NODE_STR = "PADDING_NODE"
NUM_ADJ_CHANNELS = 3  # This refers to the number of explicit edge types + no-edge type for adjacency matrix
Fan_ins = {"NODE_CONST0": 0, "NODE_PI": 0, "NODE_AND": 2, "NODE_PO": 1}

EXPLICIT_NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
EXPLICIT_EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]  # Regular and Inverted edges

ALL_NODE_KEYS = EXPLICIT_NODE_TYPE_KEYS + [PADDING_NODE_STR]  # All node types including padding
ALL_EDGE_KEYS = EXPLICIT_EDGE_TYPE_KEYS + ["NONE"]  # All edge types including no-edge

NUM2EDGETYPE = {i: k for i, k in enumerate(ALL_EDGE_KEYS)}
NUM2NODETYPE = {i: k for i, k in enumerate(ALL_NODE_KEYS)}

assert len(NUM2EDGETYPE) == len(ALL_EDGE_KEYS)
assert len(NUM2NODETYPE) == len(ALL_NODE_KEYS)

NODE_CONST0_STR = EXPLICIT_NODE_TYPE_KEYS[0]
NODE_PI_STR = EXPLICIT_NODE_TYPE_KEYS[1]
NODE_AND_STR = EXPLICIT_NODE_TYPE_KEYS[2]
NODE_PO_STR = EXPLICIT_NODE_TYPE_KEYS[3]

# Derive feature counts from the size of the derived vocabularies
NUM_EXPLICIT_NODE_FEATURES = len(EXPLICIT_NODE_TYPE_KEYS)  # Should be 4 (CONST0, PI, AND, PO)
NUM_EXPLICIT_EDGE_FEATURES = len(EXPLICIT_EDGE_TYPE_KEYS)  # Should be 2 (REG, INV)

# Total number of channels/attributes for nodes and edges in the model's expected input format
NUM_NODE_ATTRIBUTES = NUM_EXPLICIT_NODE_FEATURES + 1  # Add 1 for PADDING_NODE type
NUM_EDGE_ATTRIBUTES = NUM_EXPLICIT_EDGE_FEATURES + 1  # Add 1 for "NONE" or "NO_EDGE" type

# Indices for the special channels
NO_EDGE_CHANNEL = NUM_EXPLICIT_EDGE_FEATURES  # Index of the "NONE" edge type (e.g., if 2 explicit types, this is index 2)
PADDING_NODE_CHANNEL = NUM_EXPLICIT_NODE_FEATURES  # Index of the PADDING_NODE type (e.g., if 4 explicit types, this is index 4)

# Define one-hot encodings (Keep these hardcoded as they define the feature representation)
NODE_TYPE_ENCODING_NX = {
    # Map "NODE_CONST0" to the first encoding vector (index 0), etc.
    NODE_CONST0_STR: [1.0, 0.0, 0.0, 0.0],  # Index 0 feature
    NODE_PI_STR: [0.0, 1.0, 0.0, 0.0],  # Index 1 feature
    NODE_AND_STR: [0.0, 0.0, 1.0, 0.0],  # Index 2 feature
    NODE_PO_STR: [0.0, 0.0, 0.0, 1.0]  # Index 3 feature
}

# IMPORTANT: Ensure keys/order match EXPLICIT_EDGE_TYPE_KEYS and NUM_EXPLICIT_EDGE_FEATURES
EDGE_TYPE_ENCODING_NX = {
    EXPLICIT_EDGE_TYPE_KEYS[0]: [1.0, 0.0],  # Index 0 feature (e.g., "EDGE_REG")
    EXPLICIT_EDGE_TYPE_KEYS[1]: [0.0, 1.0]  # Index 1 feature (e.g., "EDGE_INV")
}

DECODING_NODE_TYPE_NX = {
    tuple(v): k for k, v in NODE_TYPE_ENCODING_NX.items()
}

DECODING_EDGE_TYPE_NX = {
    tuple(v): k for k, v in EDGE_TYPE_ENCODING_NX.items()
}


def to_directed_aig(graph: Union[nx.Graph, nx.DiGraph]) -> Optional[nx.DiGraph]:
    """
    Converts an input graph to a directed AIG representation.
    - If the input is already a DiGraph, it's returned as is (a shallow copy for safety if modified later).
    - If the input is an undirected Graph, it's converted to a DiGraph
      with edges pointing from the node with the smaller ID to the node
      with the larger ID. Node and edge attributes are preserved.
    - If the input is not a NetworkX Graph object, None is returned.

    Args:
        graph (Union[nx.Graph, nx.DiGraph]): The input graph.

    Returns:
        Optional[nx.DiGraph]: The directed graph, or None if conversion is not possible.
    """
    if graph is None:  # Added check for None input
        warnings.warn("Input to to_directed_aig was None. Returning None.")
        return None

    if isinstance(graph, nx.DiGraph):
        return graph.copy()  # Return a copy to avoid modifying the original if it's passed around

    if isinstance(graph, nx.Graph):
        if graph.is_directed():  # Should not happen if it's truly nx.Graph, but good check
            warnings.warn("Input graph to to_directed_aig was nx.Graph but already directed. Returning a copy.")
            return graph.copy()

        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(graph.nodes(data=True))

        for u, v, data in graph.edges(data=True):
            edge_attrs = data.copy()
            if u == v:  # Self-loops
                directed_graph.add_edge(u, v, **edge_attrs)
            # For AIGs, typically we expect u < v for feed-forward nature after topological sort or ID assignment
            # However, if IDs are arbitrary, this ensures a consistent direction.
            # If your node IDs already imply direction (e.g., from a topological sort),
            # this might need adjustment or a different conversion strategy.
            # For now, assuming smaller ID to larger ID for undirected conversion.
            elif u < v:
                directed_graph.add_edge(u, v, **edge_attrs)
            else:
                directed_graph.add_edge(v, u, **edge_attrs)
        return directed_graph

    warnings.warn(f"Input to to_directed_aig was not a NetworkX graph (type: {type(graph)}). Returning None.")
    return None


def display_graph_details(graph: nx.Graph, name: str = "Graph"):
    """
    Prints a detailed view of a NetworkX graph, including node and edge attributes.

    Args:
        graph: The NetworkX graph object (nx.Graph or nx.DiGraph).
        name (str): A name for the graph for display purposes.
    """
    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        print(f"'{name}' is not a valid NetworkX graph object (type: {type(graph)}).")
        return

    print(f"\n--- Details for {name} (Memory ID: {id(graph)}) ---")
    print(f"Type: {type(graph)}")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Is directed: {graph.is_directed()}")

    print("\nNodes:")
    if not graph.nodes:
        print("  (No nodes in graph)")
    for node_id, attributes in sorted(graph.nodes(data=True)):  # Sort for consistent output
        attr_str_list = []
        for key, value in attributes.items():
            if key == 'type' and isinstance(value, (list, tuple)) and DECODING_NODE_TYPE_NX:
                try:
                    type_tuple = tuple(float(x) for x in value)  # Ensure float tuple for lookup
                    decoded_type = DECODING_NODE_TYPE_NX.get(type_tuple, f"Unknown encoding {value}")
                    attr_str_list.append(f"'type': '{decoded_type}' (raw: {value})")
                except (ValueError, TypeError):
                    attr_str_list.append(f"'type': {value} (raw, error decoding)")
            else:
                attr_str_list.append(f"'{key}': {value}")
        print(f"  Node {node_id}: {{{', '.join(attr_str_list)}}}")

    print("\nEdges:")
    if not graph.edges:
        print("  (No edges in graph)")
    # Sort edges for consistent output, especially if graph is undirected initially
    sorted_edges = sorted(list(graph.edges(data=True)))
    for u, v, attributes in sorted_edges:
        attr_str_list = []
        for key, value in attributes.items():
            if key == 'type' and isinstance(value, (list, tuple)) and DECODING_EDGE_TYPE_NX:
                try:
                    type_tuple = tuple(float(x) for x in value)  # Ensure float tuple for lookup
                    decoded_type = DECODING_EDGE_TYPE_NX.get(type_tuple, f"Unknown encoding {value}")
                    attr_str_list.append(f"'type': '{decoded_type}' (raw: {value})")
                except (ValueError, TypeError):
                    attr_str_list.append(f"'type': {value} (raw, error decoding)")
            else:
                attr_str_list.append(f"'{key}': {value}")
        # For directed graphs, u -> v. For undirected, (u,v) might be equivalent to (v,u) in listing.
        edge_repr = f"({u} -> {v})" if graph.is_directed() else f"({u} -- {v})"
        print(f"  Edge {edge_repr}: {{{', '.join(attr_str_list)}}}")
    print(f"--- End of Details for {name} ---\n")


def remove_padding_nodes(graph: Optional[nx.DiGraph]) -> Optional[nx.DiGraph]:
    """
    Removes nodes explicitly typed as "PADDING_NODE" from a directed graph.

    Args:
        graph (Optional[nx.DiGraph]): The input directed graph. Can be None.

    Returns:
        Optional[nx.DiGraph]: A new graph with padding nodes removed, or None if input was None.
                              Returns an empty DiGraph if the input graph becomes empty after removal.
    """
    if graph is None:
        return None

    graph_copy = graph.copy()
    padding_nodes_to_remove: List[int] = []
    for node, data in graph_copy.nodes(data=True):
        if data.get('type') == PADDING_NODE_STR:
            padding_nodes_to_remove.append(node)
        elif isinstance(data.get('type'), (list, tuple)):
            try:
                if len(data['type']) == NUM_NODE_ATTRIBUTES and data['type'][PADDING_NODE_CHANNEL] == 1.0:
                    padding_nodes_to_remove.append(node)
            except:
                pass
    if padding_nodes_to_remove:
        graph_copy.remove_nodes_from(padding_nodes_to_remove)
    return graph_copy


def check_interim_validity(graph_with_candidate_edge: nx.Graph,
                           source_of_candidate_edge: int,
                           target_of_candidate_edge: int) -> bool:
    """
    Checks if adding a specific candidate edge would immediately violate local AIG fan-in/fan-out rules
    for the two nodes involved in that edge. This is a lightweight check for intermediate generation steps.

    Args:
        graph_with_candidate_edge (nx.Graph): The graph object (can be nx.Graph or nx.DiGraph)
                                              *after* the candidate edge (source, target) has been added.
        source_of_candidate_edge (int): The source node ID of the newly added candidate edge.
        target_of_candidate_edge (int): The target node ID of the newly added candidate edge.

    Returns:
        bool: True if the new edge is locally valid, False otherwise.
    """
    if not graph_with_candidate_edge:
        return False  # Should not happen if called correctly

    # 1. Convert to directed graph to correctly assess in-degrees and out-degrees
    #    This operates on the graph that *already includes* the candidate edge.
    directed_graph = to_directed_aig(graph_with_candidate_edge)

    if directed_graph is None:
        warnings.warn("check_interim_validity: Failed to convert graph to directed. Assuming invalid.")
        return False

    # Ensure nodes exist in the directed graph (they should if to_directed_aig worked)
    if not directed_graph.has_node(target_of_candidate_edge) or \
            not directed_graph.has_node(source_of_candidate_edge):
        warnings.warn(
            f"check_interim_validity: Candidate edge nodes ({source_of_candidate_edge}, {target_of_candidate_edge}) not in directed graph. Assuming invalid.")
        return False

    # 2. Get node types
    try:
        target_node_attrs = directed_graph.nodes[target_of_candidate_edge]
        source_node_attrs = directed_graph.nodes[source_of_candidate_edge]
        target_node_type = target_node_attrs.get('type')
        source_node_type = source_node_attrs.get('type')
    except KeyError:
        warnings.warn(
            f"check_interim_validity: Node attributes not found for edge ({source_of_candidate_edge}->{target_of_candidate_edge}). Assuming invalid.")
        return False  # Node not found, should not happen

    if target_node_type is None or source_node_type is None:
        # This implies a node is missing its 'type' attribute, which is an issue.
        warnings.warn(
            f"check_interim_validity: Node type missing for {target_of_candidate_edge} or {source_of_candidate_edge}. Assuming invalid.")
        return False

    # 3. Check Target Node (target_of_candidate_edge) for fan-in violations
    #    The in_degree is checked *after* the candidate edge has been added.
    current_in_degree_target = directed_graph.in_degree(target_of_candidate_edge)

    if target_node_type == NODE_PI_STR or target_node_type == NODE_CONST0_STR:
        # PIs and CONST0 nodes must have an in-degree of 0.
        # If current_in_degree_target > 0, it means the candidate edge (or another) made it non-zero.
        if current_in_degree_target > 0:
            # print(f"Debug Interim: Target {target_of_candidate_edge} ({target_node_type}) has in-degree {current_in_degree_target} > 0. Invalid.")
            return False

    elif target_node_type == NODE_AND_STR:
        # AND nodes must not have in-degree > Fan_ins[NODE_AND_STR] (typically 2)
        if current_in_degree_target > Fan_ins[NODE_AND_STR]:
            # print(f"Debug Interim: Target {target_of_candidate_edge} ({target_node_type}) has in-degree {current_in_degree_target} > {Fan_ins[NODE_AND_STR]}. Invalid.")
            return False

    elif target_node_type == NODE_PO_STR:
        # PO nodes must not have in-degree > Fan_ins[NODE_PO_STR] (typically 1)
        if current_in_degree_target > Fan_ins[NODE_PO_STR]:
            # print(f"Debug Interim: Target {target_of_candidate_edge} ({target_node_type}) has in-degree {current_in_degree_target} > {Fan_ins[NODE_PO_STR]}. Invalid.")
            return False

    # 4. Check Source Node (source_of_candidate_edge) for fan-out violations
    #    Specifically, a PO node cannot have outgoing edges.
    #    The out_degree is checked *after* the candidate edge has been added.
    if source_node_type == NODE_PO_STR:
        current_out_degree_source = directed_graph.out_degree(source_of_candidate_edge)
        if current_out_degree_source > 0:
            # print(f"Debug Interim: Source {source_of_candidate_edge} ({source_node_type}) has out-degree {current_out_degree_source} > 0. Invalid.")
            return False

    # 5. Check if the graph (with the new edge) is still a DAG. This is a crucial local property.
    #    If adding the edge creates a cycle involving the new edge, it's invalid.
    if not nx.is_directed_acyclic_graph(directed_graph):
        # To be more specific, we can check if the cycle involves the new edge, but a general DAG check here is often sufficient.
        # For a more precise check:
        # try:
        #     cycle = nx.find_cycle(directed_graph, source=target_of_candidate_edge) # Check if target can reach source
        #     if cycle: # A cycle exists
        #          # Check if the cycle involves the reverse of the candidate edge or the source node
        #          # This logic can get complex; a general DAG check is simpler for now.
        #          # print(f"Debug Interim: Adding edge ({source_of_candidate_edge}->{target_of_candidate_edge}) creates a cycle. Invalid.")
        #          return False
        # except nx.NetworkXNoCycle:
        #     pass # No cycle found starting from target, good.
        # print(f"Debug Interim: Graph with edge ({source_of_candidate_edge}->{target_of_candidate_edge}) is not a DAG. Invalid.")
        return False

    return True


def check_validity(graph_input: Union[nx.Graph, nx.DiGraph, None], is_intermediate_check: bool = False) -> bool:
    """
    Checks the structural validity of an AIG.
    Converts to directed, removes padding nodes.
    If not an intermediate check, it also removes isolated (non-CONST0) nodes and checks min components.

    Args:
        graph_input: The AIG graph to validate.
        is_intermediate_check: If True, skips aggressive isolated node removal and min component checks.
    Returns:
        bool: True if the graph is valid, False otherwise.
    """
    if not graph_input:
        return True  # An empty or None graph can be considered valid in some contexts, or handle as error

    directed_graph = to_directed_aig(graph_input)
    if directed_graph is None:
        return False

    graph_no_padding = remove_padding_nodes(directed_graph)
    if graph_no_padding is None or not graph_no_padding.nodes():
        return True  # Empty after padding removal is valid

    graph_to_validate = graph_no_padding.copy()

    # Full DAG check (always important)
    if not nx.is_directed_acyclic_graph(graph_to_validate):
        return False

    # Node Types and Degrees (these are generally always important)
    pi_count = 0
    po_count = 0
    and_count = 0
    const0_count = 0

    for node, data in graph_to_validate.nodes(data=True):
        node_type = data.get('type')
        if node_type is None or node_type not in EXPLICIT_NODE_TYPE_KEYS:
            return False  # Invalid or missing node type

        if node_type == NODE_PI_STR:
            pi_count += 1
        elif node_type == NODE_AND_STR:
            and_count += 1
        elif node_type == NODE_PO_STR:
            po_count += 1
        elif node_type == NODE_CONST0_STR:
            const0_count += 1

        in_degree = graph_to_validate.in_degree(node)
        out_degree = graph_to_validate.out_degree(node)

        if node_type == NODE_PI_STR or node_type == NODE_CONST0_STR:
            if in_degree != 0: return False
        elif node_type == NODE_AND_STR:
            if in_degree > Fan_ins[NODE_AND_STR]: return False  # Cannot exceed fan-in
            if not is_intermediate_check and in_degree != Fan_ins[
                NODE_AND_STR]: return False  # Must be exactly fan-in for final
        elif node_type == NODE_PO_STR:
            if out_degree != 0: return False
            if in_degree > Fan_ins[NODE_PO_STR]: return False  # Cannot exceed fan-in
            if not is_intermediate_check and in_degree != Fan_ins[
                NODE_PO_STR]: return False  # Must be exactly fan-in for final

    # Edge Types (always important)
    for u, v, data in graph_to_validate.edges(data=True):
        edge_type = data.get('type')
        if edge_type is None or edge_type not in EXPLICIT_EDGE_TYPE_KEYS:
            return False

    # Checks for final validation (when is_intermediate_check is False)
    if not is_intermediate_check:
        # Minimum component counts
        if pi_count == 0 and const0_count == 0: return False  # Must have some input source
        if po_count < MIN_PO_COUNT: return False
        # if and_count < MIN_AND_COUNT: return False # Optional: if AND gates are strictly required

        # Check for isolated non-CONST0 nodes (only for final validation)
        # This check can be computationally intensive for large graphs if done often.
        all_isolates = list(nx.isolates(graph_to_validate))
        for isolated_node in all_isolates:
            if graph_to_validate.nodes[isolated_node].get('type') != NODE_CONST0_STR:
                return False  # An isolated PI, AND, or PO is invalid

    return True


# --- Base Configuration Dictionary (Defaults) ---
base_conf = {
    "data_name": "aig",
    "model": {
        "max_size": MAX_NODE_COUNT,
        "node_dim": NUM_NODE_ATTRIBUTES,
        "bond_dim": NUM_EDGE_ATTRIBUTES,
        "use_gpu": True,
        "edge_unroll": 25,
        "num_flow_layer": 12,
        "num_rgcn_layer": 3,
        "nhid": 128,
        "nout": 128,
        "deq_coeff": 0.9,
        "st_type": "exp",
        "use_df": False,

        "c": 0.05,
        "ld_step": 60,
        "ld_noise": 0.005,
        "ld_step_size": 30.0,
        "clamp": True,
        "alpha": 1.0,
        "hidden": 64
    },
    "lr": 0.001,
    "weight_decay": 0.0,
    "batch_size": 32,
    "max_epochs": 30,
    "save_interval": 3,
    "grad_clip_value": 1.0,
}
