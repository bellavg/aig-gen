# G2PT/configs/aig.py
import math
import os # Added for potential path joining if needed
import networkx as nx
from typing import Union, Optional, List # Added for type hinting
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
MIN_AND_COUNT = 1 # Assuming at least one AND gate needed

PADDING_NODE_STR = "PADDING_NODE"
NUM_ADJ_CHANNELS = 3
Fan_ins = {"NODE_CONST0":0, "NODE_PI":0, "NODE_AND":2, "NODE_PO":1}

EXPLICIT_NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
EXPLICIT_EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]

ALL_NODE_KEYS = EXPLICIT_NODE_TYPE_KEYS + ["PADDING_NODE"]
ALL_EDGE_KEYS = EXPLICIT_EDGE_TYPE_KEYS + ["NONE"]

NUM2EDGETYPE = {0: ALL_EDGE_KEYS[0], 1: ALL_EDGE_KEYS[1], 2: ALL_EDGE_KEYS[2]}
NUM2NODETYPE = {0: ALL_NODE_KEYS[0], 1: ALL_NODE_KEYS[1], 2: ALL_NODE_KEYS[2], 3: ALL_NODE_KEYS[3], 4: ALL_NODE_KEYS[4]}

assert len(NUM2EDGETYPE) == len(ALL_EDGE_KEYS)
assert len(NUM2NODETYPE) == len(ALL_NODE_KEYS)

NODE_CONST0_STR = EXPLICIT_NODE_TYPE_KEYS[0]
NODE_PI_STR = EXPLICIT_NODE_TYPE_KEYS[1]
NODE_AND_STR = EXPLICIT_NODE_TYPE_KEYS[2]
NODE_PO_STR = EXPLICIT_NODE_TYPE_KEYS[3]
# Derive feature counts from the size of the derived vocabularies
NUM_EXPLICIT_NODE_FEATURES = len(EXPLICIT_NODE_TYPE_KEYS) # Should be 4
NUM_EXPLICIT_EDGE_FEATURES = len(EXPLICIT_EDGE_TYPE_KEYS) # Should be 2

NUM_NODE_ATTRIBUTES = NUM_EXPLICIT_NODE_FEATURES + 1 # for none type
NUM_EDGE_ATTRIBUTES = NUM_EXPLICIT_EDGE_FEATURES + 1 # for virtual no edge
NO_EDGE_CHANNEL = NUM_EXPLICIT_EDGE_FEATURES
PADDING_NODE_CHANNEL = NUM_EXPLICIT_NODE_FEATURES


# Define one-hot encodings (Keep these hardcoded as they define the feature representation)
NODE_TYPE_ENCODING_NX = {
    # Map "NODE_CONST0" to the first encoding vector (index 0), etc.
    "NODE_CONST0": [1.0, 0.0, 0.0, 0.0], # Index 0 feature
    "NODE_PI":     [0.0, 1.0, 0.0, 0.0], # Index 1 feature
    "NODE_AND":    [0.0, 0.0, 1.0, 0.0], # Index 2 feature
    "NODE_PO":     [0.0, 0.0, 0.0, 1.0]  # Index 3 feature
}

# IMPORTANT: Ensure keys/order match EDGE_TYPE_VOCAB derivation and NUM_EDGE_FEATURES
# The order here should ideally match the order in EDGE_TYPE_KEYS
EDGE_TYPE_ENCODING_NX = {
    "EDGE_REG": [1.0, 0.0],  # Index 0 feature
    "EDGE_INV": [0.0, 1.0]   # Index 1 feature
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
    if isinstance(graph, nx.DiGraph):
        return graph.copy()  # Return a copy to avoid modifying the original if it's passed around

    if isinstance(graph, nx.Graph):
        if graph.is_directed():
            warnings.warn("Input graph to to_directed_aig was nx.Graph but already directed. Returning a copy.")
            return graph.copy()

        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(graph.nodes(data=True))

        for u, v, data in graph.edges(data=True):
            edge_attrs = data.copy()
            if u == v:
                directed_graph.add_edge(u, v, **edge_attrs)
            elif u < v:
                directed_graph.add_edge(u, v, **edge_attrs)
            else:
                directed_graph.add_edge(v, u, **edge_attrs)
        return directed_graph

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

    print(f"\n--- Details for {name} ---")
    print(f"Type: {type(graph)}")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")

    print("\nNodes:")
    if not graph.nodes:
        print("  (No nodes in graph)")
    for node_id, attributes in graph.nodes(data=True):
        attr_str_list = []
        for key, value in attributes.items():
            if key == 'type' and isinstance(value, (list, tuple)) and DECODING_NODE_TYPE_NX:
                try:
                    type_tuple = tuple(float(x) for x in value) # Ensure float tuple for lookup
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
    for u, v, attributes in graph.edges(data=True):
        attr_str_list = []
        for key, value in attributes.items():
            if key == 'type' and isinstance(value, (list, tuple)) and DECODING_EDGE_TYPE_NX:
                try:
                    type_tuple = tuple(float(x) for x in value) # Ensure float tuple for lookup
                    decoded_type = DECODING_EDGE_TYPE_NX.get(type_tuple, f"Unknown encoding {value}")
                    attr_str_list.append(f"'type': '{decoded_type}' (raw: {value})")
                except (ValueError, TypeError):
                     attr_str_list.append(f"'type': {value} (raw, error decoding)")
            else:
                attr_str_list.append(f"'{key}': {value}")
        print(f"  Edge ({u} -> {v}): {{{', '.join(attr_str_list)}}}")
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

    # Work on a copy to avoid modifying the original graph passed to this function
    graph_copy = graph.copy()

    padding_nodes_to_remove: List[int] = []
    for node, data in graph_copy.nodes(data=True):
        if data.get('type') == PADDING_NODE_STR:
            padding_nodes_to_remove.append(node)

    graph_copy.remove_nodes_from(padding_nodes_to_remove)

    # After removing padding nodes, some previously non-isolated nodes might become isolated.
    # It's common practice to remove isolates from the "actual" graph part.
    # However, the original check_validity didn't remove isolates until evaluate_aigs.py.
    # For consistency with the previous structure, we'll leave isolate removal to a later stage
    # or make it a separate utility if needed by check_validity's consumers.
    # If isolates (other than CONST0) should make a graph invalid, that logic
    # would be in check_validity itself, operating on the graph_after_padding_removal.

    return graph_copy


def check_validity(graph_input: Union[nx.Graph, nx.DiGraph, None]) -> bool:
    """
    Checks the structural validity of an AIG after converting to directed,
    removing padding nodes, and then removing isolated (non-CONST0) nodes.

    Args:
        graph_input (Union[nx.Graph, nx.DiGraph, None]): The AIG graph to validate.
                                                       Node and edge 'type' attributes
                                                       are assumed to be strings.
                                                       None or an empty graph (after processing)
                                                       is considered valid.

    Returns:
        bool: True if the graph is valid according to AIG rules, False otherwise.
    """
    if not graph_input:  # Handles None or an empty container like an empty list/dict if passed by mistake
        # print("Debug: Validity Check - Input graph_input is None or empty container. Considered valid.")
        return True

    directed_graph = to_directed_aig(graph_input)
    if directed_graph is None:  # to_directed_aig might return None for invalid input types
        # print(f"Debug: Validity Check Failed - Input graph type '{type(graph_input)}' could not be processed into a DiGraph.")
        return False  # Or handle as per your requirements

    # Remove padding nodes first
    graph_no_padding = remove_padding_nodes(directed_graph)

    if graph_no_padding is None:  # Should ideally not happen if directed_graph was not None
        # print(f"Debug: Validity Check Failed - Graph became None after attempting to remove padding nodes.")
        return False

    # If graph is empty after removing padding nodes, it's valid.
    if not graph_no_padding.nodes():
        # print("Debug: Validity Check - Graph is empty after removing padding nodes. Considered valid.")
        return True

    # --- Remove isolated nodes (except NODE_CONST0) ---
    # Create a copy to modify if removing isolates, or work on graph_no_padding directly
    graph_to_validate = graph_no_padding.copy()

    # Iteratively remove isolated nodes until no more such nodes are found
    # This is because removing one isolated node might make another node isolated.
    while True:
        all_isolates = list(nx.isolates(graph_to_validate))
        if not all_isolates:  # No isolated nodes left
            break

        # Identify isolated nodes that are NOT _NODE_CONST0_STR
        # These are the ones we want to remove.
        undesired_isolates_to_remove = [
            node for node in all_isolates
            if graph_to_validate.nodes[node].get('type') != NODE_CONST0_STR
        ]

        if not undesired_isolates_to_remove:  # All remaining isolates are allowed (e.g. CONST0)
            break

        # print(f"Debug: Removing undesired isolated nodes: {undesired_isolates_to_remove}")
        graph_to_validate.remove_nodes_from(undesired_isolates_to_remove)

    # If the graph becomes empty after removing isolates (and padding), it's valid.
    if not graph_to_validate.nodes():
        # print("Debug: Validity Check - Graph is empty after removing padding and isolated nodes. Considered valid.")
        return True
    # --- End of isolated node removal ---

    # 1. Check DAG property (on graph_to_validate)
    if not nx.is_directed_acyclic_graph(graph_to_validate):
        # print(f"Debug: Validity Check Failed - Not a DAG. Nodes: {list(graph_to_validate.nodes())}")
        return False

    # 2. Check Node Types and Degrees
    for node, data in graph_to_validate.nodes(data=True):
        if 'type' not in data:
            # print(f"Debug: Validity Check Failed - Node {node} is missing 'type' attribute.")
            return False
        node_type = data['type']

        if node_type not in EXPLICIT_NODE_TYPE_KEYS:
            # print(f"Debug: Validity Check Failed - Node {node} has type '{node_type}' not in EXPLICIT_NODE_TYPE_KEYS.")
            return False

        in_degree = graph_to_validate.in_degree(node)
        out_degree = graph_to_validate.out_degree(node)

        if node_type == NODE_CONST0_STR:
            if in_degree != 0:
                # print(f"Debug: CONST0 node {node} in-degree violation: {in_degree}")
                return False
        elif node_type == NODE_PI_STR:
            if in_degree != 0:
                # print(f"Debug: PI node {node} in-degree violation: {in_degree}")
                return False
        elif node_type == NODE_AND_STR:
            # For AND gates, after all processing, in-degree should ideally be 2.
            # If partial generation is allowed, it could be < 2.
            # The original check was `in_degree > 2`. Sticking to that for now.
            # For a final validity check, you might want `in_degree != 2`.
            if in_degree > 2:
                # print(f"Debug: AND node {node} in-degree violation: {in_degree}")
                return False
        elif node_type == NODE_PO_STR:
            # For PO gates, after all processing, in-degree should ideally be 1.
            # Original check was `in_degree > 1`.
            # For a final validity check, you might want `in_degree != 1`.
            if in_degree > 1:  # Or in_degree == 0 if POs must be driven
                # print(f"Debug: PO node {node} in-degree violation: {in_degree}")
                return False
            if out_degree != 0:
                # print(f"Debug: PO node {node} out-degree violation: {out_degree}")
                return False

    # 3. Check Edge Types
    for u, v, data in graph_to_validate.edges(data=True):
        if 'type' not in data:
            # print(f"Debug: Validity Check Failed - Edge ({u}-{v}) is missing 'type' attribute.")
            return False
        edge_type = data['type']

        if edge_type not in EXPLICIT_EDGE_TYPE_KEYS:
            # print(f"Debug: Validity Check Failed - Edge ({u}-{v}) has type '{edge_type}' not in EXPLICIT_EDGE_TYPE_KEYS.")
            return False

    # Optional: Minimum component checks (e.g., must have PIs, POs, ANDs)
    # This was part of check_aig_component_minimums in your evaluate_aigs.py
    # You can add similar logic here if these are strict validity rules.
    # For example:
    # num_pi_nodes = sum(1 for _, data in graph_to_validate.nodes(data=True) if data.get('type') == _NODE_PI_STR)
    # num_po_nodes = sum(1 for _, data in graph_to_validate.nodes(data=True) if data.get('type') == _NODE_PO_STR)
    # if num_pi_nodes == 0 or num_po_nodes == 0:
    #     # print("Debug: Validity Check Failed - Missing PIs or POs.")
    #     return False

    return True


# --- Base Configuration Dictionary (Defaults) ---
base_conf = {
    "data_name": "aig",  # This will be overridden by args.dataset_name
    "model": {
        "max_size": MAX_NODE_COUNT, "node_dim": NUM_NODE_ATTRIBUTES, "bond_dim": NUM_ADJ_CHANNELS, "use_gpu": True,
        "edge_unroll": 25, "num_flow_layer": 12, "num_rgcn_layer": 3,
        "nhid": 128, "nout": 128,
        "deq_coeff": 0.9, "st_type": "exp", "use_df": False,
        "c": 0.05,  # Corresponds to former 'dequantization_scale_c'
        "ld_step": 60,  # Corresponds to former 'langevin_steps_k'
        "ld_noise": 0.005,  # Corresponds to former 'langevin_noise_sigma'
        "ld_step_size": 30.0,  # Corresponds to former 'langevin_step_size_lambda_half'
        "clamp": True,  # Corresponds to former 'langevin_clamp_grad' (user requested 'clamp')
        "alpha": 1.0,  # Corresponds to former 'loss_alpha_reg_weight'
        "hidden": 64

    },
    "lr": 0.001, "weight_decay": 1e-5, "batch_size": 32, "max_epochs": 30,
    "save_interval": 3, "grad_clip_value": 1.0,
}

