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

_PADDING_NODE_STR = "PADDING_NODE"
NUM_ADJ_CHANNELS = 3

EXPLICIT_NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
EXPLICIT_EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]

ALL_NODE_KEYS = EXPLICIT_NODE_TYPE_KEYS + ["PADDING_NODE"]
ALL_EDGE_KEYS = EXPLICIT_EDGE_TYPE_KEYS + ["NONE"]

NUM2EDGETYPE = {0: ALL_EDGE_KEYS[0], 1: ALL_EDGE_KEYS[1], 2: ALL_EDGE_KEYS[2]}
NUM2NODETYPE = {0: ALL_NODE_KEYS[0], 1: ALL_NODE_KEYS[1], 2: ALL_NODE_KEYS[2], 3: ALL_NODE_KEYS[3], 4: ALL_NODE_KEYS[4]}

assert len(NUM2EDGETYPE) == len(ALL_EDGE_KEYS)
assert len(NUM2NODETYPE) == len(ALL_NODE_KEYS)


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
        if data.get('type') == _PADDING_NODE_STR:
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
    Checks the structural validity of an AIG after converting to directed
    and removing any padding nodes.

    Args:
        graph_input (Union[nx.Graph, nx.DiGraph, None]): The AIG graph to validate.
                                                       Node and edge 'type' attributes
                                                       are assumed to be strings.
                                                       None or an empty graph (after processing)
                                                       is considered valid.

    Returns:
        bool: True if the graph is valid according to AIG rules, False otherwise.
    """
    if not graph_input:
        return True  # None or an empty container is valid by this definition

    directed_graph = to_directed_aig(graph_input)
    if directed_graph is None:
        print(
            f"Debug: Validity Check Failed - Input graph type '{type(graph_input)}' could not be processed into a DiGraph.")
        return False

    # Remove padding nodes before further checks
    # The remove_padding_nodes function handles None input and returns a copy.
    graph_to_validate = remove_padding_nodes(directed_graph)

    if graph_to_validate is None:  # Should not happen if directed_graph was not None
        print(f"Debug: Validity Check Failed - Graph became None after attempting to remove padding nodes.")
        return False

    # If graph_to_validate is empty after removing padding nodes, it's valid.
    if not graph_to_validate.nodes():  # Check if there are any nodes left
        # print("Debug: Validity Check - Graph is empty after removing padding nodes. Considered valid.")
        return True

    # 1. Check DAG property
    if not nx.is_directed_acyclic_graph(graph_to_validate):
        # print(f"Debug: Validity Check Failed - Not a DAG after removing padding nodes. Graph: {list(graph_to_validate.nodes())}")
        return False

    # Access EXPLICIT_NODE_TYPE_KEYS and EXPLICIT_EDGE_TYPE_KEYS
    try:
        _NODE_CONST0_STR = EXPLICIT_NODE_TYPE_KEYS[0]
        _NODE_PI_STR = EXPLICIT_NODE_TYPE_KEYS[1]
        _NODE_AND_STR = EXPLICIT_NODE_TYPE_KEYS[2]
        _NODE_PO_STR = EXPLICIT_NODE_TYPE_KEYS[3]
    except NameError:  # Should be caught by global try-except if aig_config is missing
        print("Debug: Validity Check Failed - EXPLICIT_NODE_TYPE_KEYS is not defined (NameError).")
        return False
    except IndexError:
        print("Debug: Validity Check Failed - EXPLICIT_NODE_TYPE_KEYS does not contain enough elements.")
        return False

    # 2. Check Node Types and Degrees (on graph_to_validate)
    for node, data in graph_to_validate.nodes(data=True):
        if 'type' not in data:
            print(f"Debug: Validity Check Failed - Node {node} is missing 'type' attribute.")
            return False
        node_type = data['type']

        # After removing padding nodes, no node should have PADDING_NODE type.
        # The check below ensures it's one of the EXPLICIT types.
        if node_type == "UNKNOWN_TYPE_ATTRIBUTE":
            print(f"Debug: Validity Check Failed - Node {node} has explicit 'UNKNOWN_TYPE_ATTRIBUTE': {data}")
            return False
        if node_type not in EXPLICIT_NODE_TYPE_KEYS:  # This check is crucial.
            print(
                f"Debug: Validity Check Failed - Node {node} has type '{node_type}' which is not in EXPLICIT_NODE_TYPE_KEYS (and not PADDING_NODE, as they should be removed).")
            return False

        in_degree = graph_to_validate.in_degree(node)
        out_degree = graph_to_validate.out_degree(node)

        if node_type == _NODE_CONST0_STR:
            if in_degree != 0:
                return False
        elif node_type == _NODE_PI_STR:
            if in_degree != 0:
                return False
        elif node_type == _NODE_AND_STR:
            if in_degree > 2:  # Or use `in_degree != 2` if strictly two inputs are required
                return False
        elif node_type == _NODE_PO_STR:
            if in_degree > 1:  # Or use `in_degree != 1` if strictly one input is required
                return False
            if out_degree != 0:
                return False

    # 3. Check Edge Types (on graph_to_validate)
    try:
        _ = EXPLICIT_EDGE_TYPE_KEYS
    except NameError:  # Should be caught by global try-except
        print("Debug: Validity Check Failed - EXPLICIT_EDGE_TYPE_KEYS is not defined (NameError).")
        return False

    for u, v, data in graph_to_validate.edges(data=True):
        if 'type' not in data:
            print(f"Debug: Validity Check Failed - Edge ({u}-{v}) is missing 'type' attribute.")
            return False
        edge_type = data['type']

        if edge_type == "UNKNOWN_TYPE_ATTRIBUTE":
            return False
        if edge_type not in EXPLICIT_EDGE_TYPE_KEYS:
            return False

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

