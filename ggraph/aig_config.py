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
NUM_ADJ_CHANNELS = 3 # This refers to the number of explicit edge types + no-edge type for adjacency matrix
Fan_ins = {"NODE_CONST0":0, "NODE_PI":0, "NODE_AND":2, "NODE_PO":1}

EXPLICIT_NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
EXPLICIT_EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"] # Regular and Inverted edges

ALL_NODE_KEYS = EXPLICIT_NODE_TYPE_KEYS + [PADDING_NODE_STR] # All node types including padding
ALL_EDGE_KEYS = EXPLICIT_EDGE_TYPE_KEYS + ["NONE"] # All edge types including no-edge

NUM2EDGETYPE = {i: k for i, k in enumerate(ALL_EDGE_KEYS)}
NUM2NODETYPE = {i: k for i, k in enumerate(ALL_NODE_KEYS)}


assert len(NUM2EDGETYPE) == len(ALL_EDGE_KEYS)
assert len(NUM2NODETYPE) == len(ALL_NODE_KEYS)

NODE_CONST0_STR = EXPLICIT_NODE_TYPE_KEYS[0]
NODE_PI_STR = EXPLICIT_NODE_TYPE_KEYS[1]
NODE_AND_STR = EXPLICIT_NODE_TYPE_KEYS[2]
NODE_PO_STR = EXPLICIT_NODE_TYPE_KEYS[3]

# Derive feature counts from the size of the derived vocabularies
NUM_EXPLICIT_NODE_FEATURES = len(EXPLICIT_NODE_TYPE_KEYS) # Should be 4 (CONST0, PI, AND, PO)
NUM_EXPLICIT_EDGE_FEATURES = len(EXPLICIT_EDGE_TYPE_KEYS) # Should be 2 (REG, INV)

# Total number of channels/attributes for nodes and edges in the model's expected input format
NUM_NODE_ATTRIBUTES = NUM_EXPLICIT_NODE_FEATURES + 1 # Add 1 for PADDING_NODE type
NUM_EDGE_ATTRIBUTES = NUM_EXPLICIT_EDGE_FEATURES + 1 # Add 1 for "NONE" or "NO_EDGE" type

# Indices for the special channels
NO_EDGE_CHANNEL = NUM_EXPLICIT_EDGE_FEATURES # Index of the "NONE" edge type (e.g., if 2 explicit types, this is index 2)
PADDING_NODE_CHANNEL = NUM_EXPLICIT_NODE_FEATURES # Index of the PADDING_NODE type (e.g., if 4 explicit types, this is index 4)


# Define one-hot encodings (Keep these hardcoded as they define the feature representation)
NODE_TYPE_ENCODING_NX = {
    # Map "NODE_CONST0" to the first encoding vector (index 0), etc.
    NODE_CONST0_STR: [1.0, 0.0, 0.0, 0.0], # Index 0 feature
    NODE_PI_STR:     [0.0, 1.0, 0.0, 0.0], # Index 1 feature
    NODE_AND_STR:    [0.0, 0.0, 1.0, 0.0], # Index 2 feature
    NODE_PO_STR:     [0.0, 0.0, 0.0, 1.0]  # Index 3 feature
}

# IMPORTANT: Ensure keys/order match EXPLICIT_EDGE_TYPE_KEYS and NUM_EXPLICIT_EDGE_FEATURES
EDGE_TYPE_ENCODING_NX = {
    EXPLICIT_EDGE_TYPE_KEYS[0]: [1.0, 0.0],  # Index 0 feature (e.g., "EDGE_REG")
    EXPLICIT_EDGE_TYPE_KEYS[1]: [0.0, 1.0]   # Index 1 feature (e.g., "EDGE_INV")
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
        if graph.is_directed(): # Should not happen if it's truly nx.Graph, but good check
            warnings.warn("Input graph to to_directed_aig was nx.Graph but already directed. Returning a copy.")
            return graph.copy()

        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(graph.nodes(data=True))

        for u, v, data in graph.edges(data=True):
            edge_attrs = data.copy()
            if u == v: # Self-loops
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
    for node_id, attributes in sorted(graph.nodes(data=True)): # Sort for consistent output
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
    # Sort edges for consistent output, especially if graph is undirected initially
    sorted_edges = sorted(list(graph.edges(data=True)))
    for u, v, attributes in sorted_edges:
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

    # Work on a copy to avoid modifying the original graph passed to this function
    graph_copy = graph.copy()

    padding_nodes_to_remove: List[int] = []
    for node, data in graph_copy.nodes(data=True):
        # Check if 'type' attribute exists and if its value is PADDING_NODE_STR
        if data.get('type') == PADDING_NODE_STR:
            padding_nodes_to_remove.append(node)
        # Also handle cases where type might be a one-hot encoding corresponding to padding
        elif isinstance(data.get('type'), (list, tuple)):
            try:
                # Attempt to decode if it's a list/tuple (potentially one-hot)
                # This assumes PADDING_NODE_STR is not in DECODING_NODE_TYPE_NX keys
                # and that padding nodes are identified by a specific index if one-hot.
                # For safety, this part is better handled if padding nodes are always string-typed
                # or if there's a clear one-hot representation for padding.
                # The current PADDING_NODE_CHANNEL logic implies one-hot.
                # If type is one-hot [0,0,0,0,1] and PADDING_NODE_CHANNEL is 4 for 5 total types:
                if len(data['type']) == NUM_NODE_ATTRIBUTES and data['type'][PADDING_NODE_CHANNEL] == 1.0:
                     padding_nodes_to_remove.append(node)
            except: # Broad except if type is not as expected
                pass


    if padding_nodes_to_remove:
        # print(f"Debug: In remove_padding_nodes, removing: {padding_nodes_to_remove}")
        graph_copy.remove_nodes_from(padding_nodes_to_remove)

    return graph_copy


def check_validity(graph_input: Union[nx.Graph, nx.DiGraph, None], is_intermediate_check: bool = False) -> bool:
    """
    Checks the structural validity of an AIG.
    Converts to directed, removes padding nodes.
    If not an intermediate check, it also removes isolated (non-CONST0) nodes.

    Args:
        graph_input (Union[nx.Graph, nx.DiGraph, None]): The AIG graph to validate.
                                                       Node and edge 'type' attributes
                                                       are assumed to be strings.
                                                       None or an empty graph (after processing)
                                                       is considered valid.
        is_intermediate_check (bool): If True, skips the aggressive isolated node removal.
                                      Defaults to False (performs all checks).

    Returns:
        bool: True if the graph is valid according to AIG rules, False otherwise.
    """
    if not graph_input:
        # print("Debug: Validity Check - Input graph_input is None or empty container. Considered valid.")
        return True

    # Ensure we are working with a DiGraph.
    # The `to_directed_aig` function should handle the conversion from nx.Graph if needed.
    # It also returns a copy, so original graph_input is not modified here.
    directed_graph = to_directed_aig(graph_input)

    if directed_graph is None:
        # print(f"Debug: Validity Check Failed - Input graph type '{type(graph_input)}' could not be processed into a DiGraph.")
        return False

    # Remove padding nodes first. This also works on a copy.
    graph_no_padding = remove_padding_nodes(directed_graph)

    if graph_no_padding is None: # Should not happen if directed_graph was not None
        # print(f"Debug: Validity Check Failed - Graph became None after attempting to remove padding nodes.")
        return False

    # If graph is empty after removing padding nodes, it's valid.
    if not graph_no_padding.nodes():
        # print("Debug: Validity Check - Graph is empty after removing padding nodes. Considered valid.")
        return True

    # --- Isolated Node Removal (Conditional) ---
    # This section is now conditional based on `is_intermediate_check`
    # For the user's current request, we are commenting this part out directly.
    # If we were using the flag, it would be:
    # if not is_intermediate_check:
    #    # ... (isolated node removal logic) ...
    # else:
    #    graph_to_validate = graph_no_padding.copy() # Still work on a copy for subsequent checks

    graph_to_validate = graph_no_padding.copy() # Always work on a copy for the checks below

    # --- Start of commented out isolated node removal section ---
    # print(f"Debug: check_validity called with is_intermediate_check={is_intermediate_check}")
    # if not is_intermediate_check:
    #     print(f"Debug: Performing aggressive isolated node removal in check_validity.")
    #     # Iteratively remove isolated nodes (except NODE_CONST0_STR) until no more such nodes are found
    #     while True:
    #         all_isolates = list(nx.isolates(graph_to_validate))
    #         if not all_isolates:
    #             break
    #
    #         undesired_isolates_to_remove = [
    #             node for node in all_isolates
    #             if graph_to_validate.nodes[node].get('type') != NODE_CONST0_STR
    #         ]
    #
    #         if not undesired_isolates_to_remove:
    #             break
    #
    #         # print(f"Debug: In check_validity (full check), removing undesired isolated nodes: {undesired_isolates_to_remove} from graph with nodes: {list(graph_to_validate.nodes())}")
    #         graph_to_validate.remove_nodes_from(undesired_isolates_to_remove)
    #
    #     # If the graph becomes empty after removing isolates (and padding), it's valid.
    #     if not graph_to_validate.nodes():
    #         # print("Debug: Validity Check - Graph is empty after removing padding and all undesired isolated nodes. Considered valid.")
    #         return True
    # else:
    #     # print(f"Debug: Skipping aggressive isolated node removal in check_validity (is_intermediate_check=True).")
    #     pass
    # --- End of commented out isolated node removal section ---


    # 1. Check DAG property (on graph_to_validate)
    # Ensure graph_to_validate is a DiGraph, which it should be from to_directed_aig
    if not isinstance(graph_to_validate, nx.DiGraph):
         # This case should ideally be caught by to_directed_aig returning None earlier
        # print(f"Debug: Validity Check Failed - graph_to_validate is not a DiGraph (type: {type(graph_to_validate)}).")
        return False

    if not nx.is_directed_acyclic_graph(graph_to_validate):
        # print(f"Debug: Validity Check Failed - Not a DAG. Nodes: {list(graph_to_validate.nodes())}")
        # try:
        #     cycles = list(nx.simple_cycles(graph_to_validate))
        #     print(f"Cycles found: {cycles}")
        # except Exception as e:
        #     print(f"Could not find cycles: {e}")
        return False

    # 2. Check Node Types and Degrees
    pi_count = 0
    po_count = 0
    and_count = 0

    for node, data in graph_to_validate.nodes(data=True):
        if 'type' not in data:
            # print(f"Debug: Validity Check Failed - Node {node} is missing 'type' attribute.")
            return False
        node_type = data['type']

        if node_type not in EXPLICIT_NODE_TYPE_KEYS: # Check against only explicit (non-padding) types
            # print(f"Debug: Validity Check Failed - Node {node} has type '{node_type}' not in EXPLICIT_NODE_TYPE_KEYS.")
            return False

        # Increment counts for minimum component check later (optional, but good for AIGs)
        if node_type == NODE_PI_STR: pi_count +=1
        elif node_type == NODE_AND_STR: and_count +=1
        elif node_type == NODE_PO_STR: po_count +=1

        # Degree checks
        in_degree = graph_to_validate.in_degree(node)
        out_degree = graph_to_validate.out_degree(node)
        expected_fan_in = Fan_ins.get(node_type)

        if expected_fan_in is not None: # Check if fan-in is defined for this type
            if node_type == NODE_AND_STR:
                # For AND gates, in-degree should ideally be 2 for a complete gate.
                # During generation, it might be 0 or 1 temporarily.
                # The original check was `in_degree > 2`.
                # A stricter check for a *final* valid AIG might be `in_degree != 2`.
                # For intermediate checks, allowing <2 might be necessary.
                # Let's stick to the original intent for now which was to prevent over-connection.
                if in_degree > expected_fan_in:
                    # print(f"Debug: {node_type} node {node} in-degree violation: {in_degree} > {expected_fan_in}")
                    return False
            elif node_type == NODE_PO_STR:
                if in_degree > expected_fan_in: # PO fan-in should be 1.
                    # print(f"Debug: {node_type} node {node} in-degree violation: {in_degree} > {expected_fan_in}")
                    return False
                # PO specific check: out-degree must be 0
                if out_degree != 0:
                    # print(f"Debug: {node_type} node {node} out-degree violation: {out_degree} != 0")
                    return False
            else: # For CONST0 and PI
                if in_degree != expected_fan_in:
                    # print(f"Debug: {node_type} node {node} in-degree violation: {in_degree} != {expected_fan_in}")
                    return False
        # else: node_type has no specific fan-in rule from Fan_ins dict (should not happen for explicit types)


    # 3. Check Edge Types
    for u, v, data in graph_to_validate.edges(data=True):
        if 'type' not in data:
            # print(f"Debug: Validity Check Failed - Edge ({u}-{v}) is missing 'type' attribute.")
            return False
        edge_type = data['type']

        if edge_type not in EXPLICIT_EDGE_TYPE_KEYS:
            # print(f"Debug: Validity Check Failed - Edge ({u}-{v}) has type '{edge_type}' not in EXPLICIT_EDGE_TYPE_KEYS.")
            return False

    # 4. Optional: Minimum component checks (e.g., must have PIs, POs for a functional circuit)
    # These are typically for final graph validation, not intermediate.
    # if not is_intermediate_check:
    #     if pi_count < MIN_PI_COUNT :
    #         # print(f"Debug: Validity Check Failed - Not enough PIs. Found {pi_count}, Min required {MIN_PI_COUNT}.")
    #         return False
    #     if po_count < MIN_PO_COUNT:
    #         # print(f"Debug: Validity Check Failed - Not enough POs. Found {po_count}, Min required {MIN_PO_COUNT}.")
    #         return False
        # if and_count < MIN_AND_COUNT: # If even trivial pass-through circuits are disallowed
            # print(f"Debug: Validity Check Failed - Not enough ANDs. Found {and_count}, Min required {MIN_AND_COUNT}.")
            # return False

    return True


# --- Base Configuration Dictionary (Defaults) ---
base_conf = {
    "data_name": "aig",
    "model": {
        "max_size": MAX_NODE_COUNT, # Max nodes in graph
        "node_dim": NUM_NODE_ATTRIBUTES, # Dimension of node features (e.g. 4 explicit + 1 padding = 5)
        "bond_dim": NUM_EDGE_ATTRIBUTES, # Dimension of edge features in adj matrix (e.g. 2 explicit + 1 no-edge = 3)
        "use_gpu": True,
        "edge_unroll": 25, # Max number of previous nodes an edge can connect to
        "num_flow_layer": 12, # Number of flow layers in GraphAF/GraphDF
        "num_rgcn_layer": 3,  # Number of layers in RGCN
        "nhid": 128,          # Hidden dimension for RGCN and ST-Nets
        "nout": 128,          # Output dimension for RGCN (embedding size)
        "deq_coeff": 0.9,     # Dequantization coefficient for GraphAF
        "st_type": "exp",     # Type of ST-Net for GraphAF ('sigmoid', 'exp', 'softplus')
        "use_df": False,      # Flag to indicate if GraphDF specific logic is used (might be redundant if model class implies it)

        # GraphEBM specific parameters (can be nested or flat, depending on how train_graphs.py consumes them)
        "c": 0.05,                      # Dequantization scale for GraphEBM data
        "ld_step": 60,                  # Langevin dynamics steps for GraphEBM sampling
        "ld_noise": 0.005,              # Langevin dynamics noise sigma for GraphEBM sampling
        "ld_step_size": 30.0,           # Langevin dynamics step size (lambda_half) for GraphEBM sampling
        "clamp": True,                  # Whether to clamp gradients during Langevin sampling in GraphEBM
        "alpha": 1.0,                   # Weight for regularizer loss in GraphEBM
        "hidden": 64                    # Hidden dimension for GraphEBM's EnergyFunc
    },
    "lr": 0.001,
    "weight_decay": 0.0, # Changed from 1e-5 to 0.0 as per many graph models
    "batch_size": 32,
    "max_epochs": 30, # Default epochs
    "save_interval": 3, # Save model every N epochs
    "grad_clip_value": 1.0, # Gradient clipping value
}
