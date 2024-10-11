from aigverse import read_aiger_into_aig, to_edge_list
import os
import networkx as nx
import pickle
# Parse .aig files
# check size
# Get node features get edge features
# Make networkx directed graph
# Save graphs

MAX_GRAPH_SIZE = 2500 # for training purposes and proof of concept starting with constrained graph size

# Directory where all AIG folders are stored
base_dir = './aigs'




# Function to save all graphs to a single pickle file
def save_all_graphs(all_graphs, output_file):
    with open(output_file, "wb") as f:
        pickle.dump(all_graphs, f)
    print(f"Saved {len(all_graphs)} graphs to {output_file}")



def get_nodes(aig, G):
    """for primary inputs and gates get node information: type, fanins, fanouts, depth?"""
    # Iterate only over primary inputs
    # TODO: Add depth??

    G.add_node(0, type="CONST_0") # add constant 0 node

    for pi in aig.pis(): # add input nodes
        fanouts = aig.fanout_size(pi)
        G.add_node(pi, type="PI", fanins=0, fanouts=fanouts)

    for gate in aig.gates(): # add all gates
        fanouts = aig.fanout_size(gate)
        G.add_node(gate, type="AND", fanins=2, fanouts=fanouts)

    return G



def get_edges(aig, G):
    "Add to networkx graph: AIG edges and edge label"
    edges = to_edge_list(aig, inverted_weight=1, regular_weight=0)

    for e in edges: # add all edges
        # Assign label based on weight
        if e.weight == 1:
            label = "INV"
        elif e.weight == 0:
            label = "REG"
        else:
            label = "UNKNOWN"  # Handle unexpected cases if necessary
        G.add_edge(e.source, e.target, label=label)

    return G



def get_outs(aig, G, size):
    # # no off by one error. Example size of aig is 1033, last node is index 1032
    for ind, po in enumerate(aig.pos()):
        pre_node = aig.get_node(po)
        if aig.is_complemented(po):
            label = "INV"
        else:
            label = "REG"
        new_out_node_id = size + ind
        G.add_node(new_out_node_id, type="PO", num=ind+1, fanins=1, fanouts=0)
        G.add_edge(pre_node, new_out_node_id, label=label)

    return G


# Function to parse .aig file and create a directed graph
def get_graph(aig, name):
    # Create a new directed graph
    G = nx.DiGraph(name=name)

    # get nodes and node features: type, fan-ins, fan-outs, depth
    G = get_nodes(aig, G)
    # Check if the number of nodes matches
    # Get the number of nodes in G
    num_nodes_G = G.number_of_nodes()
    # Get the number of nodes in the AIG
    aig_size = aig.size()
    assert num_nodes_G == aig_size, f"Node count mismatch: G has {num_nodes_G}, AIG has {aig_size}"

    # Add edges
    G = get_edges(aig, G)

    # Add output nodes and edges
    G = get_outs(aig, G, aig_size)
    pos = aig.num_pos()
    assert G.number_of_nodes() == aig_size+aig.num_pos(), f"Node count mismatch: G has {G.number_of_nodes()}, Should be {aig_size+aig.num_pos()}"

    return G

all_graphs = []

# Iterate through all subfolders and .aig files
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)

    if os.path.isdir(folder_path):  # Check if it's a folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.aig'):  # Process only .aig files
                file_path = os.path.join(folder_path, filename)

                # Create an aig object from the .aig file
                aig = read_aiger_into_aig(file_path)

                # Add size check
                #  the size of the graph (number of nodes in this example)
                if aig.size() > MAX_GRAPH_SIZE:
                    print(f"Skipping graph {filename} in {folder_name}, too large ({aig.size()} nodes)")
                    continue  # Skip this graph if it's too large

                # Set the graph name as "foldername_filename"
                graph_name = f"{folder_name}_{os.path.splitext(filename)[0]}"

                G = get_graph(aig, graph_name)

                # Add the graph to the list
                all_graphs.append(G)

# Save all the graphs into one pickle file at the end
save_all_graphs(all_graphs, "all_graphs.pkl")







