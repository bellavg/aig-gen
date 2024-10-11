import os
import pickle
import networkx as nx
from aigverse import read_aiger_into_aig, to_edge_list
import numpy as np


MAX_GRAPH_SIZE = 2500  # for training purposes and proof of concept starting with constrained graph size

# Directory where all AIG folders are stored
base_dir = './aigs'

# Define one-hot encodings for node types and edge labels
node_type_encoding = {
    "CONST_0": [1, 0, 0, 0, 0],  # Constant 0 node
    "PI": [0, 1, 0, 0, 0],  # Primary Input node
    "AND": [0, 0, 1, 0, 0],  # AND gate node
    "PO": [0, 0, 0, 1, 0]  # Primary Output node
}

edge_label_encoding = {
    "INV": [1, 0],  # Inverted edge
    "REG": [0, 1]  # Regular edge
}


# Function to save all graphs to a single pickle file
def save_all_graphs(all_graphs, output_file):
    with open(output_file, "wb") as f:
        pickle.dump(all_graphs, f)
    print(f"Saved {len(all_graphs)} graphs to {output_file}")


# def get_nodes(aig, G):
#     """For primary inputs and gates, get node information: type, fanins, fanouts, if out then ind of out else 0, and one-hot encode node type"""
#     # Add constant 0 node
#     feature=np.array(node_type_encoding["CONST_0"] + [0, aig.fanout_size(0), 0]) #node_type + [fanins, fanouts, out]
#     G.add_node(0, feature=feature)
#
#     for pi in aig.pis():  # Add input nodes
#         fanouts = aig.fanout_size(pi)
#         out = 0
#         fanins = 0
#         node_type = node_type_encoding["PI"]
#         feature = np.array(node_type + [fanins, fanouts, out])
#         G.add_node(pi, feature=feature)
#
#
#     for gate in aig.gates():  # Add all gate nodes (AND gates)
#         fanouts = aig.fanout_size(gate)
#         node_type = node_type_encoding["AND"]
#         fanins = 2
#         out = 0
#         feature = np.array(node_type + [fanins, fanouts, out])
#         G.add_node(gate, feature=feature)
#
#     return G

def get_nodes(aig, G):
    """Add nodes to the graph with one-hot encoded node types and other features."""
    # Add constant 0 node with feature as NumPy array
    feature = np.array(node_type_encoding["CONST_0"] + [0, aig.fanout_size(0), 0], dtype=np.float32)
    G.add_node(0, feature=feature)

    for pi in aig.pis():  # Add input nodes
        fanouts = aig.fanout_size(pi)
        feature = np.array(node_type_encoding["PI"] + [0, fanouts, 0], dtype=np.float32)
        G.add_node(pi, feature=feature)

    for gate in aig.gates():  # Add gate nodes
        fanouts = aig.fanout_size(gate)
        feature = np.array(node_type_encoding["AND"] + [2, fanouts, 0], dtype=np.float32)
        G.add_node(gate, feature=feature)

    return G


def get_edges(aig, G):
    """Add edges to the graph with one-hot encoded edge labels."""
    edges = to_edge_list(aig, inverted_weight=1, regular_weight=0)

    for e in edges:
        # Assign one-hot encoded edge labels
        onehot_label = np.array(edge_label_encoding["INV"] if e.weight == 1 else edge_label_encoding["REG"], dtype=np.float32)
        G.add_edge(e.source, e.target, label_onehot=onehot_label)

    return G




def get_outs(aig, G, size):
    """Add output nodes and edges to the graph with one-hot encoded output nodes."""
    for ind, po in enumerate(aig.pos()):
        pre_node = aig.get_node(po)
        onehot_label = np.array(edge_label_encoding["INV"] if aig.is_complemented(po) else edge_label_encoding["REG"], dtype=np.float32)

        new_out_node_id = size + ind
        feature = np.array(node_type_encoding["PO"] + [1, 0, ind + 1], dtype=np.float32)  # [type_onehot + fanins, fanouts, out]
        G.add_node(new_out_node_id, feature=feature)
        G.add_edge(pre_node, new_out_node_id, label_onehot=onehot_label)

    return G


# Function to parse .aig file and create a directed graph
def get_graph(aig, name):
    """Create the graph, process nodes and edges, and apply one-hot encoding"""
    G = nx.DiGraph(id=name)

    # Add nodes with one-hot encoding
    G = get_nodes(aig, G)



    # Check if the number of nodes matches
    num_nodes_G = G.number_of_nodes()
    aig_size = aig.size()
    assert num_nodes_G == aig_size, f"Node count mismatch: G has {num_nodes_G}, AIG has {aig_size}"

    # Add edges with one-hot encoding
    G = get_edges(aig, G)

    # Add output nodes and edges
    G = get_outs(aig, G, aig_size)
    pos = aig.num_pos()
    assert G.number_of_nodes() == aig_size + pos, f"Node count mismatch: G has {G.number_of_nodes()}, Should be {aig_size + pos}"


    # for node in G.nodes:
    #     node_type = G.nodes[node]['type_onehot']
    #     fanins = G.nodes[node]['fanins']
    #     fanouts = G.nodes[node]['fanouts']
    #     out = G.nodes[node]['out']
    #
    #
    #     # Concatenate the one-hot encoding of 'type' with the numerical features
    #     G.nodes[node]['feature'] = np.array(node_type + [fanins, fanouts, out]) # 8 dim feature vector per node.


    return G


all_graphs = []

all_aigs_used = []

count = 0
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
                if aig.size() > MAX_GRAPH_SIZE:
                    print(f"Skipping graph {filename} in {folder_name}, too large ({aig.size()} nodes)")
                    continue  # Skip this graph if it's too large

                # Set the graph name as "foldername_filename"
                graph_name = f"{folder_name}_{os.path.splitext(filename)[0]}"

                all_aigs_used.append((graph_name,count))

                G = get_graph(aig, count)

                count += 1

                # Add the graph to the list
                all_graphs.append(G)

# Save all the graphs into one pickle file at the end
save_all_graphs(all_graphs, "all_graphs.pkl")
save_all_graphs(all_aigs_used, "all_aigs_file_ind.pkl")





