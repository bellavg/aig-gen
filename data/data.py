import os
import pickle
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

# Assuming aigverse is installed or available in the path
try:
    from aigverse import read_aiger_into_aig, to_edge_list
except ImportError:
    print("Error: 'aigverse' library not found. Please install it (e.g., pip install aigverse)")
    exit(1)

# Assuming utils_dag.py is in the same directory or accessible in the python path
try:
    from utils_dag import add_order_info
except ImportError:
    print("Error: 'utils_dag.py' not found. Please ensure it is in the same directory as this script.")
    exit(1)

# --- Configuration ---
# Define the one-hot encodings for the different AIG node types.
# The order is: Const0, PI, AND, PO
NODE_TYPE_ENCODING = {
    'CONST0': [1, 0, 0, 0],
    'PI':     [0, 1, 0, 0],
    'AND':    [0, 0, 1, 0],
    'PO':     [0, 0, 0, 1]
}
NUM_NODE_FEATURES = len(NODE_TYPE_ENCODING['PI'])

def create_aig_pyg_data(aig):
    """
    Converts a single AIG object from aigverse into a torch_geometric.data.Data
    object suitable for the DAGformer model. It handles node feature creation,
    edge indexing, and runs the necessary preprocessing steps.

    Args:
        aig: An AIG object from the aigverse library.

    Returns:
        A torch_geometric.data.Data object with all necessary attributes, or None if an error occurs.
    """
    try:
        # --- 1. Map AIG node IDs to a new, contiguous index space ---
        node_mapping = {}
        
        # Helper to add a node to the mapping if it's not already there
        def add_to_mapping(node_id):
            if node_id not in node_mapping:
                node_mapping[node_id] = len(node_mapping)

        # Build the mapping for all existing nodes (Const0, PIs, Gates)
        add_to_mapping(0) # Const0
        for pi_id in aig.pis():
            add_to_mapping(pi_id)
        for gate_id in aig.gates():
            add_to_mapping(gate_id)

        # --- 2. Get Node Features (x) in the correct order ---
        node_features = [None] * len(node_mapping)
        node_features[node_mapping[0]] = NODE_TYPE_ENCODING['CONST0']
        for pi_id in aig.pis():
            node_features[node_mapping[pi_id]] = NODE_TYPE_ENCODING['PI']
        for gate_id in aig.gates():
            node_features[node_mapping[gate_id]] = NODE_TYPE_ENCODING['AND']
        
        # --- 3. Add PO nodes and their edges ---
        # POs are new nodes that connect from existing driver nodes.
        po_start_idx = len(node_mapping)
        source_nodes, target_nodes = [], []
        
        for i, po_literal in enumerate(aig.pos()):
            po_idx = po_start_idx + i
            node_features.append(NODE_TYPE_ENCODING['PO'])
            
            driver_node_id = aig.get_node(po_literal)
            if driver_node_id in node_mapping:
                source_nodes.append(node_mapping[driver_node_id])
                target_nodes.append(po_idx)

        x_tensor = torch.tensor(node_features, dtype=torch.float)

        # --- 4. Get Edge Index (edge_index) ---
        # Add internal edges (to AND gates)
        for edge in to_edge_list(aig):
            if edge.source in node_mapping and edge.target in node_mapping:
                source_nodes.append(node_mapping[edge.source])
                target_nodes.append(node_mapping[edge.target])
        
        edge_index_tensor = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        # --- 5. Create the initial PyG Data object ---
        pyg_data = Data(x=x_tensor, edge_index=edge_index_tensor)
        
        # Add a placeholder for the target variable 'y'.
        # This can be replaced with actual labels later.
        pyg_data.y = torch.zeros((1, 1), dtype=torch.float)

        # --- 6. Run the original preprocessing function from utils_dag.py ---
        # This is a crucial step. It calculates and attaches all the necessary
        # attributes like depth (abs_pe), the attention mask (mask_rc), etc.
        add_order_info(pyg_data)

        return pyg_data

    except Exception as e:
        print(f"    [Error] Failed to process AIG: {e}")
        return None

def process_aig_directory(input_dir, output_file):
    """
    Processes all .aig or .aag files in a directory, converts them to
    a list of PyG Data objects, and saves them to a single pickle file.

    Args:
        input_dir (str): The path to the directory containing AIG files.
        output_file (str): The path where the output pickle file will be saved.
    """
    aig_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.aig', '.aag'))]
    if not aig_files:
        print(f"No .aig or .aag files found in '{input_dir}'")
        return

    print(f"Found {len(aig_files)} AIG files. Starting processing...")
    
    all_graphs_data = []
    for i, filename in enumerate(aig_files):
        file_path = os.path.join(input_dir, filename)
        print(f"--> Processing file {i+1}/{len(aig_files)}: {filename}")
        try:
            aig = read_aiger_into_aig(file_path)
            pyg_data = create_aig_pyg_data(aig)
            if pyg_data:
                all_graphs_data.append(pyg_data)
        except Exception as e:
            print(f"  -> [Critical Error] Could not process {filename}. Error: {e}")

    print(f"\nSuccessfully processed {len(all_graphs_data)} graphs.")
    
    if all_graphs_data:
        print(f"Saving dataset to '{output_file}'...")
        with open(output_file, 'wb') as f:
            pickle.dump(all_graphs_data, f)
        print("Save complete.")

if __name__ == '__main__':
    # --- Example Usage ---
    # This block demonstrates how to use the script.
    
    # 1. Set the directory where your AIG files are located.
    #    For example: INPUT_AIG_DIR = "/path/to/your/aigs"
    INPUT_AIG_DIR = "./sample_aigs" 
    
    # 2. Set the desired name for the output pickle file.
    OUTPUT_PICKLE_FILE = "aig_pyg_dataset.pkl"

    # Create a dummy directory and a sample AIG file for demonstration
    if not os.path.exists(INPUT_AIG_DIR):
        print(f"Creating dummy directory '{INPUT_AIG_DIR}' for demonstration.")
        os.makedirs(INPUT_AIG_DIR)
        # This AAG represents a simple circuit: out = (in1 AND in2)
        # It has 3 nodes (1 const0, 2 PIs), 1 gate, and 1 PO.
        dummy_aag_content = "aag 3 2 0 1 1\n2\n4\n6\n"
        with open(os.path.join(INPUT_AIG_DIR, "and_gate.aag"), "w") as f:
            f.write(dummy_aag_content)
        print(f"Created a dummy 'and_gate.aag' file inside '{INPUT_AIG_DIR}'.")

    # 3. Run the main processing function.
    process_aig_directory(INPUT_AIG_DIR, OUTPUT_PICKLE_FILE)

    # You can now load this .pkl file in your training script.
    print("\n--- To load the data in your training script, use: ---")
    print(f"import pickle")
    print(f"with open('{OUTPUT_PICKLE_FILE}', 'rb') as f:")
    print(f"    train_data = pickle.load(f)")
    print(f"    test_data = [] # or create a split")
    print(f"    graph_args = ... # define your graph args")

