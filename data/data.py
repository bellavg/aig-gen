import os
import pickle
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
import argparse

# Assuming aigverse is installed or available in the path
try:
    from aigverse import read_aiger_into_aig, to_edge_list
except ImportError:
    print("Error: 'aigverse' library not found. Please install it (e.g., pip install aigverse)")
    exit(1)

# Assuming utils_dag.py is in the same directory or accessible in the python path
try:
    from utils_dag import add_order_info
except (ImportError, FileNotFoundError):
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

# Define one-hot encodings for edge types (regular vs. inverted)
EDGE_TYPE_ENCODING = {
    'REGULAR':  [1, 0],
    'INVERTED': [0, 1]
}
NUM_EDGE_FEATURES = len(EDGE_TYPE_ENCODING['REGULAR'])

def create_aig_pyg_data(aig):
    """
    Converts a single AIG object from aigverse into a torch_geometric.data.Data
    object suitable for the DAGformer model. It handles node and edge feature creation,
    and runs the necessary preprocessing steps.

    Args:
        aig: An AIG object from the aigverse library.

    Returns:
        A torch_geometric.data.Data object with all necessary attributes, or None if an error occurs.
    """
    try:
        # --- 1. Map AIG node IDs to a new, contiguous index space ---
        node_mapping = {}
        
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
        
        # --- 3. Add PO nodes and prepare for edge creation ---
        po_start_idx = len(node_mapping)
        source_nodes, target_nodes = [], []
        edge_features = [] # List to store edge feature vectors

        for i, po_literal in enumerate(aig.pos()):
            po_idx = po_start_idx + i
            node_features.append(NODE_TYPE_ENCODING['PO'])
            
            driver_node_id = aig.get_node(po_literal)
            if driver_node_id in node_mapping:
                source_nodes.append(node_mapping[driver_node_id])
                target_nodes.append(po_idx)
                # Add edge feature for this PO connection
                is_inverted = aig.is_complemented(po_literal)
                edge_features.append(EDGE_TYPE_ENCODING['INVERTED'] if is_inverted else EDGE_TYPE_ENCODING['REGULAR'])

        x_tensor = torch.tensor(node_features, dtype=torch.float)

        # --- 4. Get Edge Index (edge_index) and Edge Attributes (edge_attr) ---
        # Add internal edges (to AND gates)
        for edge in to_edge_list(aig):
            if edge.source in node_mapping and edge.target in node_mapping:
                source_nodes.append(node_mapping[edge.source])
                target_nodes.append(node_mapping[edge.target])
                # Add edge feature for this internal connection
                is_inverted = (edge.weight == 1)
                edge_features.append(EDGE_TYPE_ENCODING['INVERTED'] if is_inverted else EDGE_TYPE_ENCODING['REGULAR'])
        
        edge_index_tensor = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_features, dtype=torch.float)

        # --- 5. Create the initial PyG Data object ---
        pyg_data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
        
        # Add a placeholder for the target variable 'y'.
        pyg_data.y = torch.zeros((1, 1), dtype=torch.float)

        # --- 6. Run the original preprocessing function from utils_dag.py ---
        # This calculates and attaches depth (abs_pe), attention mask (mask_rc), etc.
        add_order_info(pyg_data)

        return pyg_data

    except Exception as e:
        print(f"     [Error] Failed to process AIG: {e}")
        return None

def _save_data(data, output_path):
    """Saves a list of graph data objects to a pickle file."""
    if not data:
        return
    
    print(f"  Saving {len(data)} graphs to '{output_path}'...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved.")

def process_aig_directory(input_dir, output_dir, base_filename, chunk_size=1000, use_chunking=True):
    """
    Processes all .aig or .aag files in a directory, converts them to
    a list of PyG Data objects, and saves them to pickle files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    aig_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.aig', '.aag'))]
    if not aig_files:
        print(f"No .aig or .aag files found in '{input_dir}'")
        return

    print(f"Found {len(aig_files)} AIG files. Starting processing...")
    if use_chunking:
        print(f"Chunking is enabled with chunk size {chunk_size}.")
    
    processed_data = []
    chunk_count = 1
    
    for i, filename in enumerate(aig_files):
        file_path = os.path.join(input_dir, filename)
        print(f"--> Processing file {i+1}/{len(aig_files)}: {filename}")
        try:
            aig = read_aiger_into_aig(file_path)
            pyg_data = create_aig_pyg_data(aig)
            if pyg_data:
                processed_data.append(pyg_data)
            
            if use_chunking and len(processed_data) >= chunk_size:
                chunk_filename = f"{base_filename}_part_{chunk_count}.pkl"
                output_path = os.path.join(output_dir, chunk_filename)
                _save_data(processed_data, output_path)
                processed_data = []
                chunk_count += 1

        except Exception as e:
            print(f"  -> [Critical Error] Could not process {filename}. Error: {e}")

    # Save any remaining data
    if use_chunking:
        if processed_data:
            chunk_filename = f"{base_filename}_part_{chunk_count}.pkl"
            output_path = os.path.join(output_dir, chunk_filename)
            _save_data(processed_data, output_path)
        print(f"\nProcessing complete. Saved {chunk_count} chunk(s) to '{output_dir}'.")
    else:
        output_path = os.path.join(output_dir, f"{base_filename}.pkl")
        _save_data(processed_data, output_path)
        print(f"\nProcessing complete. Saved all data to '{output_path}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process AIG files into PyTorch Geometric Data objects.")
    parser.add_argument('--input-dir', type=str, default="./sample_aigs", help="Directory containing AIG files.")
    parser.add_argument('--output-dir', type=str, default="./processed_aigs", help="Directory to save processed data.")
    parser.add_argument('--output-filename', type=str, default="aig_pyg_dataset", help="Base name for the output file(s).")
    parser.add_argument('--chunk-size', type=int, default=1000, help="Number of graphs per chunk (if chunking is enabled).")
    parser.add_argument('--no-chunking', action='store_true', help="Disable chunking and save all data to a single file.")
    
    args = parser.parse_args()

    # --- Example Usage ---
    if not os.path.exists(args.input_dir):
        print(f"Creating dummy directory '{args.input_dir}' for demonstration.")
        os.makedirs(args.input_dir)
        dummy_aag_content = "aag 3 2 0 1 1\n2\n4\n6\n"
        with open(os.path.join(args.input_dir, "and_gate.aag"), "w") as f:
            f.write(dummy_aag_content)
        print(f"Created a dummy 'and_gate.aag' file inside '{args.input_dir}'.")

    process_aig_directory(
        args.input_dir, 
        args.output_dir, 
        args.output_filename, 
        chunk_size=args.chunk_size, 
        use_chunking=not args.no_chunking
    )

    print("\n--- To load the processed dataset in your training script, use: ---")
    if not args.no_chunking:
        print("import os, pickle, glob")
        print(f"data_files = sorted(glob.glob(os.path.join('{args.output_dir}', '{args.output_filename}_part_*.pkl')))")
        print("all_data = []")
        print("for file in data_files:")
        print("    with open(file, 'rb') as f:")
        print("        all_data.extend(pickle.load(f))")
        print("print(f'Loaded {len(all_data)} graphs from {len(data_files)} files.')")
    else:
        print("import pickle")
        print(f"with open(os.path.join('{args.output_dir}', '{args.output_filename}.pkl'), 'rb') as f:")
        print("    all_data = pickle.load(f)")
        print("print(f'Loaded {len(all_data)} graphs.')")
