import pickle
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import argparse
from model.encoder import GATEncoder


# Function to convert NetworkX graph to PyTorch Geometric format
def convert_nx_to_pyg(G):
    # Convert NetworkX graph to PyTorch Geometric Data format, gives warning due to
    # numpy arrays but its internal and works better than tensors
    data = from_networkx(G, group_edge_attrs=["label_onehot"], group_node_attrs=["feature"])
    return data


# Function to load graphs from pickle file and convert them to PyTorch Geometric Data
def load_graphs(pickle_file):
    with open(pickle_file, 'rb') as f:
        all_graphs = pickle.load(f)  # List of NetworkX graphs
    # Convert each NetworkX graph to PyTorch Geometric Data
    all_graph_data = [convert_nx_to_pyg(g) for g in all_graphs]
    return all_graph_data


# Main function with argument parser
def main():
    parser = argparse.ArgumentParser(description="Test GATv2 Graph Neural Network on AIG Graphs")

    # Arguments for the test
    parser.add_argument('--graph_file', type=str, default="./data/all_graphs.pkl",
                        help="Path to the .pkl file containing the graphs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for testing the model")
    parser.add_argument('--split', type=float, default=0.2, help="Train, test split.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for data loader and co.")
    parser.add_argument('--hidden_dim', type=int, default=32, help="Hidden layer size")
    parser.add_argument('--out_dim', type=int, default=64, help="Output dimension size (latent space)")
    parser.add_argument('--heads', type=int, default=4, help="Number of attention heads")
    parser.add_argument('--dropout', type=float, default=0.6, help="Dropout rate")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of GATv2Conv layers")

    args = parser.parse_args()

    # Load the graphs from the pickle file
    print(f"Loading graphs from {args.graph_file}...")
    all_graph_data = load_graphs(args.graph_file)

    # Split data into train and test sets
    train_graphs, test_graphs = train_test_split(all_graph_data, test_size=args.split, random_state=args.random_seed)

    # Create DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    # Define model parameters based on the first graph's feature dimensions
    node_in_dim = len(train_graphs[0].x[0])  # Dimension of node features
    edge_in_dim = len(train_graphs[0].edge_attr[0])  # Dimension of edge features

    # Initialize GATv2 encoder
    gatv2_encoder = GATEncoder(
        in_channels=node_in_dim,
        hidden_channels=args.hidden_dim,
        latent_dim=args.out_dim,
        heads=args.heads,
        num_layers=args.num_layers,
        edge_dim=edge_in_dim,
        dropout=args.dropout
    )

    # Decoder

    # Functional Equivalence

    # Loss

    

    # Test on one batch from the test loader
    gatv2_encoder.eval()  # Set the encoder to evaluation mode

    # Iterate through one batch from the test loader
    for batch in test_loader:
        print(f"Batch contains {batch.num_graphs} graphs.")

        # Forward pass (no need for gradients)
        with torch.no_grad():
            output = gatv2_encoder(batch.x, batch.edge_index, batch.edge_attr)

        print("Output shape (latent representation):", output.shape)
        break  # Stop after one batch


if __name__ == "__main__":
    main()
