class SiameseDagformer(nn.Module):
    """
    A Siamese network using the GraphTransformer (Dagformer) as the backbone.

    This network processes two graphs, generates an embedding for each,
    combines them, and then passes them through a prediction head to produce
    the final output.
    """
    def __init__(self, dagformer_params, output_dim=16, hidden_dim_ratio=0.5):
        """
        Args:
            dagformer_params (dict): A dictionary of parameters to initialize the GraphTransformer.
            output_dim (int): The dimension of the final output vector.
            hidden_dim_ratio (float): The ratio of the hidden layer size in the prediction
                                      head relative to the dagformer's model dimension.
        """
        super().__init__()
        
        # 1. Shared Backbone: Instantiate a single Dagformer.
        #    This same instance will be used to process both input graphs.
        self.dagformer = GraphTransformer(**dagformer_params)
        
        d_model = dagformer_params.get('d_model', 512)
        
        # 2. Pooling Layer: We need a single vector representation for each graph.
        #    We'll use the pooling method specified in the dagformer params.
        self.pooling = self.dagformer.pooling

        # 3. Prediction Head: An MLP that takes the combined graph embeddings
        #    and maps them to the final desired output dimension.
        hidden_dim = int(d_model * hidden_dim_ratio)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data1, data2):
        """
        Forward pass for the Siamese network.

        Args:
            data1 (torch_geometric.data.Data): The first graph (or batch of graphs).
            data2 (torch_geometric.data.Data): The second graph (or batch of graphs).
        
        Returns:
            torch.Tensor: The final prediction vector of shape (batch_size, output_dim).
        """
        
        # --- Process Graph 1 ---
        # Get node-level embeddings from the shared backbone.
        node_embedding1 = self.dagformer.forward_encoder(data1)
        # Pool the node embeddings to get a single graph-level embedding.
        graph_embedding1 = self.pooling(node_embedding1, data1.batch)
        
        # --- Process Graph 2 ---
        # Get node-level embeddings for the second graph using the same backbone.
        node_embedding2 = self.dagformer.forward_encoder(data2)
        # Pool to get the graph-level embedding.
        graph_embedding2 = self.pooling(node_embedding2, data2.batch)
        
        # --- Combine and Predict ---
        # Combine the two graph embeddings. The absolute difference is a common
        # and effective choice for learning a distance or similarity metric.
        combined_embedding = torch.abs(graph_embedding1 - graph_embedding2)
        
        # Pass the combined representation through the final prediction head.
        prediction = self.prediction_head(combined_embedding)
        
        return prediction

# TO TRY 
# combined = torch.cat([graph_embedding1, graph_embedding2, torch.abs(graph_embedding1 - graph_embedding2), graph_embedding1 * graph_embedding2], dim=-1)
# Attention-Based Comparison for prediction head
# Node-Level Comparison - pre pooling cross attention 


import os
import pickle
import torch
import glob
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
import math

class PairedGraphDataset(Dataset):
    """
    Custom PyTorch Dataset to handle the list of paired graph dictionaries.
    """

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        pair = self.data_list[idx]
        return pair['graph1'], pair['graph2'], pair['y']


def collate_paired_graphs(batch):
    """
    Custom collate function to batch pairs of graphs separately.
    """
    graphs1, graphs2, ys = zip(*batch)

    # Use PyG's Batch class to create single batch objects for each set of graphs
    batch1 = Batch.from_data_list(graphs1)
    batch2 = Batch.from_data_list(graphs2)

    # Stack the target tensors
    ys = torch.cat(ys, dim=0)

    return batch1, batch2, ys


def load_dataset(data_path):
    """
    Loads a dataset from a single pickle file or a directory of chunked files.
    """
    if os.path.isdir(data_path):
        # Path is a directory, load all chunks
        search_path = os.path.join(data_path, '*_part_*.pkl')
        chunk_files = sorted(glob.glob(search_path))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found at '{search_path}'")
        print(f"Loading {len(chunk_files)} data chunks from '{data_path}'...")
        all_data = []
        for file in chunk_files:
            with open(file, 'rb') as f:
                all_data.extend(pickle.load(f))
    elif os.path.isfile(data_path):
        # Path is a single file
        print(f"Loading single dataset file from '{data_path}'...")
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
    else:
        raise FileNotFoundError(f"Dataset path '{data_path}' not found.")

    print(f"Successfully loaded {len(all_data)} graph pairs.")
    return all_data


# =====================================================================================
# Section 3: Training and Evaluation Logic
# =====================================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one full epoch of training.
    """
    model.train()
    total_loss = 0
    for data1, data2, y in loader:
        data1, data2, y = data1.to(device), data2.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(data1, data2)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data1.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on a given dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data1, data2, y in loader:
            data1, data2, y = data1.to(device), data2.to(device), y.to(device)
            out = model(data1, data2)
            loss = criterion(out, y)
            total_loss += loss.item() * data1.num_graphs

    return total_loss / len(loader.dataset)


# =====================================================================================
#  Main Execution
# =====================================================================================

def main(args):
    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load and Split Data ---
    full_dataset_list = load_dataset(args.data_path)
    dataset = PairedGraphDataset(full_dataset_list)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_paired_graphs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_paired_graphs)

    # --- Initialize Model, Loss, and Optimizer ---
    dagformer_config = {
        'in_size': 4,  # Node feature dimension from your data prep script
        'd_model': args.d_model,
        'num_heads': args.heads,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'num_layers': args.layers,
        'global_pool': 'mean',
        'abs_pe': 'dagpe',
        'use_edge_attr': True,  # Your data includes edge attributes
        'num_edge_features': 2,  # Edge feature dimension
    }

    model = SiameseDagformer(
        dagformer_params=dagformer_config,
        output_dim=16  # Your structural vector dimension
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()  # Mean Squared Error is suitable for regression

    print("\nModel initialized:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # --- Training Loop ---
    print("\nStarting training...")
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the best model
            if args.save_path:
                torch.save(model.state_dict(), args.save_path)
                print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} -> Model Saved")
        else:
            print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if args.save_path:
        print(f"Best model saved to '{args.save_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Siamese Dagformer model.")
    # Data and saving
    parser.add_argument('--data-path', type=str, required=True,
                        help="Path to the processed dataset file or directory of chunks.")
    parser.add_argument('--save-path', type=str, default="siamese_dagformer_best.pth",
                        help="Path to save the best model checkpoint.")

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--val-split', type=float, default=0.15,
                        help="Proportion of the dataset to use for validation.")

    # Model hyperparameters
    parser.add_argument('--d-model', type=int, default=128, help="Hidden dimension size of the model.")
    parser.add_argument('--layers', type=int, default=4, help="Number of layers in the GraphTransformer.")
    parser.add_argument('--heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--dim-feedforward', type=int, default=256, help="Dimension of the FFN in the transformer.")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate.")

    args = parser.parse_args()
    main(args)
