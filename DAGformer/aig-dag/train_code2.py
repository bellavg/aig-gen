# -*- coding: utf-8 -*-
import copy
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from dag_transformer.models import GraphTransformer
from dag_transformer.data import GraphDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dag_transformer.gnn_layers import GNN_TYPES
import os
from sklearn.model_selection import train_test_split
import pickle
import random


def load_args():
    parser = argparse.ArgumentParser(
        description='DAG transformer for AIG truth table prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--num-heads', type=int, default=8, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=6, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=1028, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout")
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--eval_freq', type=int, default=5,)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=2, help="number of epochs for warmup")
    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--use-edge-attr', action='store_true', help='use edge features')
    parser.add_argument('--edge-dim', type=int, default=256, help='edge features hidden dim')
    parser.add_argument('--gnn-type', type=str, default='gcn',
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")
    parser.add_argument('--global-pool', type=str, default='mean', choices=['mean', 'cls', 'add'],
                        help='global pooling method')
    parser.add_argument('--use_mpnn', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--k', type=int, default=1000)

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    if args.outdir != '':
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

    return args


def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(loader):
        if use_cuda:
            data = data.to(args.device)

        optimizer.zero_grad()
        pred = model(data)  # [batch_size, max_num_outputs, max_truth_table_size]

        # Apply mask to extract valid predictions
        valid_preds = pred[data.y_mask]
        valid_targets = data.y[data.y_mask]

        # Compute loss only for valid predictions
        loss = criterion(valid_preds, valid_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.num_graphs

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


# Save the processed datasets
def save_processed_datasets(train_dset, val_dset, test_dset, save_dir="./processed_data"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "train_dset.pkl"), "wb") as f:
        pickle.dump(train_dset, f)
    with open(os.path.join(save_dir, "val_dset.pkl"), "wb") as f:
        pickle.dump(val_dset, f)
    with open(os.path.join(save_dir, "test_dset.pkl"), "wb") as f:
        pickle.dump(test_dset, f)

    print(f"Datasets saved to {save_dir}")


# Load the processed datasets
def load_processed_datasets(save_dir="./processed_data"):
    with open(os.path.join(save_dir, "train_dset.pkl"), "rb") as f:
        train_dset = pickle.load(f)
    with open(os.path.join(save_dir, "val_dset.pkl"), "rb") as f:
        val_dset = pickle.load(f)
    with open(os.path.join(save_dir, "test_dset.pkl"), "rb") as f:
        test_dset = pickle.load(f)

    print(f"Datasets loaded from {save_dir}")
    return train_dset, val_dset, test_dset


def eval_epoch(model, loader, criterion, use_cuda=False):
    model.eval()
    running_loss = 0.0
    total_accuracy, total_hamming_loss = 0, 0

    with torch.no_grad():
        for data in loader:
            if use_cuda:
                data = data.to(args.device)

            pred = model(data)

            # Apply mask to exclude padded (-inf) values
            mask = data.y_mask
            loss = criterion(pred[mask], data.y[mask])
            running_loss += loss.item() * data.num_graphs

            pred_binary = (torch.sigmoid(pred[mask]) > 0.5).int()
            target_binary = data.y[mask].int()

            total_accuracy += (pred_binary == target_binary).float().mean().item() * data.num_graphs
            total_hamming_loss += (pred_binary != target_binary).float().mean().item() * data.num_graphs

    num_samples = len(loader.dataset)
    avg_loss = running_loss / num_samples
    avg_accuracy = total_accuracy / num_samples
    avg_hamming_loss = total_hamming_loss / num_samples

    return avg_loss, avg_accuracy, avg_hamming_loss


def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load the AIG dataset
    # Check if processed datasets exist
    save_dir = "./processed_data"
    if os.path.exists(os.path.join(save_dir, "train_dset.pkl")):
        # Load processed datasets
        train_dset, val_dset, test_dset = load_processed_datasets(save_dir)
    else:
        with open("./more_graphs_depth_pad_tts.pkl", "rb") as f:
            nx_graphs = pickle.load(f)
        # Process the dataset and save it
        # Limit the dataset to 10,000 samples
        sampled_indices = random.sample(range(len(nx_graphs)), 7500)

        # Create a subset of the original dataset
        sampled_graphs = [nx_graphs[i] for i in sampled_indices]

        # Create the dataset with the sampled graphs
        dataset = GraphDataset(sampled_graphs, use_mpnn=args.use_mpnn, k=args.k)

        train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.3, random_state=args.seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)

        train_dset = [dataset[i] for i in train_idx]
        val_dset = [dataset[i] for i in val_idx]
        test_dset = [dataset[i] for i in test_idx]

        save_processed_datasets(train_dset, val_dset, test_dset, save_dir)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)

    # Define the model
    model = GraphTransformer(
        in_size=4,
        d_model=args.dim_hidden,
        dim_feedforward=2 * args.dim_hidden,
        dropout=args.dropout,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        batch_norm=args.batch_norm,
        gnn_type=args.gnn_type,
        use_edge_attr=args.use_edge_attr,
        num_edge_features=2,
        edge_dim=args.edge_dim,
        in_embed=True,
        edge_embed=False,
        global_pool=args.global_pool
    )

    if args.use_cuda:
        model.to(args.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_accuracy = 0
    best_model_weights = None
    print("Training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=args.use_cuda)
        if epoch % args.eval_freq == 0:
            val_loss, val_accuracy, val_hamming_loss = eval_epoch(model, val_loader, criterion, use_cuda=args.use_cuda)
            print(f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, Val Hamming Loss = {val_hamming_loss:.4f}")
            lr_scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{args.epochs}: Train Loss = {train_loss:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = os.path.join("./checkpoints", f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)

    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

    model.load_state_dict(best_model_weights)
    test_loss, test_accuracy, test_hamming_loss = eval_epoch(model, test_loader, criterion, use_cuda=args.use_cuda)

    print(f"Test Results: Loss = {test_loss:.4f}, Accuracy = {test_accuracy:.4f}, Hamming Loss = {test_hamming_loss:.4f}")


if __name__ == "__main__":
    main()
