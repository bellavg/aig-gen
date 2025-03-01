
from masking import create_masked_aig_with_edges, create_masked_batch
from torch.utils.data import random_split
import torch
from torch_geometric.loader import DataLoader  # <-- Note the differenc
from torch_geometric.data import Batch
from collections import defaultdict
from aig_dataset import AIGDataset
from model import AIGTransformer
import torch
import torch.optim as optim


# Instantiate your dataset (make sure your AIGDataset has been processed)
full_dataset = AIGDataset(file_path="gpt_graphs.pkl", num_graphs=80)
# If the processed file is not created yet, call:
# dataset.process()

# Instantiate your model with the appropriate parameters
model = AIGTransformer(
    node_features=full_dataset[0].x.size(1),  # Assuming all graphs have the same feature dimension
    edge_features=full_dataset[0].edge_attr.size(1) if hasattr(full_dataset[0], 'edge_attr') else 0,
    hidden_dim=64,
    num_layers=2,
    num_heads=2,
    dropout=0.1,
    max_nodes=120
)

num_epochs = 10


# # 2) Decide on the train/val split ratio
val_ratio = 0.2  # e.g. 20% of your data goes to validation
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
#
# # 3) Split into train & validation subsets
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
optimizer = optim.Adam(model.parameters())
# 4) Build train & val DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
# Set device and optimizer
device = next(model.parameters()).device
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training phase
    model.train()
    epoch_losses = defaultdict(float)

    for batch in train_loader:
        batch = batch.to(device)

        # Create masked versions of the AIGs
        masked_batch = create_masked_batch(batch)

        # Forward pass
        predictions = model(masked_batch)

        # Compute loss
        loss, loss_dict = model.compute_loss(
            predictions,
            {
                'node_features': masked_batch.x_target,
                'edge_index': masked_batch.edge_index_target,
                'edge_attr': masked_batch.edge_attr_target if hasattr(masked_batch, 'edge_attr_target') else None
            }
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        for key, value in loss_dict.items():
            epoch_losses[key] += value

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= len(train_loader)

    # Print training metrics
    print(f"Epoch {epoch + 1}/{num_epochs}, Train losses: {dict(epoch_losses)}")

    # Validation phase
    if val_loader is not None and epoch % 2 == 0:
        model.eval()
        val_losses = defaultdict(float)

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # Create masked versions of the AIGs (correctly for a batch)
                masked_batch = create_masked_batch(batch)

                # Forward pass
                predictions = model(masked_batch)

                # Compute loss
                loss, loss_dict = model.compute_loss(
                    predictions,
                    {
                        'node_features': masked_batch.x_target,
                        'edge_index': masked_batch.edge_index_target,
                        'edge_attr': masked_batch.edge_attr_target if hasattr(masked_batch,
                                                                              'edge_attr_target') else None
                    }
                )

                # Accumulate losses
                for key, value in loss_dict.items():
                    val_losses[key] += value

        # Average validation losses
        for key in val_losses:
            val_losses[key] /= len(val_loader)

        print(f"Validation losses: {dict(val_losses)}")

        # Save best model (fixed to use 'node_loss' instead of 'total_loss')
        if val_losses['node_loss'] < best_val_loss:
            best_val_loss = val_losses['node_loss']
            torch.save(model.state_dict(), "best_masked_node.pt")

torch.save(model.state_dict(), "final_masked_node.pt")