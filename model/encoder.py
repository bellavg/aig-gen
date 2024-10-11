import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, edge_dim, heads=4, num_layers=3, dropout=0.6):
        super(GATEncoder, self).__init__()

        # Initial GATv2Conv layer with `heads` number of attention heads
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim, dropout=dropout,
                               add_self_loops=True, residual=False)

        # Additional GATv2Conv layers (keeping the number of heads the same)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels * heads, heads=heads, edge_dim=edge_dim,
                                        dropout=dropout, add_self_loops=True, residual=True))

        # Final GATv2Conv layer to project into latent space
        self.conv_final = GATv2Conv(hidden_channels * heads * heads, latent_dim, heads=1, edge_dim=edge_dim, dropout=dropout,
                                    add_self_loops=True, residual=False)  # No residual in the final projection

    def forward(self, x, edge_index, edge_attr):
        # Forward pass through the first GATv2Conv layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Forward pass through the intermediate GATv2Conv layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        # Forward pass through the final GATv2Conv layer (latent projection)
        x = self.conv_final(x, edge_index, edge_attr)

        return x
