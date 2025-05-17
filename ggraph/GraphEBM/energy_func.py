import torch
from torch import nn
from torch.nn import functional as F


def swish(x):
    return x * torch.sigmoid(x)


class SpectralNorm:
    def __init__(self, name, bound=False):
        self.name = name
        self.bound = bound

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)

        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()

        sigma = u @ weight_mat @ v

        if self.bound:
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1)

        else:
            weight_sn = weight / sigma

        return weight_sn, u

    @staticmethod
    def apply(module, name, bound):
        fn = SpectralNorm(name, bound)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)  # Store current weight_sn
        module.register_buffer(name + '_u', u)  # Store current u

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, init=True, std=1, bound=False):
    if init:
        nn.init.normal_(module.weight, 0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    SpectralNorm.apply(module, 'weight', bound=bound)

    return module


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_edge_type, std, bound=True, add_self=False):
        """
        Graph Convolution Layer.
        Args:
            in_channels (int): Number of input features per node (F_total).
            out_channels (int): Number of output features per node (Hidden dimension H).
            num_edge_type (int): Number of edge types/channels in the adjacency matrix (E_total).
            std (float): Standard deviation for weight initialization.
            bound (bool): Whether to bound spectral norm.
            add_self (bool): Whether to add a self-connection path.
        """
        super(GraphConv, self).__init__()

        self.add_self = add_self
        if self.add_self:
            # Linear transformation for self-loop path: F_total -> H
            self.linear_node = spectral_norm(nn.Linear(in_channels, out_channels), std=std, bound=bound)

        # Linear transformation for messages from neighbors: F_total -> (H * E_total)
        # This will be reshaped to (H, E_total) per node later.
        self.linear_edge = spectral_norm(nn.Linear(in_channels, out_channels * num_edge_type), std=std, bound=bound)

        self.num_edge_type = num_edge_type  # E_total
        self.in_ch = in_channels  # F_total
        self.out_ch = out_channels  # H (hidden dimension)

    def forward(self, adj, h):
        """
        Forward pass for Graph Convolution.
        Args:
            adj (torch.Tensor): Adjacency tensor of shape (Batch, E_total, Nodes, Nodes).
            h (torch.Tensor): Node feature tensor of shape (Batch, F_total, Nodes).
                              This is after permutation in EnergyFunc.
        Returns:
            torch.Tensor: Output node features of shape (Batch, Nodes, H).
        """
        # h input shape: (Batch, Features_in, Num_Nodes) -> e.g., (32, 5, 64)
        # adj input shape: (Batch, Edge_Channels, Num_Nodes, Num_Nodes) -> e.g., (32, 3, 64, 64)

        batch_size, num_features_input, num_nodes = h.shape

        # Permute h to (Batch, Num_Nodes, Features_in) for nn.Linear layers
        # h_permuted shape: (32, 64, 5)
        h_permuted = h.permute(0, 2, 1).contiguous()  # Ensure contiguous after permute

        if self.add_self:
            # self.linear_node expects input (*, self.in_ch) where self.in_ch = F_total (5)
            # h_node shape: (Batch, Num_Nodes, self.out_ch) -> (32, 64, 64)
            h_node = self.linear_node(h_permuted)

        # self.linear_edge expects input (*, self.in_ch)
        # m shape: (Batch, Num_Nodes, self.out_ch * self.num_edge_type) -> (32, 64, 64*3=192)
        m = self.linear_edge(h_permuted)

        # Reshape m to separate out_channels (H) and edge_types (E_total)
        # m shape: (Batch, Num_Nodes, self.out_ch, self.num_edge_type) -> (32, 64, 64, 3)
        m = m.view(batch_size, num_nodes, self.out_ch, self.num_edge_type)

        # Permute m for batched matrix multiplication with adj
        # m shape: (Batch, self.num_edge_type, Num_Nodes, self.out_ch) -> (32, 3, 64, 64)
        m = m.permute(0, 3, 1, 2)

        # adj shape: (Batch, self.num_edge_type, Num_Nodes, Num_Nodes) -> (32, 3, 64, 64)
        # m shape:   (Batch, self.num_edge_type, Num_Nodes, self.out_ch)  -> (32, 3, 64, 64)
        # Batched matrix multiplication: (N,N) @ (N,H) -> (N,H) for each batch and edge type
        # hr shape: (Batch, self.num_edge_type, Num_Nodes, self.out_ch) -> (32, 3, 64, 64)
        hr = torch.matmul(adj, m)

        # Sum messages over different edge types
        # hr shape: (Batch, Num_Nodes, self.out_ch) -> (32, 64, 64)
        hr = hr.sum(dim=1)

        if self.add_self:
            return hr + h_node
        else:
            return hr


class EnergyFunc(nn.Module):

    def __init__(self, n_atom_type, hidden, num_edge_type=3, swish_act=True, depth=2, add_self=False,
                 dropout=0.0):  # Changed swish to swish_act
        super(EnergyFunc, self).__init__()

        self.depth = depth
        # n_atom_type is total node features (e.g., NUM_NODE_ATTRIBUTES = 5)
        # hidden is the hidden dimension (e.g., 64)
        # num_edge_type is total edge channels (e.g., NUM_EDGE_ATTRIBUTES = 3)
        self.graphconv1 = GraphConv(n_atom_type, hidden, num_edge_type, std=1, bound=False, add_self=add_self)

        self.graphconv_layers = nn.ModuleList(  # Renamed from graphconv to graphconv_layers
            GraphConv(hidden, hidden, num_edge_type, std=1e-10, add_self=add_self) for _ in range(self.depth)
            # Corrected loop variable
        )
        self.use_swish = swish_act  # Use the renamed variable
        self.dropout_p = dropout  # Use the renamed variable
        self.final_linear = nn.Linear(hidden, 1)  # Renamed from linear

    def forward(self, adj, h):
        # adj input shape: (Batch, E_total, Max_Nodes, Max_Nodes) e.g. (32, 3, 64, 64)
        # h input shape: (Batch, Max_Nodes, F_total) e.g. (32, 64, 5) - from DataLoader

        # Permute h to (Batch, F_total, Max_Nodes) for GraphConv input convention
        # h_permuted shape: (32, 5, 64)
        h_permuted = h.permute(0, 2, 1).contiguous()

        # First graph convolution
        # out shape: (Batch, Max_Nodes, Hidden) -> (32, 64, 64)
        out = self.graphconv1(adj, h_permuted)

        out = F.dropout(out, p=self.dropout_p, training=self.training)

        if self.use_swish:
            out = swish(out)
        else:
            out = F.leaky_relu(out, negative_slope=0.2)

        # Subsequent graph convolution layers
        for i in range(self.depth):
            # For subsequent layers, input 'out' is (Batch, Max_Nodes, Hidden)
            # GraphConv expects h as (Batch, Features, Max_Nodes)
            # So, permute 'out' before passing to graphconv_layers[i]
            out_permuted_for_gcn = out.permute(0, 2, 1).contiguous()  # (Batch, Hidden, Max_Nodes)
            out = self.graphconv_layers[i](adj, out_permuted_for_gcn)  # Output (Batch, Max_Nodes, Hidden)

            out = F.dropout(out, p=self.dropout_p, training=self.training)
            if self.use_swish:
                out = swish(out)
            else:
                # Original code used F.relu(out) for subsequent layers.
                # Using leaky_relu for consistency or relu as per original.
                # Sticking to relu as per original deeper layers.
                out = F.relu(out)

                # Sum pooling over nodes
        # out shape before sum: (Batch, Max_Nodes, Hidden) -> (32, 64, 64)
        # out_summed shape: (Batch, Hidden) -> (32, 64)
        out_summed = out.sum(dim=1)

        # Final linear layer to get energy value
        # energy_val shape: (Batch, 1)
        energy_val = self.final_linear(out_summed)

        return energy_val
