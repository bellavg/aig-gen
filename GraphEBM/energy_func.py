import torch
from torch import nn
from torch.nn import functional as F


def swish(x):
    """
    Swish activation function.
    x * sigmoid(x)
    """
    return x * torch.sigmoid(x)


class SpectralNorm:
    """
    Spectral Normalization for a given module.
    This class implements spectral normalization as a forward pre-hook.
    It estimates the largest singular value of the weight matrix and normalizes the weights.

    Args:
        name (str): Name of the weight parameter to normalize (e.g., 'weight').
        bound (bool): If True, applies bounded spectral normalization, clamping sigma to 1.
                      Otherwise, divides by sigma.
        n_power_iterations (int): Number of power iterations to estimate spectral norm.
                                  Usually 1 is sufficient.
    """

    def __init__(self, name, bound=False, n_power_iterations=1):
        self.name = name
        self.bound = bound
        self.n_power_iterations = n_power_iterations

    def compute_weight(self, module):
        """
        Computes the spectrally normalized weight and the updated 'u' vector.
        Uses power iteration to estimate the singular value.
        """
        weight = getattr(module, self.name + '_orig')  # Original weight matrix
        u = getattr(module, self.name + '_u')  # Left singular vector estimate

        weight_mat = weight.contiguous().view(weight.size(0), -1)  # Flatten weight for matrix operations

        with torch.no_grad():  # Power iteration steps
            for _ in range(self.n_power_iterations):
                v = weight_mat.t() @ u
                v = v / (v.norm(p=2) + 1e-12)  # Normalize v, add epsilon for stability
                u = weight_mat @ v
                u = u / (u.norm(p=2) + 1e-12)  # Normalize u, add epsilon for stability

        sigma = (u.unsqueeze(0) @ weight_mat @ v.unsqueeze(1)).squeeze()  # Estimate of the largest singular value

        if self.bound:
            # Bounded spectral normalization: sigma is clamped at 1.0
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1.0)
        else:
            # Unbounded spectral normalization: divides by sigma.
            weight_sn = weight / (sigma + 1e-12)  # Added epsilon for stability

        return weight_sn, u

    @staticmethod
    def apply(module, name, bound, n_power_iterations=1):
        """
        Applies spectral normalization to a parameter of a module.
        Modifies the module in-place.
        """
        fn = SpectralNorm(name, bound, n_power_iterations)

        weight = getattr(module, name)  # Get the original weight parameter
        if name in module._parameters:
            del module._parameters[name]  # Remove original weight from parameters

        # Register original weight with suffix '_orig'
        module.register_parameter(name + '_orig', weight)

        # Initialize 'u' vector for power iteration
        # For nn.Linear, weight size is (out_features, in_features)
        # u should be of size (out_features)
        # For nn.Conv, weight size is (out_channels, in_channels/groups, *kernel_size)
        # u should be of size (out_channels)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_(0, 1)  # Random initialization for u

        # Register 'u' as a buffer (not a parameter, but part of state)
        # Also register the (temporary) normalized weight as a buffer to be updated by the hook
        module.register_buffer(name, weight.data.clone())  # Initialize buffer with current weight data
        module.register_buffer(name + '_u', u)

        # Register the forward pre-hook that will update the weight before each forward pass
        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input_val):  # input_val is the input to the module's forward
        """
        Forward pre-hook: called before the module's forward pass.
        Computes and sets the spectrally normalized weight.
        """
        weight_sn, u_new = self.compute_weight(module)
        setattr(module, self.name, weight_sn)  # Set the normalized weight
        # Ensure u_new is correctly assigned back to the buffer.
        # Direct assignment to buffer is okay.
        module.set_buffer(self.name + '_u', u_new)


def spectral_norm(module, name='weight', init=True, std=1.0, bound=False, n_power_iterations=1):
    """
    Helper function to apply spectral normalization to a module's specified parameter.

    Args:
        module (nn.Module): The module to apply spectral normalization to.
        name (str): Name of the parameter to normalize (e.g., 'weight').
        init (bool): If True, initializes the weights using a normal distribution with mean 0 and std.
        std (float): Standard deviation for weight initialization if init is True.
        bound (bool): If True, uses bounded spectral normalization.
        n_power_iterations (int): Number of power iterations for SN.

    Returns:
        nn.Module: The module with spectral normalization applied.
    """
    if init and hasattr(module, name):
        # Initialize weights if requested
        nn.init.normal_(getattr(module, name), 0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        # Initialize bias to zero
        module.bias.data.zero_()

    # Apply spectral normalization to the specified parameter
    if hasattr(module, name):
        SpectralNorm.apply(module, name, bound=bound, n_power_iterations=n_power_iterations)
    return module


class GraphConv(nn.Module):
    """
    Graph Convolutional Layer.
    Operates on graph-structured data with node features and adjacency matrices.
    Supports multiple edge types and optional self-loops.

    Args:
        in_channels (int): Number of input node features.
        out_channels (int): Number of output node features.
        num_edge_type (int): Number of different types of edges in the graph.
        std (float): Standard deviation for weight initialization in spectral_norm.
        bound (bool): Whether to use bounded spectral normalization. Defaults to True.
        add_self (bool): If True, adds a self-connection for each node (learnable transformation).
        n_power_iterations (int): Number of power iterations for SN.
    """

    def __init__(self, in_channels, out_channels, num_edge_type, std, bound=True, add_self=False, n_power_iterations=1):
        super(GraphConv, self).__init__()

        self.add_self = add_self
        if self.add_self:
            # Linear transformation for node's own features (self-loop)
            linear_node_layer = nn.Linear(in_channels, out_channels)
            self.linear_node = spectral_norm(linear_node_layer, std=std, bound=bound,
                                             n_power_iterations=n_power_iterations)

        # Linear transformation for messages aggregated from neighbors along different edge types
        linear_edge_layer = nn.Linear(in_channels, out_channels * num_edge_type)
        self.linear_edge = spectral_norm(linear_edge_layer, std=std, bound=bound, n_power_iterations=n_power_iterations)

        self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels

    def forward(self, adj, h):
        """
        Forward pass for the Graph Convolution layer.

        Args:
            adj (torch.Tensor): Adjacency tensor of shape (batch_size, num_edge_type, num_nodes, num_nodes).
            h (torch.Tensor): Node feature tensor of shape (batch_size, num_nodes, in_channels).

        Returns:
            torch.Tensor: Output node features of shape (batch_size, num_nodes, out_channels).
        """
        batch_size, num_nodes, _ = h.shape

        h_node = self.linear_node(h) if self.add_self else None

        m = self.linear_edge(h)
        m = m.reshape(batch_size, num_nodes, self.out_ch, self.num_edge_type)
        m = m.permute(0, 3, 1, 2)

        hr = torch.matmul(adj, m)
        hr = hr.sum(dim=1)

        if self.add_self and h_node is not None:
            return hr + h_node
        else:
            return hr


class EnergyFunc(nn.Module):
    """
    Energy Function for the GraphEBM model, aligned with the paper's description.

    Args:
        n_atom_type (int): Number of possible atom types (input feature dimension for nodes).
        hidden (int): Number of hidden units in GraphConv layers and the final linear layer.
        num_edge_type (int): Number of edge types in the graph. Defaults to 3.
        swish_act (bool): If True, uses Swish activation. Otherwise, uses LeakyReLU. Defaults to True (as per paper Appendix D).
        depth (int): Number of additional GraphConv layers after the first one. Defaults to 2 (total L=3 layers as per paper Appendix D).
        add_self (bool): If True, GraphConv layers include self-connections. Defaults to False.
        dropout (float): Dropout probability after each GraphConv layer. Defaults to 0.
        n_power_iterations (int): Number of power iterations for SN. Defaults to 1.
    """

    def __init__(self, n_atom_type, hidden, num_edge_type=3, swish_act=True, depth=2, add_self=False, dropout=0,
                 n_power_iterations=1):
        super(EnergyFunc, self).__init__()

        self.depth = depth
        self.swish_act = swish_act
        self.dropout_p = dropout

        # First Graph Convolutional layer
        # Paper Appendix D: Spectral norm to all layers. std=1, bound=False is a choice from original code.
        self.graphconv1 = GraphConv(n_atom_type, hidden, num_edge_type, std=1.0, bound=False, add_self=add_self,
                                    n_power_iterations=n_power_iterations)

        # Subsequent Graph Convolutional layers
        # std=0.02 and bound=True (default in GraphConv) for deeper layers.
        self.graphconv_layers = nn.ModuleList(
            GraphConv(hidden, hidden, num_edge_type, std=0.02, bound=True, add_self=add_self,
                      n_power_iterations=n_power_iterations) for _ in range(self.depth)
        )

        # Final linear layer to output a single energy value
        # Paper Eq.4: E = h_G^T W (no bias). Appendix D: SN to all layers.
        final_linear_layer = nn.Linear(hidden, 1, bias=False)
        self.linear = spectral_norm(final_linear_layer, std=1.0, bound=False,
                                    n_power_iterations=n_power_iterations)  # std for final layer can be 1.0 or smaller.

    def forward(self, adj, h):
        """
        Forward pass for the Energy Function.

        Args:
            adj (torch.Tensor): Adjacency tensor of shape (batch_size, num_edge_type, num_nodes, num_nodes).
            h (torch.Tensor): Node feature tensor of shape (batch_size, num_nodes, n_atom_type).

        Returns:
            torch.Tensor: Energy value for each graph in the batch, shape (batch_size, 1).
        """
        out = self.graphconv1(adj, h)
        out = F.dropout(out, p=self.dropout_p, training=self.training)

        if self.swish_act:
            out = swish(out)
        else:
            out = F.leaky_relu(out, negative_slope=0.2)

        for i in range(self.depth):
            out = self.graphconv_layers[i](adj, out)
            out = F.dropout(out, p=self.dropout_p, training=self.training)
            if self.swish_act:
                out = swish(out)
            else:
                out = F.leaky_relu(out, negative_slope=0.2)

                # Global pooling: aggregate node features into a single graph-level representation
        # Paper Section 2.1 Eq.3: "sum operation to compute the graph-level representation"
        out = out.sum(dim=1)  # (batch_size, num_nodes, hidden_channels) --> (batch_size, hidden_channels)

        # Final linear layer to get the energy value
        out = self.linear(out)  # (batch_size, 1)

        return out
