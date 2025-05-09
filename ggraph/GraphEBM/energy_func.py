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

        # Recalculate sigma with the final u and v
        # sigma = u.T @ weight_mat @ v # This is for u as row vector
        # If u is (out_features) and v is (in_features_flat), and weight_mat is (out_features, in_features_flat)
        # then u.T @ weight_mat @ v is (1, out) @ (out, in_flat) @ (in_flat, 1) -> scalar
        sigma = (u.unsqueeze(0) @ weight_mat @ v.unsqueeze(1)).squeeze()

        if self.bound:
            # Bounded spectral normalization: sigma is clamped at 1.0
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1.0)
        else:
            # Unbounded spectral normalization: divides by sigma.
            weight_sn = weight / (sigma + 1e-12)  # Added epsilon for stability

        return weight_sn, u  # u is the updated u_new

    @staticmethod
    def apply(module, name, bound, n_power_iterations=1):
        """
        Applies spectral normalization to a parameter of a module.
        Modifies the module in-place.
        """
        fn = SpectralNorm(name, bound, n_power_iterations)

        weight = getattr(module, name)
        if name in module._parameters:
            del module._parameters[name]

        module.register_parameter(name + '_orig', weight)

        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_(0, 1)

        module.register_buffer(name, weight.data.clone())
        module.register_buffer(name + '_u', u)  # u is registered here

        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input_val):
        """
        Forward pre-hook: called before the module's forward pass.
        Computes and sets the spectrally normalized weight and updates u.
        """
        weight_sn, u_new = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        # Correctly update the buffer 'name_u' by direct assignment or setattr
        setattr(module, self.name + '_u', u_new)  # u_new is the updated u vector from compute_weight


def spectral_norm(module, name='weight', init=True, std=1.0, bound=False, n_power_iterations=1):
    """
    Helper function to apply spectral normalization to a module's specified parameter.
    """
    if init and hasattr(module, name):
        param_to_init = getattr(module, name)
        if isinstance(param_to_init, torch.nn.Parameter):
            nn.init.normal_(param_to_init.data, 0, std)  # Initialize .data for parameters
        else:  # Should not happen if 'name' is a registered parameter
            nn.init.normal_(param_to_init, 0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    if hasattr(module, name):
        SpectralNorm.apply(module, name, bound=bound, n_power_iterations=n_power_iterations)
    return module


class GraphConv(nn.Module):
    """
    Graph Convolutional Layer.
    """

    def __init__(self, in_channels, out_channels, num_edge_type, std, bound=True, add_self=False, n_power_iterations=1):
        super(GraphConv, self).__init__()

        self.add_self = add_self
        if self.add_self:
            linear_node_layer = nn.Linear(in_channels, out_channels)
            self.linear_node = spectral_norm(linear_node_layer, std=std, bound=bound,
                                             n_power_iterations=n_power_iterations)

        linear_edge_layer = nn.Linear(in_channels, out_channels * num_edge_type)
        self.linear_edge = spectral_norm(linear_edge_layer, std=std, bound=bound, n_power_iterations=n_power_iterations)

        self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels

    def forward(self, adj, h):
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
    """

    def __init__(self, n_atom_type, hidden, num_edge_type=3, swish_act=True, depth=2, add_self=False, dropout=0,
                 n_power_iterations=1):
        super(EnergyFunc, self).__init__()

        self.depth = depth
        self.swish_act = swish_act
        self.dropout_p = dropout

        self.graphconv1 = GraphConv(n_atom_type, hidden, num_edge_type, std=1.0, bound=False, add_self=add_self,
                                    n_power_iterations=n_power_iterations)

        self.graphconv_layers = nn.ModuleList(
            GraphConv(hidden, hidden, num_edge_type, std=0.02, bound=True, add_self=add_self,
                      n_power_iterations=n_power_iterations) for _ in range(self.depth)
        )

        final_linear_layer = nn.Linear(hidden, 1, bias=False)  # Paper: E = h_G^T W (no bias)
        # Apply SN to the final linear layer as per paper Appendix D: "SN to all layers"
        self.linear = spectral_norm(final_linear_layer, std=1.0, bound=False, n_power_iterations=n_power_iterations)

    def forward(self, adj, h):
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

                # Paper Section 2.1 Eq.3: "sum operation to compute the graph-level representation"
        out = out.sum(dim=1)
        out = self.linear(out)
        return out
