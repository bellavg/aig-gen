import torch


def rescale_adj(adj_dequantized, epsilon=1e-9):
    r"""
    Normalizes the dequantized positive adjacency tensor A' according to Equation 9
    of the GraphEBM paper: A_{(:,:,k)}^{\oplus} = D^{-1}A_{(:,:,k)}^{\prime},
    where D is the diagonal degree matrix D_{(i,i)} = \sum_{j,k} A'_{(i,j,k)}.

    This is a row-normalization applied to each edge-type slice of the adjacency
    tensor, using a degree matrix D calculated by summing connections across all
    edge types for each node.

    Args:
        adj_dequantized (torch.Tensor): The dequantized adjacency tensor A' of shape
                                       (batch_size, num_edge_types, num_nodes, num_nodes).
        epsilon (float): A small value to add to the denominator for numerical stability.

    Returns:
        torch.Tensor: The normalized adjacency tensor A^{\oplus}.
    """
    # adj_dequantized has shape (batch_size, num_edge_types, num_nodes, num_nodes)

    # Calculate D_{(i,i)} = \sum_{j,k} A'_{(i,j,k)}
    # Sum over target nodes (dim 3) and edge types (dim 1) to get degree for each source node i
    # Keep num_nodes dimension for D.
    # degrees will have shape (batch_size, num_nodes)
    degrees = adj_dequantized.sum(dim=(1, 3))  # Sum over edge types and target nodes j

    # Invert degrees to get D^{-1} values.
    # D_inv_diag will have shape (batch_size, num_nodes)
    D_inv_diag = degrees.pow(-1)
    D_inv_diag[D_inv_diag == float('inf')] = 0  # Handle nodes with zero degree

    # To perform D^{-1}A', we need to expand D_inv_diag to match dimensions for broadcasting.
    # D_inv_diag needs to multiply rows of A'.
    # A' is (B, E, N_src, N_tgt). We want to scale rows N_src.
    # D_inv_diag is (B, N_src). Unsqueeze to (B, 1, N_src, 1)
    D_inv_expanded = D_inv_diag.unsqueeze(1).unsqueeze(-1)  # Shape: (batch_size, 1, num_nodes, 1)

    # Apply normalization: A^{\oplus}_{:,:,k} = D^{-1} A'_{:,:,k}
    # Broadcasting D_inv_expanded will effectively multiply each row of A' by the corresponding D_inv_diag value.
    adj_normalized = D_inv_expanded * adj_dequantized

    return adj_normalized


def requires_grad(parameters, flag=True):
    """
    Sets the requires_grad attribute for a list of model parameters.

    Args:
        parameters (iterable): An iterable of torch.Tensor (model parameters).
        flag (bool): The boolean value to set for requires_grad.
    """
    for p in parameters:
        p.requires_grad = flag


def clip_grad(optimizer, grad_clip_norm_type=2, grad_clip_value=None):
    """
    Clips gradients of the optimizer's parameters.
    This specific implementation is based on a common PyTorch recipe if grad_clip_value is set,
    or can be adapted for norm clipping.
    The GraphEBM paper (Appendix D) mentions clipping gradients for Langevin Dynamics samples,
    which is different from this function (this one clips model parameter gradients).
    This function is a general utility for training stability if desired for model parameters.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose parameter gradients are to be clipped.
        grad_clip_norm_type (float or int): Type of the p-norm to use for norm clipping. Default is 2 (L2 norm).
                                           Only used if grad_clip_value is for norm clipping.
        grad_clip_value (float, optional): Maximum norm of the gradients. If None, this specific
                                          clipping logic (based on Adam's exp_avg_sq) might not be
                                          what's intended or a standard torch.nn.utils.clip_grad_norm_
                                          should be used instead. The original code had a custom logic.
                                          For simplicity and standard practice, let's use torch.nn.utils.clip_grad_norm_
                                          if a value is provided.
                                          The original custom clipping logic from the user's code is preserved below
                                          if grad_clip_value is specifically meant for that.
    """
    if grad_clip_value is not None:
        # Standard PyTorch gradient norm clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=(p for group in optimizer.param_groups for p in group['params'] if p.grad is not None),
            max_norm=grad_clip_value,
            norm_type=grad_clip_norm_type
        )
    # The custom clipping logic from the original user code:
    # This type of clipping is very specific and tied to Adam's internal state.
    # It's generally less common than standard norm or value clipping.
    # Keeping it commented out unless explicitly requested or if grad_clip_value is intended for this.
    # else:
    #     # This is the custom clipping logic from the initial user code.
    #     # It relies on Adam optimizer's internal state.
    #     with torch.no_grad():
    #         for group in optimizer.param_groups:
    #             for p in group['params']:
    #                 if p.grad is None:
    #                     continue
    #                 state = optimizer.state[p]

    #                 # Check if Adam's state 'step' and 'exp_avg_sq' are initialized
    #                 if 'step' not in state or state['step'] < 1 or 'exp_avg_sq' not in state:
    #                     # Fallback to simple value clipping if Adam state not ready or not Adam
    #                     # This part is an addition for robustness if the custom clip is used.
    #                     # p.grad.data.clamp_(-0.1, 0.1) # Example fallback, adjust as needed
    #                     continue # Or skip clipping for this param

    #                 step = state['step']
    #                 exp_avg_sq = state['exp_avg_sq']
    #                 _, beta2 = group['betas'] # Assumes Adam optimizer

    #                 bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
    #                 p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))

