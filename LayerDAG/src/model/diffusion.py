import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'DiscreteDiffusion',
    'EdgeDiscreteDiffusion'
]


class DiscreteDiffusion(nn.Module):
    def __init__(self,
                 marginal_list,  # List of 1D tensors: marginal_list[d] is p(x_0) for d-th attr
                 T,
                 s=0.008):
        """
        Parameters
        ----------
        marginal_list : list of torch.Tensor
            marginal_list[d] is the marginal distribution of the d-th attribute.
            Each tensor should be of shape (num_classes_d,).
        s : float
            Constant in noise schedule.
        """
        super().__init__()

        if not isinstance(marginal_list, list):
            marginal_list = [marginal_list]

        self.num_classes_list = []
        self.I_list = nn.ParameterList([])
        self.m_list = nn.ParameterList([])  # Stores M_d matrices

        for marginal_d_p0 in marginal_list:
            if marginal_d_p0.ndim != 1:
                raise ValueError("Each marginal in marginal_list should be a 1D tensor.")
            num_classes_d = len(marginal_d_p0)
            self.num_classes_list.append(num_classes_d)
            self.I_list.append(nn.Parameter(
                torch.eye(num_classes_d), requires_grad=False))

            # m_list[d] should be the transition matrix M_d where rows are all p(x_0) for dim d
            # marginal_d_p0 is p(x_0) of shape (K_d)
            # M_d should be (K_d, K_d) where each row is marginal_d_p0
            m_d_matrix = marginal_d_p0.unsqueeze(0).expand(num_classes_d, -1).clone()
            self.m_list.append(nn.Parameter(m_d_matrix, requires_grad=False))

        self.T = T
        # Cosine schedule
        num_steps = T + 2  # Original LayerDAG uses T+2 for linspace
        t_range = np.linspace(0, num_steps, num_steps)
        alpha_bars = np.cos(0.5 * np.pi * ((t_range / num_steps) + s) / (1 + s)) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]  # Normalize to start at 1

        # Original LayerDAG derives alphas from alpha_bars then betas from alphas
        # Then clamps alphas. Let's follow that logic.
        alphas = alpha_bars[1:] / alpha_bars[:-1]
        betas = 1. - alphas

        # Ensure parameters are float32 and on default device initially
        self.betas = torch.from_numpy(betas).float()
        # Clamp alphas to prevent issues with log(0) or instability
        self.alphas = 1. - torch.clamp(self.betas, min=0., max=0.9999)

        log_alphas = torch.log(self.alphas)
        log_alpha_bars = torch.cumsum(log_alphas, dim=0)
        self.alpha_bars = torch.exp(log_alpha_bars)

        # Register as non-trainable parameters
        self.betas = nn.Parameter(self.betas, requires_grad=False)
        self.alphas = nn.Parameter(self.alphas, requires_grad=False)
        self.alpha_bars = nn.Parameter(self.alpha_bars, requires_grad=False)

    def get_Q(self, alpha_val, d_idx):
        """
        Constructs the transition matrix Q_d = alpha * I_d + (1-alpha) * M_d.
        alpha_val: scalar float for alpha.
        d_idx: attribute dimension index.
        Returns: (K_d, K_d) transition matrix.
        """
        # Ensure alpha_val is a scalar tensor or float for broadcasting
        if isinstance(alpha_val, torch.Tensor) and alpha_val.ndim > 0:
            alpha_val = alpha_val.item()  # Ensure scalar for this formulation

        Q_d = alpha_val * self.I_list[d_idx] + (1. - alpha_val) * self.m_list[d_idx]
        return Q_d

    def apply_noise(self, z_0, t_steps=None):
        """
        Applies noise to the clean data z_0 for a given timestep t_steps.
        z_0: clean data, (N, D) or (N,) tensor of categorical indices.
        t_steps: scalar tensor or int, timestep. If None, samples uniformly.
        Returns: t_steps (sampled or given), z_t (noised data).
        """
        target_device = self.alpha_bars.device  # Use device of registered buffers
        z_0 = z_0.to(target_device)

        if t_steps is None:
            # Sample a timestep t uniformly from 0 to self.T (inclusive for alpha_bars indexing)
            t_steps = torch.randint(low=0, high=self.T + 1, size=(1,), device=target_device)

        if isinstance(t_steps, torch.Tensor) and t_steps.ndim > 0:
            t_idx = t_steps.item()  # For indexing alpha_bars
        else:
            t_idx = t_steps

        alpha_bar_t = self.alpha_bars[t_idx]  # Scalar alpha_bar_t for this step

        if z_0.ndim == 1:
            z_0 = z_0.unsqueeze(-1)  # Make it (N, 1) if single feature dimension

        N, D = z_0.shape
        z_t_list = []

        for d in range(D):
            Q_bar_t_d = self.get_Q(alpha_bar_t, d)  # (K_d, K_d)
            # z_0[:, d] are categorical indices for feature d
            z_0_one_hot_d = F.one_hot(z_0[:, d], num_classes=self.num_classes_list[d]).float()  # (N, K_d)

            # p(z_t | z_0) = z_0_one_hot @ Q_bar_t
            prob_z_t_d = z_0_one_hot_d @ Q_bar_t_d  # (N, K_d)

            # Sample z_t from the categorical distribution
            z_t_d = torch.multinomial(prob_z_t_d, num_samples=1).squeeze(-1)  # (N,)
            z_t_list.append(z_t_d)

        z_t = torch.stack(z_t_list, dim=1) if D > 1 else z_t_list[0].unsqueeze(-1)

        return t_steps, z_t

    def posterior(self, Z_t, Q_t, Q_bar_s, Q_bar_t, Z_0):
        """
        Calculates the posterior probability p(x_s | x_t, x_0_hat) for a single attribute dimension.
        This is used in the reverse diffusion (sampling) process.
        Z_t: (N, K_d) one-hot tensor for current noisy state x_t.
        Q_t: (K_d, K_d) transition matrix for alpha_t.
        Q_bar_s: (K_d, K_d) transition matrix for alpha_bar_s.
        Q_bar_t: (K_d, K_d) transition matrix for alpha_bar_t.
        Z_0: (N, K_d) tensor of predicted probabilities for clean state x_0_hat.
        Returns: (N, K_d) tensor of probabilities for x_s.
        """
        # Formula from DiGress / original DDPM for discrete states:
        # p(x_s | x_t, x_0) propto sum_{x_0_val} Q_t[x_t, x_s] * Q_bar_s[x_s, x_0_val] * P(x_0_val)
        # This can be computed efficiently.
        # Numerator term for each x_s state k: P(x_t | x_s=k) * P(x_s=k | x_0_hat)
        # P(x_t | x_s=k) is Q_t[x_t_observed, k]
        # P(x_s=k | x_0_hat) = sum_{x_0_j} Q_bar_s[k, x_0_j] * P(x_0_j | data)

        # Q_bar_s @ Z_0.T -> (K_d, K_d) @ (K_d, N) -> (K_d, N)
        # This gives P(x_s | x_0_hat) for each x_s state, for each item in batch N.
        # Transpose to (N, K_d)
        prob_xs_given_x0_hat = (Q_bar_s @ Z_0.T).T  # (N, K_d_xs)

        # Q_t is (K_d_xt, K_d_xs). We need P(x_t_observed | x_s=k)
        # This means selecting the row of Q_t corresponding to observed x_t,
        # which is Z_t (one-hot).
        # Z_t @ Q_t -> (N, K_d_xt) @ (K_d_xt, K_d_xs) -> (N, K_d_xs)
        # This gives P(x_t_observed | x_s) for each possible x_s.
        prob_xt_given_xs = Z_t @ Q_t  # (N, K_d_xs)

        # Posterior p(x_s | x_t, x_0_hat) propto prob_xt_given_xs * prob_xs_given_x0_hat
        unnormalized_posterior_probs = prob_xt_given_xs * prob_xs_given_x0_hat  # (N, K_d_xs)

        # Normalize
        posterior_probs = unnormalized_posterior_probs / (unnormalized_posterior_probs.sum(dim=-1, keepdim=True) + 1e-8)
        return posterior_probs


class EdgeDiscreteDiffusion(nn.Module):
    def __init__(self,
                 avg_in_deg,  # Average in-degree, used for marginal
                 T,
                 s=0.008):
        super().__init__()

        self.avg_in_deg = avg_in_deg  # This is a scalar
        self.T = T

        num_steps = T + 2
        t_range = np.linspace(0, num_steps, num_steps)
        alpha_bars = np.cos(0.5 * np.pi * ((t_range / num_steps) + s) / (1 + s)) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]
        alphas = alpha_bars[1:] / alpha_bars[:-1]
        betas = 1. - alphas

        self.betas = torch.from_numpy(betas).float()
        self.alphas = 1. - torch.clamp(self.betas, min=0., max=0.9999)

        log_alphas = torch.log(self.alphas)
        log_alpha_bars = torch.cumsum(log_alphas, dim=0)
        self.alpha_bars = torch.exp(log_alpha_bars)

        self.betas = nn.Parameter(self.betas, requires_grad=False)
        self.alphas = nn.Parameter(self.alphas, requires_grad=False)
        self.alpha_bars = nn.Parameter(self.alpha_bars, requires_grad=False)

        # For edges, num_classes is always 2 (exists or not)
        self.num_classes = 2
        self.I_edge = nn.Parameter(torch.eye(self.num_classes), requires_grad=False)

    def get_M_edge(self, marginal_p1, device):
        """
        Constructs the M matrix for edges given marginal_p1 (prob of edge existing).
        M_edge = [[1-p1, p1], [1-p1, p1]]
        """
        if isinstance(marginal_p1, torch.Tensor):
            marginal_p1 = marginal_p1.item()  # Ensure scalar
        m_row = torch.tensor([1. - marginal_p1, marginal_p1], device=device)
        M_edge = m_row.unsqueeze(0).expand(self.num_classes, -1)
        return M_edge

    def get_Qs_edge(self, alpha_val, marginal_p1, device):
        """
        Constructs Q_t, Q_bar_s, Q_bar_t for edges.
        alpha_val: scalar or tensor of alpha values.
        marginal_p1: scalar marginal probability of edge existence.
        Returns Q matrix (2,2).
        """
        M_edge = self.get_M_edge(marginal_p1, device)
        if isinstance(alpha_val, torch.Tensor) and alpha_val.ndim > 0:
            # If alpha_val is batched (e.g. per query), Q will be (B, 2, 2)
            # This requires careful handling if Q matrices are expected to be (2,2) by posterior.
            # For now, assume alpha_val is scalar for a single Q matrix.
            # The original LayerDAG sample_edge_layer implies scalar alphas for get_Qs.
            alpha_val = alpha_val.item()

        Q = alpha_val * self.I_edge + (1. - alpha_val) * M_edge
        return Q

    def apply_noise(self, z_0_adj, t_steps=None, num_candidate_sources_for_marginal=None):
        """
        Applies noise to a batch of edge adjacency matrices z_0_adj.
        z_0_adj : torch.Tensor of shape (NumQueries) or (NumDstNodes, NumSrcNodes)
                  Represents clean edge states (0 or 1).
                  If flat (NumQueries), it's assumed to be reshaped correctly by caller if needed.
                  This method expects it to be effectively (NumDstNodes, NumSrcNodes) for marginal calc.
        t_steps : Timestep. If None, samples uniformly.
        num_candidate_sources_for_marginal: If z_0_adj is already flat, this is needed
                                            to compute a meaningful marginal.
                                            Typically z_0_adj.shape[1] if it's a matrix.
        Returns: t_steps, z_t (noised edge states, same shape as input z_0_adj or flattened).
        """
        target_device = self.alpha_bars.device
        z_0_adj = z_0_adj.to(target_device).float()  # Ensure float for bernoulli

        if t_steps is None:
            t_steps = torch.randint(low=0, high=self.T + 1, size=(1,), device=target_device)

        t_idx = t_steps.item() if isinstance(t_steps, torch.Tensor) else t_steps
        alpha_bar_t = self.alpha_bars[t_idx]

        # Marginal probability for an edge to exist.
        # This needs careful definition. If z_0_adj is (NumDst, NumSrc), then z_0_adj.shape[1] is num_candidate_sources.
        # If z_0_adj is flat, this needs to be passed or estimated.
        _num_candidate_sources = num_candidate_sources_for_marginal
        if z_0_adj.ndim == 2:  # (NumDst, NumSrc)
            _num_candidate_sources = z_0_adj.shape[1]

        if _num_candidate_sources is None or _num_candidate_sources == 0:
            # Fallback if num_candidate_sources cannot be determined or is zero
            # This can happen if a new layer has no potential sources.
            # In such a case, the marginal probability of an edge should be 0.
            marginal_p1 = 0.0
        else:
            mean_in_deg_effective = min(self.avg_in_deg, _num_candidate_sources)
            marginal_p1 = mean_in_deg_effective / _num_candidate_sources

        marginal_p1 = torch.clamp(torch.tensor(marginal_p1), 0.0, 1.0)  # Ensure valid prob

        # m_z_t is the prior p(z_T) or p(z_0) depending on interpretation.
        # For q(z_t | z_0), M_edge's rows are p(z_0_edge_state)
        # So m_z_t should be a tensor of shape like z_0_adj, filled with marginal_p1.
        m_z_t_val = marginal_p1  # Scalar prob of edge existing

        # prob_z_t = alpha_bar_t * z_0_adj + (1 - alpha_bar_t) * M_edge[z_0_adj_int, 1]
        # More directly: p(z_t=1|z_0) = alpha_bar_t * z_0 + (1-alpha_bar_t) * p1
        #              p(z_t=0|z_0) = alpha_bar_t * (1-z_0) + (1-alpha_bar_t) * p0
        # So, sample z_t=1 with probability p(z_t=1|z_0)
        prob_z_t_is_1 = alpha_bar_t * z_0_adj + (1. - alpha_bar_t) * m_z_t_val
        z_t = torch.bernoulli(prob_z_t_is_1)

        # Optional: Heuristic to ensure connectivity (matches original user's version)
        # This is applied if z_0_adj was passed as a 2D matrix.
        if z_0_adj.ndim == 2 and z_t.ndim == 2:  # Only if z_t is still 2D
            isolated_mask = (z_t.sum(dim=1) == 0).bool()  # Check for rows (dest nodes) with no incoming edges
            if isolated_mask.any():
                # For isolated dest nodes, pick the edge with highest probability from prob_z_t_is_1
                # This requires prob_z_t_is_1 to be 2D.
                if prob_z_t_is_1.ndim == 2:
                    # Get argmax only for relevant rows
                    argmax_indices = prob_z_t_is_1[isolated_mask].argmax(dim=1)
                    z_t[isolated_mask, argmax_indices] = 1.
                # If prob_z_t_is_1 is not 2D (e.g. scalar if z_0_adj was scalar), this heuristic is harder to apply.
                # For now, assume it's 2D if z_t is 2D.

        # The original LayerDAGEdgePredDataset flattens label_t after this.
        # So, if the input z_0_adj was 2D, the output z_t should also be flattened by the caller if needed.
        # This function will return z_t with the same shape it had after bernoulli sampling.
        return t_steps, z_t

    def posterior(self, Z_t_one_hot, Q_t, Q_bar_s, Q_bar_t, Z_0_probs):
        """
        Calculates the posterior probability p(edge_s | edge_t, edge_0_hat) for edges.
        Z_t_one_hot: (NumQueries, 2) one-hot tensor for current noisy edge state edge_t.
        Q_t, Q_bar_s, Q_bar_t: (2, 2) transition matrices.
        Z_0_probs: (NumQueries, 2) tensor of predicted probabilities for clean edge state edge_0_hat.
        Returns: (NumQueries, 2) tensor of probabilities for edge_s.
        """
        # This is identical to DiscreteDiffusion.posterior as it's a general formula
        # for discrete states, specialized here for K=2 classes.
        prob_xs_given_x0_hat = (Q_bar_s @ Z_0_probs.T).T
        prob_xt_given_xs = Z_t_one_hot @ Q_t
        unnormalized_posterior_probs = prob_xt_given_xs * prob_xs_given_x0_hat
        posterior_probs = unnormalized_posterior_probs / (unnormalized_posterior_probs.sum(dim=-1, keepdim=True) + 1e-8)
        return posterior_probs

