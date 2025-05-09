import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming MaskedGraphAF is in .graphaf (i.e., a sibling file in the 'model' directory)
from .graphaf import MaskedGraphAF

# --- AIG Configuration Import ---
# Attempt to import AIG configuration.
# This assumes aig_config.py might be in different locations relative to this model file.
# Common locations:
# 1. Sibling 'data' directory to the 'GraphAF' package root (e.g., project_root/data/aig_config.py)
# 2. Inside the 'GraphAF' package itself (e.g., GraphAF/aig_config.py)
# 3. Directly alongside this graphflow.py file (GraphAF/model/aig_config.py)

AIG_NODE_TYPE_KEYS = None
AIG_EDGE_TYPE_KEYS = None
NODE_CONST0_STR, NODE_PI_STR, NODE_AND_STR, NODE_PO_STR = "NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"  # Defaults

try:
    # Path assuming graphflow.py is in GraphAF/model/ and aig_config.py is in GraphAF/
    from .. import aig_config as aig_config_module_pkg

    AIG_NODE_TYPE_KEYS = aig_config_module_pkg.NODE_TYPE_KEYS
    AIG_EDGE_TYPE_KEYS = aig_config_module_pkg.EDGE_TYPE_KEYS
    NODE_CONST0_STR = getattr(aig_config_module_pkg, 'NODE_CONST0_STR', NODE_CONST0_STR)
    NODE_PI_STR = getattr(aig_config_module_pkg, 'NODE_PI_STR', NODE_PI_STR)
    NODE_AND_STR = getattr(aig_config_module_pkg, 'NODE_AND_STR', NODE_AND_STR)
    NODE_PO_STR = getattr(aig_config_module_pkg, 'NODE_PO_STR', NODE_PO_STR)
    print("Successfully imported AIG config from package level (GraphAF/aig_config.py) for GraphFlowModel.")
except ImportError:
    try:
        # Path assuming graphflow.py is in GraphAF/model/ and aig_config.py is in GraphAF/model/
        from . import aig_config as aig_config_module_local

        AIG_NODE_TYPE_KEYS = aig_config_module_local.NODE_TYPE_KEYS
        AIG_EDGE_TYPE_KEYS = aig_config_module_local.EDGE_TYPE_KEYS
        NODE_CONST0_STR = getattr(aig_config_module_local, 'NODE_CONST0_STR', NODE_CONST0_STR)
        NODE_PI_STR = getattr(aig_config_module_local, 'NODE_PI_STR', NODE_PI_STR)
        NODE_AND_STR = getattr(aig_config_module_local, 'NODE_AND_STR', NODE_AND_STR)
        NODE_PO_STR = getattr(aig_config_module_local, 'NODE_PO_STR', NODE_PO_STR)
        print(
            "Successfully imported AIG config from local model directory (GraphAF/model/aig_config.py) for GraphFlowModel.")
    except ImportError:
        warnings.warn(
            "GraphFlowModel: Could not import AIG configuration. Using default AIG type keys and strings. "
            "Ensure 'aig_config.py' is accessible (e.g., in 'GraphAF/' or 'GraphAF/model/')."
        )

# Fallback default AIG type keys if import fails
if AIG_NODE_TYPE_KEYS is None:
    AIG_NODE_TYPE_KEYS = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO']
if AIG_EDGE_TYPE_KEYS is None:
    AIG_EDGE_TYPE_KEYS = ['EDGE_REG', 'EDGE_INV']  # For NetworkX output; model uses bond_dim channels

# Get AIG node type indices safely
try:
    NODE_CONST0_IDX = AIG_NODE_TYPE_KEYS.index(NODE_CONST0_STR)
    NODE_PI_IDX = AIG_NODE_TYPE_KEYS.index(NODE_PI_STR)
    NODE_AND_IDX = AIG_NODE_TYPE_KEYS.index(NODE_AND_STR)
    NODE_PO_IDX = AIG_NODE_TYPE_KEYS.index(NODE_PO_STR)
except (ValueError, AttributeError) as e:
    warnings.warn(
        f"GraphFlowModel: Error accessing AIG node indices from AIG_NODE_TYPE_KEYS ('{AIG_NODE_TYPE_KEYS}'). Using default indices 0,1,2,3. Error: {e}")
    NODE_CONST0_IDX, NODE_PI_IDX, NODE_AND_IDX, NODE_PO_IDX = 0, 1, 2, 3

# AIG Constraints (can be overridden by aig_config if defined there)
MAX_PI_COUNT = getattr(aig_config_module_pkg if 'aig_config_module_pkg' in locals() else (
    aig_config_module_local if 'aig_config_module_local' in locals() else object), 'MAX_PI_COUNT', 8)
MAX_PO_COUNT = getattr(aig_config_module_pkg if 'aig_config_module_pkg' in locals() else (
    aig_config_module_local if 'aig_config_module_local' in locals() else object), 'MAX_PO_COUNT', 8)

print(f"GraphFlowModel using AIG Node Types: {AIG_NODE_TYPE_KEYS}")
print(f"GraphFlowModel using AIG Edge Types (for output): {AIG_EDGE_TYPE_KEYS}")
print(f"GraphFlowModel Node Indices: CONST0={NODE_CONST0_IDX}, PI={NODE_PI_IDX}, AND={NODE_AND_IDX}, PO={NODE_PO_IDX}")
print(f"GraphFlowModel Constraints: MAX_PI={MAX_PI_COUNT}, MAX_PO={MAX_PO_COUNT}")


# --- End AIG Configuration Import ---


class GraphFlowModel(nn.Module):
    def __init__(self, model_conf_dict):
        super(GraphFlowModel, self).__init__()
        self.max_size = model_conf_dict['max_size']
        self.edge_unroll = model_conf_dict.get('edge_unroll', 12)
        self.node_dim = model_conf_dict['node_dim']  # Expected 4 for AIG
        self.bond_dim = model_conf_dict['bond_dim']  # Expected 3 for AIG (REG, INV, NO_EDGE channels)
        self.deq_coeff = model_conf_dict.get('deq_coeff', 0.9)

        # Store AIG specific indices and counts for generation
        self.aig_node_type_keys = AIG_NODE_TYPE_KEYS
        self.node_const0_idx = NODE_CONST0_IDX
        self.node_pi_idx = NODE_PI_IDX
        self.node_and_idx = NODE_AND_IDX
        self.node_po_idx = NODE_PO_IDX
        self.max_pi_count = MAX_PI_COUNT
        self.max_po_count = MAX_PO_COUNT
        # NO_EDGE category index is assumed to be the last channel in bond_dim
        self.no_edge_category_idx = self.bond_dim - 1 if self.bond_dim > 0 else -1

        node_masks, adj_masks, link_prediction_index, self.flow_core_edge_masks = \
            self.initialize_masks(max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)

        self.latent_step = node_masks.size(0)
        self.latent_node_length = self.max_size * self.node_dim
        self.latent_edge_length = (self.latent_step - self.max_size) * self.bond_dim

        self.use_gpu = model_conf_dict.get('use_gpu', False)
        target_device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")

        self.flow_core = MaskedGraphAF(
            mask_node=node_masks,
            mask_edge=adj_masks,
            index_select_edge=link_prediction_index,
            st_type=model_conf_dict.get('st_type', 'sigmoid'),
            num_flow_layer=model_conf_dict.get('num_flow_layer', 12),
            graph_size=self.max_size,
            num_node_type=self.node_dim,
            num_edge_type=self.bond_dim,  # MaskedGraphAF uses this for its ST-Nets and RGCN edge_dim
            num_rgcn_layer=model_conf_dict.get('num_rgcn_layer', 3),
            nhid=model_conf_dict.get('nhid', 128),
            nout=model_conf_dict.get('nout', 128)
        )

        constant_pi = torch.Tensor([np.pi])  # Using np.pi for better precision
        prior_ln_var = torch.zeros([1])  # Prior variance is 1

        if self.use_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"GraphFlowModel: Using nn.DataParallel for flow_core across {torch.cuda.device_count()} GPUs.")
            self.flow_core = nn.DataParallel(self.flow_core)

        self.flow_core.to(target_device)
        self.constant_pi = nn.Parameter(constant_pi.to(target_device), requires_grad=False)
        self.prior_ln_var = nn.Parameter(prior_ln_var.to(target_device), requires_grad=False)

        print(f"GraphFlowModel initialized on device: {target_device}")
        if isinstance(self.flow_core, nn.DataParallel):
            print(f"Flow core (MaskedGraphAF) is wrapped with DataParallel.")

    def forward(self, inp_node_features, inp_adj_features):
        """
        Forward pass for training.
        Args:
            inp_node_features (torch.Tensor): Input node features (Batch, N_max, NodeDim).
                                             For AIGs, these are typically one-hot.
            inp_adj_features (torch.Tensor): Input adjacency features (Batch, BondDim, N_max, N_max).
                                            BondDim channels (e.g., REG, INV, NO_EDGE), one-hot.
        Returns:
            list: Transformed latent representations [z_node, z_edge].
            list: Log determinants [logdet_node, logdet_edge].
        """
        current_device = inp_node_features.device

        # Dequantization: Add noise to one-hot features to make them continuous for the flow
        inp_node_features_cont = inp_node_features.clone().float()  # Ensure float
        inp_node_features_cont += self.deq_coeff * torch.rand_like(inp_node_features_cont)

        # Select and reshape adjacency features for edge flow steps
        # self.flow_core_edge_masks should select the edges in the autoregressive order
        if hasattr(self.flow_core_edge_masks, 'dtype') and self.flow_core_edge_masks.dtype != torch.bool:
            flow_core_edge_masks_bool = self.flow_core_edge_masks.bool()
        else:
            flow_core_edge_masks_bool = self.flow_core_edge_masks

        # inp_adj_features: (B, BondDim, N, N)
        # flow_core_edge_masks_bool: (N, N), true for edges in the schedule
        # We need to select for each channel in BondDim
        # Correct selection: inp_adj_features[:, :, flow_core_edge_masks_bool]
        # This gives (B, BondDim, NumEdgesInSchedule)
        inp_adj_features_cont = inp_adj_features[:, :, flow_core_edge_masks_bool].clone().float()
        inp_adj_features_cont = inp_adj_features_cont.permute(0, 2, 1).contiguous()  # (B, NumEdgesInSchedule, BondDim)
        inp_adj_features_cont += self.deq_coeff * torch.rand_like(inp_adj_features_cont)

        # Pass through the flow core (MaskedGraphAF)
        # MaskedGraphAF expects original x, adj for RGCN context, and dequantized versions for flow transformation
        z, logdet = self.flow_core(inp_node_features.float(), inp_adj_features.float(),
                                   inp_node_features_cont, inp_adj_features_cont)
        return z, logdet

    def _check_aig_edge_validity(self, u_idx, v_idx, proposed_aig_edge_type_idx,  # 0 for REG, 1 for INV, -1 for NO_EDGE
                                 current_node_types_dict, current_in_degrees_dict):
        """
        Checks if adding a specific AIG edge type between u_idx and v_idx is valid.
        Args:
            u_idx (int): Source node index.
            v_idx (int): Target node index.
            proposed_aig_edge_type_idx (int): 0 for 'EDGE_REG', 1 for 'EDGE_INV'. -1 if checking NO_EDGE.
            current_node_types_dict (dict): Maps node index to its type string (e.g., "NODE_PI").
            current_in_degrees_dict (defaultdict): Maps node index to its current in-degree.
        Returns:
            bool: True if adding the edge is valid, False otherwise.
        """
        if proposed_aig_edge_type_idx == -1:  # NO_EDGE is always "valid" in terms of graph rules here
            return True

        source_type_str = current_node_types_dict.get(u_idx)
        target_type_str = current_node_types_dict.get(v_idx)

        if source_type_str is None or target_type_str is None:
            warnings.warn(f"Edge validity check: Node type not found for u={u_idx} or v={v_idx}. Assuming invalid.")
            return False  # Cannot determine validity if types are unknown

        # Rule 1: Source node cannot be a PO (Primary Output)
        if source_type_str == NODE_PO_STR:
            return False

        # Rule 2: Target node cannot be PI (Primary Input) or CONST0
        if target_type_str == NODE_PI_STR or target_type_str == NODE_CONST0_STR:
            return False

        # Rule 3: AND gate (target) cannot have more than 2 inputs
        if target_type_str == NODE_AND_STR:
            if current_in_degrees_dict.get(v_idx, 0) >= 2:
                return False

        # Rule 4: PO gate (target) cannot have more than 1 input
        if target_type_str == NODE_PO_STR:
            if current_in_degrees_dict.get(v_idx, 0) >= 1:
                return False

        # Add more AIG specific rules if needed (e.g., no direct edge from CONST0 to PO, etc.)

        return True

    def generate_aig_raw_data(self, max_nodes, temperature, device, disconnection_patience=5):
        """
        Generates raw node features (continuous scores) and a list of typed AIG edges.
        Incorporates AIG-specific constraints during generation.

        Args:
            max_nodes (int): Max number of nodes (model's capacity, self.max_size).
            temperature (float): Sampling temperature for the prior.
            device (torch.device): Device for generation.
            disconnection_patience (int): Max consecutive nodes added without connection before stopping.

        Returns:
            tuple:
                - raw_node_features_scores (torch.Tensor): Shape (max_nodes, self.node_dim). Continuous scores.
                - typed_edges_generated (list): List of (source, target, actual_aig_edge_type_idx)
                                                where actual_aig_edge_type_idx is 0 for 'EDGE_REG', 1 for 'EDGE_INV'.
                - actual_num_nodes (int): Actual number of nodes generated.
        """
        if temperature <= 1e-9: temperature = 1e-9  # Prevent zero temperature issues

        with torch.no_grad():
            flow_core_model = self.flow_core.module if isinstance(self.flow_core, nn.DataParallel) else self.flow_core

            # Priors for latent space sampling (Gaussian)
            prior_node_dist = torch.distributions.normal.Normal(
                torch.zeros([self.node_dim], device=device),  # Mean 0
                temperature * torch.ones([self.node_dim], device=device)  # Stddev = temperature
            )
            prior_edge_dist = torch.distributions.normal.Normal(
                torch.zeros([self.bond_dim], device=device),  # Mean 0
                temperature * torch.ones([self.bond_dim], device=device)  # Stddev = temperature
            )

            # Initialize generation state
            # cur_node_features_scores: stores the *continuous scores* from the flow model
            cur_node_features_scores = torch.zeros([1, max_nodes, self.node_dim], device=device)
            # cur_adj_features_one_hot: stores the *one-hot representation* of chosen edge categories (REG, INV, NO_EDGE)
            # This is what conditions the MaskedGraphAF.reverse() calls.
            cur_adj_features_one_hot = torch.zeros([1, self.bond_dim, max_nodes, max_nodes], device=device)

            typed_edges_generated = []  # Store (u, v, aig_edge_type_idx: 0 for REG, 1 for INV)

            # AIG specific state tracking
            current_node_types_dict = {}  # node_idx -> node_type_str
            current_in_degrees_dict = defaultdict(int)
            const0_count, pi_count, po_count = 0, 0, 0

            actual_num_nodes = 0
            disconnection_streak = 0  # For early stopping if graph is fragmented

            print(f"--- GraphAF: Starting AIG Generation (Max Nodes: {max_nodes}, Temp: {temperature:.2f}) ---")

            for i in range(max_nodes):  # Node `i` to be generated
                # --- 1. Node Generation for node `i` ---
                latent_node_sample = prior_node_dist.sample().view(1, -1)  # (1, node_dim)

                # Get continuous scores from the reverse flow
                # Note: MaskedGraphAF.reverse expects the *conditioning* x and adj to be one-hot or discrete.
                # However, for generation, the x and adj passed to reverse are the *current state* of generation.
                # cur_node_features_scores contains scores, but MaskedGraphAF's _get_embs_node will use argmax if needed,
                # or if RGCN can handle continuous scores, that's also possible.
                # Let's assume MaskedGraphAF expects discrete conditioning inputs (x, adj) for its RGCN.
                # So, we create a temporary one-hot version of cur_node_features_scores for conditioning.

                temp_cond_node_features = torch.zeros_like(cur_node_features_scores)
                if i > 0:  # Only fill for previously generated nodes
                    # Discretize previously generated node scores for conditioning
                    prev_node_indices = torch.argmax(cur_node_features_scores[0, :i, :], dim=1)
                    temp_cond_node_features[0, torch.arange(i), prev_node_indices] = 1.0

                # generated_node_i_scores are continuous scores from the flow
                generated_node_i_scores = flow_core_model.reverse(
                    x=temp_cond_node_features,  # Pass discretized previous nodes for RGCN context
                    adj=cur_adj_features_one_hot,  # Pass one-hot adj for RGCN context
                    latent=latent_node_sample,
                    mode=0  # Node generation mode
                ).view(-1)  # (node_dim)

                # Apply AIG constraints to the scores *before* argmax
                constrained_node_scores = generated_node_i_scores.clone()
                if const0_count >= 1: constrained_node_scores[self.node_const0_idx] = -float('inf')
                if pi_count >= self.max_pi_count: constrained_node_scores[self.node_pi_idx] = -float('inf')
                if po_count >= self.max_po_count: constrained_node_scores[self.node_po_idx] = -float('inf')

                # If all options are -inf (e.g., max PIs, max POs, and CONST0 already exists)
                if torch.isinf(constrained_node_scores).all():
                    warnings.warn(f"Node {i}: All node types disallowed by constraints. Stopping generation early.")
                    break  # actual_num_nodes remains i (nodes 0 to i-1 were generated)

                chosen_node_type_idx = torch.argmax(constrained_node_scores).item()
                chosen_node_type_str = self.aig_node_type_keys[chosen_node_type_idx]

                # Store the *original, unconstrained scores* for potential use in loss (if this was training)
                # or for a more nuanced representation if needed. For generation, we proceed with the chosen type.
                cur_node_features_scores[0, i, :] = generated_node_i_scores  # Store original scores

                # Update AIG state
                current_node_types_dict[i] = chosen_node_type_str
                if chosen_node_type_idx == self.node_const0_idx:
                    const0_count += 1
                elif chosen_node_type_idx == self.node_pi_idx:
                    pi_count += 1
                elif chosen_node_type_idx == self.node_po_idx:
                    po_count += 1

                actual_num_nodes = i + 1
                node_i_is_connected = (i == 0)  # First node is considered connected to the "graph system"

                # --- 2. Edge Generation for node `i` to previous nodes `j < i` ---
                if i > 0:
                    # Determine which previous nodes to attempt connections with based on edge_unroll
                    start_prev_node_for_edges = max(0, i - self.edge_unroll)
                    for prev_node_idx in range(start_prev_node_for_edges, i):
                        latent_edge_sample = prior_edge_dist.sample().view(1, -1)  # (1, bond_dim)
                        edge_indices_tensor = torch.tensor([[prev_node_idx, i]], device=device).long()

                        # Again, create temporary one-hot node features for conditioning the reverse call
                        # This time, node `i` is also included in the conditioning context.
                        temp_cond_node_features_for_edge = torch.zeros_like(cur_node_features_scores)
                        # Discretize nodes 0 to i (current node included)
                        current_node_indices = torch.argmax(cur_node_features_scores[0, :actual_num_nodes, :], dim=1)
                        temp_cond_node_features_for_edge[0, torch.arange(actual_num_nodes), current_node_indices] = 1.0

                        # generated_edge_scores are continuous scores for each edge category (REG, INV, NO_EDGE)
                        generated_edge_scores = flow_core_model.reverse(
                            x=temp_cond_node_features_for_edge,
                            adj=cur_adj_features_one_hot,
                            latent=latent_edge_sample,
                            mode=1,  # Edge generation mode
                            edge_index=edge_indices_tensor
                        ).view(-1)  # (bond_dim)

                        # Tentatively choose edge category based on highest score
                        tentative_edge_category_idx = torch.argmax(generated_edge_scores).item()

                        # Map this category_idx to AIG edge type (0 for REG, 1 for INV, -1 for NO_EDGE)
                        # This mapping depends on how bond_dim channels are defined.
                        # Assuming: Channel 0 = REG, Channel 1 = INV, Channel 2 (if bond_dim=3) = NO_EDGE
                        proposed_aig_edge_type = -1  # Default to NO_EDGE interpretation
                        if self.bond_dim == 3:  # REG, INV, NO_EDGE
                            if tentative_edge_category_idx == 0:
                                proposed_aig_edge_type = 0  # REG
                            elif tentative_edge_category_idx == 1:
                                proposed_aig_edge_type = 1  # INV
                        elif self.bond_dim == 2:  # REG, INV (NO_EDGE is implicit or handled by low scores for both)
                            # This setup is trickier for explicitly choosing NO_EDGE based on validity.
                            # If a validity check fails, it's harder to force NO_EDGE if it's not a channel.
                            # For now, assume bond_dim=3 is preferred for AIGs with GraphAF for clarity.
                            if tentative_edge_category_idx == 0:
                                proposed_aig_edge_type = 0
                            elif tentative_edge_category_idx == 1:
                                proposed_aig_edge_type = 1
                            warnings.warn(f"GraphAF AIG generation with bond_dim={self.bond_dim}. "
                                          "Validity checks might be less effective if NO_EDGE is not an explicit channel.")
                        else:
                            raise ValueError(f"Unsupported bond_dim ({self.bond_dim}) for AIG edge interpretation.")

                        # Check AIG validity
                        is_valid_aig_edge = self._check_aig_edge_validity(
                            prev_node_idx, i, proposed_aig_edge_type,
                            current_node_types_dict, current_in_degrees_dict
                        )

                        final_chosen_category_idx = tentative_edge_category_idx
                        if proposed_aig_edge_type != -1 and not is_valid_aig_edge:
                            # If a REG/INV edge was proposed but is invalid, try to force NO_EDGE
                            if self.no_edge_category_idx != -1:  # Check if NO_EDGE channel exists
                                final_chosen_category_idx = self.no_edge_category_idx
                                proposed_aig_edge_type = -1  # Reflect that NO_EDGE was chosen
                                # print(f"  Edge ({prev_node_idx}->{i}): Proposed type invalid, forcing NO_EDGE.")
                            else:
                                # No explicit NO_EDGE channel to fall back to. The invalid edge might form.
                                # Or, we could filter this graph post-generation.
                                warnings.warn(f"  Edge ({prev_node_idx}->{i}): Proposed AIG edge type invalid, "
                                              "but no explicit NO_EDGE channel to fall back to. Edge may be invalid.")

                        # Update cur_adj_features_one_hot with the *final chosen category* (REG, INV, or NO_EDGE)
                        # This conditions subsequent generation steps.
                        if 0 <= final_chosen_category_idx < self.bond_dim:
                            # For AIG (directed graph u->v), set adj[u,v] in the chosen channel.
                            # MaskedGraphAF's _get_embs_edge uses index (source, target)
                            # The adj matrix in PyG Data usually has adj[target, source] = 1 for u->v.
                            # Let's assume adj[channel, u, v] for u->v for consistency with typical generation order.
                            # If your MaskedGraphAF or RGCN expects adj[channel, v, u] for u->v, adjust this.
                            cur_adj_features_one_hot[0, final_chosen_category_idx, prev_node_idx, i] = 1.0
                            # If undirected representation was used for adj in training (symmetric):
                            # cur_adj_features_one_hot[0, final_chosen_category_idx, i, prev_node_idx] = 1.0
                        else:
                            warnings.warn(
                                f"Internal error: final_chosen_category_idx {final_chosen_category_idx} out of bounds for bond_dim {self.bond_dim}.")

                        # If an actual AIG edge (REG or INV) was formed and is valid:
                        if proposed_aig_edge_type != -1:  # i.e., not NO_EDGE
                            typed_edges_generated.append((prev_node_idx, i, proposed_aig_edge_type))
                            current_in_degrees_dict[i] += 1
                            node_i_is_connected = True

                # --- 3. Disconnection Patience Check for node `i` ---
                if i > 0 and not node_i_is_connected:
                    disconnection_streak += 1
                    if disconnection_streak >= disconnection_patience:
                        warnings.warn(
                            f"Node {i}: Disconnection patience ({disconnection_patience}) reached. Stopping generation for this graph.")
                        actual_num_nodes = i  # Roll back the last disconnected node (node i was not successfully connected)
                        # Clear its features from the scores tensor to signify it's not part of the final graph
                        cur_node_features_scores[0, i:, :] = 0
                        # Remove from type tracking
                        if i in current_node_types_dict:
                            # Revert counts if node i was added then rolled back
                            rolled_back_type_idx = self.aig_node_type_keys.index(current_node_types_dict[i])
                            if rolled_back_type_idx == self.node_const0_idx:
                                const0_count -= 1
                            elif rolled_back_type_idx == self.node_pi_idx:
                                pi_count -= 1
                            elif rolled_back_type_idx == self.node_po_idx:
                                po_count -= 1
                            del current_node_types_dict[i]
                        if i in current_in_degrees_dict: del current_in_degrees_dict[i]
                        break  # Exit main node generation loop
                elif node_i_is_connected:
                    disconnection_streak = 0  # Reset streak

            # --- End of Node Generation Loop ---

            # Prepare final outputs
            raw_node_features_output_scores = cur_node_features_scores.squeeze(0)[:actual_num_nodes,
                                              :]  # Only generated nodes

            # Filter typed_edges_generated to ensure no edges point to/from nodes beyond actual_num_nodes
            # (This should be handled by disconnection patience, but as a safeguard)
            final_typed_edges = [(u, v, t) for u, v, t in typed_edges_generated if
                                 u < actual_num_nodes and v < actual_num_nodes]

            if actual_num_nodes == 0 and max_nodes > 0:  # Handle case where generation stopped immediately
                warnings.warn("GraphAF generated an empty graph (0 actual nodes).")
                # Return empty tensors/lists of correct type but 0 size for first dim
                raw_node_features_output_scores = torch.empty((0, self.node_dim), device=device)

            # print(f"--- GraphAF: Finished AIG Generation (Actual Nodes: {actual_num_nodes}) ---")
            return raw_node_features_output_scores, final_typed_edges, actual_num_nodes

    def initialize_masks(self, max_node_unroll, max_edge_unroll):
        """
        Initializes masks for the autoregressive generation process.
        This defines the order of node and edge generation.
        The implementation here is standard for GraphAF-like models.
        """
        # Calculate total number of autoregressive steps (num_masks)
        # and number of edge generation steps (num_mask_edge)
        # This calculation depends on the specific autoregressive schedule.
        # The original GraphAF paper/code has a specific way to calculate this.
        # For simplicity, using the DIG library's typical calculation:
        num_masks = int(max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (
                max_node_unroll - max_edge_unroll) * max_edge_unroll)
        num_mask_edge = int(num_masks - max_node_unroll)

        # node_masks1: for node generation steps (conditioning on previous nodes/edges)
        # adj_masks1: for node generation steps (conditioning on previous subgraph structure)
        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).bool()
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).bool()

        # node_masks2: for edge generation steps
        # adj_masks2: for edge generation steps
        node_masks2 = torch.zeros([num_mask_edge, max_node_unroll]).bool()
        adj_masks2 = torch.zeros([num_mask_edge, max_node_unroll, max_node_unroll]).bool()

        # link_prediction_index: stores (source, target) for each edge generation step
        link_prediction_index = torch.zeros([num_mask_edge, 2]).long()

        # flow_core_edge_masks: defines which entries of the adjacency matrix are modeled by the flow
        # This is used in the forward pass to select adj features for the flow.
        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).bool()

        cnt_node_step = 0
        cnt_edge_step = 0
        for i in range(max_node_unroll):  # For each new node 'i'
            # Node generation step for node 'i'
            node_masks1[cnt_node_step, :i] = True  # Condition on nodes 0 to i-1
            adj_masks1[cnt_node_step, :i, :i] = True  # Condition on adj among nodes 0 to i-1
            cnt_node_step += 1

            # Edge generation steps for node 'i'
            num_edges_for_node_i = min(i, max_edge_unroll)  # Number of previous nodes to connect to
            start_prev_node = i - num_edges_for_node_i

            for j in range(num_edges_for_node_i):
                prev_node_to_connect = start_prev_node + j

                # Edge (prev_node_to_connect -> i) generation step
                node_masks2[cnt_edge_step, :i + 1] = True  # Condition on nodes 0 to i (node i is now known)

                # Adjacency mask for this edge step:
                # Start with adj among 0..i-1 (from adj_masks1 of node i's step)
                # Then add edges already generated for node i to prev_nodes < prev_node_to_connect
                if j == 0:
                    adj_masks2[cnt_edge_step] = adj_masks1[cnt_node_step - 1].clone()
                    # Also mark node i itself as present in the conditioning context (e.g. for self-loops if modeled)
                    adj_masks2[cnt_edge_step, i, i] = True  # Or based on specific modeling choices
                else:
                    adj_masks2[cnt_edge_step] = adj_masks2[cnt_edge_step - 1].clone()
                    # Make the previously generated edge for node 'i' visible
                    # Edge was (start_prev_node + j - 1) -> i
                    adj_masks2[cnt_edge_step, start_prev_node + j - 1, i] = True
                    # If symmetric/undirected modeling:
                    # adj_masks2[cnt_edge_step, i, start_prev_node + j - 1] = True

                link_prediction_index[cnt_edge_step, 0] = prev_node_to_connect
                link_prediction_index[cnt_edge_step, 1] = i
                cnt_edge_step += 1

        if not (cnt_node_step == max_node_unroll and cnt_edge_step == num_mask_edge):
            warnings.warn(
                f"Mask initialization count mismatch: Nodes {cnt_node_step}/{max_node_unroll}, Edges {cnt_edge_step}/{num_mask_edge}. "
                "This might indicate an issue with num_masks calculation or loop logic for the given schedule.")

        # Populate flow_core_edge_masks based on link_prediction_index
        # These are the (u,v) pairs that the flow explicitly models edges for.
        for k_idx in range(num_mask_edge):  # Iterate up to actual edges scheduled
            u, v = link_prediction_index[k_idx, 0].item(), link_prediction_index[k_idx, 1].item()
            if 0 <= u < max_node_unroll and 0 <= v < max_node_unroll:  # Bounds check
                flow_core_edge_masks[u, v] = True  # Edge u -> v
            else:
                warnings.warn(
                    f"initialize_masks: Invalid indices ({u},{v}) at step {k_idx} for flow_core_edge_masks. Max size: {max_node_unroll}")

        node_masks = torch.cat((node_masks1, node_masks2), dim=0)
        adj_masks = torch.cat((adj_masks1, adj_masks2), dim=0)

        node_masks_param = nn.Parameter(node_masks, requires_grad=False)
        adj_masks_param = nn.Parameter(adj_masks, requires_grad=False)
        link_prediction_index_param = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks_param = nn.Parameter(flow_core_edge_masks, requires_grad=False)

        return node_masks_param, adj_masks_param, link_prediction_index_param, flow_core_edge_masks_param

    def log_prob(self, z, logdet):
        """
        Calculates the log probability for training (negative log-likelihood loss).
        Args:
            z (list): Latent variables [z_node (B, N*node_dim), z_edge (B, E*bond_dim)].
            logdet (list): Log determinants [logdet_node (B), logdet_edge (B)].
        Returns:
            torch.Tensor: The mean negative log-likelihood loss per dimension.
        """
        logdet_node, logdet_edge = logdet[0], logdet[1]

        # Adjust logdet for dequantization volume
        # log V = D * log(deq_coeff) where D is number of dimensions
        if self.deq_coeff > 1e-9:  # Avoid log(0)
            log_deq_vol_node = self.latent_node_length * np.log(self.deq_coeff)
            log_deq_vol_edge = self.latent_edge_length * np.log(self.deq_coeff)
            logdet_node = logdet_node - log_deq_vol_node
            logdet_edge = logdet_edge - log_deq_vol_edge

        # Log-likelihood of latent z under prior N(0, exp(prior_ln_var))
        # Since prior_ln_var is 0, prior is N(0, I).
        # log p(z_i) = -0.5 * (log(2*pi) + prior_ln_var + z_i^2 / exp(prior_ln_var))
        #            = -0.5 * (log(2*pi) + z_i^2)
        ll_node = -0.5 * (
                    torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[0] ** 2))
        ll_node = ll_node.sum(-1)  # Sum over all node latent dimensions for each batch item -> (B)

        ll_edge = -0.5 * (
                    torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[1] ** 2))
        ll_edge = ll_edge.sum(-1)  # Sum over all edge latent dimensions for each batch item -> (B)

        # Total log-likelihood: log p(x) = log p(z) + log |det(dz/dx)|
        # Note: logdet from flow is log |det(dy/dz)| where y is data, z is latent.
        # So, log p(x) = log p(f(x)) + log |det(df/dx)|.
        # Here, z = f(x), so log p(x) = log p(z) + logdet.
        ll_node += logdet_node
        ll_edge += logdet_edge

        total_latent_dims = self.latent_node_length + self.latent_edge_length
        if total_latent_dims == 0:
            warnings.warn("Total latent dimensions are zero in log_prob. Returning zero loss.")
            return torch.tensor(0.0, device=ll_node.device if hasattr(ll_node, 'device') else torch.device("cpu"),
                                requires_grad=True)

        # Negative log-likelihood, averaged over batch and normalized by total latent dimensions
        loss = -(torch.mean(ll_node + ll_edge) / total_latent_dims)
        return loss

    def dis_log_prob(self, z):
        # This method is typically for discrete normalizing flows.
        # If GraphAF is purely continuous, this might not be the primary loss function.
        # It assumes 'z' contains (dequantized) one-hot vectors and calculates log-probability
        # based on learned categorical base distributions (self.node_base_log_probs, self.edge_base_log_probs).
        # These base_log_probs would need to be added to __init__ if using this loss.

        # If your GraphAF is strictly continuous as per the original paper, this method might be unused or for a hybrid variant.
        # For now, keeping it as it was, but noting its typical use case.
        if not (hasattr(self, 'node_base_log_probs') and hasattr(self, 'edge_base_log_probs')):
            raise NotImplementedError("Discrete log probability (dis_log_prob) is not configured: "
                                      "'node_base_log_probs' or 'edge_base_log_probs' parameters are missing from the model. "
                                      "This loss is for discrete flow variants.")

        z_node_transformed, z_edge_transformed = z  # These are outputs from the forward pass of a discrete flow
        batch_size = z_node_transformed.shape[0]

        # Reshape if necessary, assuming z_node is (B, N, NodeDim) and z_edge is (B, NumEdgesInSchedule, BondDim)
        # Or if they are flattened: (B, N*NodeDim) and (B, NumEdges*BondDim)
        if z_node_transformed.ndim == 2:  # (B, N*NodeDim)
            z_node_transformed = z_node_transformed.view(batch_size, self.max_size, self.node_dim)

        num_mask_edge = self.latent_step - self.max_size  # Number of edge generation steps in the schedule
        if z_edge_transformed.ndim == 2:  # (B, NumEdges*BondDim)
            z_edge_transformed = z_edge_transformed.view(batch_size, num_mask_edge, self.bond_dim)

        # Calculate log probability using learned base distributions
        node_base_log_probs_sm = F.log_softmax(self.node_base_log_probs, dim=-1)  # (N_max, NodeDim)
        # Sum over NodeDim and N_max dimensions
        ll_node = torch.sum(z_node_transformed * node_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))

        edge_base_log_probs_sm = F.log_softmax(self.edge_base_log_probs, dim=-1)  # (NumEdgesInSchedule, BondDim)
        # Sum over BondDim and NumEdgesInSchedule dimensions
        ll_edge = torch.sum(z_edge_transformed * edge_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))

        total_dims = self.latent_node_length + self.latent_edge_length  # Based on one-hot encoding sizes
        if total_dims == 0:
            warnings.warn("Total latent dimensions are zero in dis_log_prob. Returning zero loss.")
            return torch.tensor(0.0, device=ll_node.device if hasattr(ll_node, 'device') else torch.device("cpu"),
                                requires_grad=True)

        loss = -(torch.mean(ll_node + ll_edge) / total_dims)
        return loss

