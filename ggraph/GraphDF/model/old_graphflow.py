import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import warnings # Added for warnings
from .disgraphaf import DisGraphAF
# Assuming DisGraphAF is defined in .disgraphaf
try:

except ImportError:
    print("Warning: DisGraphAF not found directly. Ensure it's correctly placed for GraphDF's GraphFlowModel.")
    class DisGraphAF(nn.Module): # Placeholder
        def __init__(self, *args, **kwargs): super().__init__(); raise NotImplementedError("DisGraphAF placeholder used.")
        def forward(self, *args, **kwargs): raise NotImplementedError("DisGraphAF placeholder used.")
        def reverse(self, *args, **kwargs): raise NotImplementedError("DisGraphAF placeholder used.")

# Adjust this import path based on your actual project structure
# It assumes aig_config.py is accessible, potentially via relative path
try:
    import aig_config
except ImportError:
     # Try relative import if direct fails (common in package structures)
     # This assumes graphflow.py is inside a 'model' subfolder, and aig_config.py is in the parent 'GraphDF' folder
     try:
         from .. import aig_config
     except ImportError:
         # If relative also fails, try assuming it's in the same directory (less common)
         try:
             import G2PT.configs.aig as aig_config
         except ImportError as e:
              print(f"CRITICAL ERROR: Could not import aig_config. Please ensure aig_config.py is accessible.")
              print(f"Import error details: {e}")
              # Exit or raise a more specific error if config is essential
              raise ImportError("Failed to import AIG configuration.") from e


# --- Constants from AIG Config ---
AIG_NODE_TYPE_KEYS = getattr(aig_config, 'NODE_TYPE_KEYS', ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"])
AIG_EDGE_TYPE_KEYS = getattr(aig_config, 'EDGE_TYPE_KEYS', ["EDGE_REG", "EDGE_INV"])
NODE_CONST0_STR = "NODE_CONST0"
NODE_PI_STR = "NODE_PI"
NODE_AND_STR = "NODE_AND"
NODE_PO_STR = "NODE_PO"

# Get indices from config keys safely
try:
    NODE_CONST0_IDX = AIG_NODE_TYPE_KEYS.index(NODE_CONST0_STR)
    NODE_PI_IDX = AIG_NODE_TYPE_KEYS.index(NODE_PI_STR)
    NODE_AND_IDX = AIG_NODE_TYPE_KEYS.index(NODE_AND_STR)
    NODE_PO_IDX = AIG_NODE_TYPE_KEYS.index(NODE_PO_STR)
    print("Successfully imported AIG type keys and indices from aig_config for GraphFlowModel.")
except (ValueError, AttributeError, NameError) as e: # Catch potential errors if aig_config load failed partially
    print(f"Error accessing AIG node indices from aig_config (Expected keys: {AIG_NODE_TYPE_KEYS}): {e}")
    # Set default indices if config fails, but this might be wrong
    NODE_CONST0_IDX, NODE_PI_IDX, NODE_AND_IDX, NODE_PO_IDX = 0, 1, 2, 3
    print(f"Warning: Using default indices {NODE_CONST0_IDX}, {NODE_PI_IDX}, {NODE_AND_IDX}, {NODE_PO_IDX}")

# Use hardcoded max counts as requested
MAX_PI_COUNT = 8
MAX_PO_COUNT = 8
# --- End Constants ---


class GraphFlowModel(nn.Module):
    """
    GraphFlowModel adapted for And-Inverter Graph (AIG) generation using discrete flows.
    Includes enhanced validity checks during generation.
    """
    def __init__(self, model_conf_dict):
        """
        Initializes the GraphFlowModel.

        Args:
            model_conf_dict (dict): Configuration dictionary containing model parameters
                                     like max_size, node_dim, bond_dim, edge_unroll, etc.
        """
        super(GraphFlowModel, self).__init__()
        self.max_size = model_conf_dict['max_size']
        self.edge_unroll = model_conf_dict.get('edge_unroll', 12) # Max edges to predict per node step
        self.node_dim = model_conf_dict['node_dim'] # Expected 4 for AIG (CONST0, PI, AND, PO)
        self.bond_dim = model_conf_dict['bond_dim'] # Expected 3 for AIG (REG, INV, NO_EDGE)

        print(f"GraphFlowModel __init__: max_size={self.max_size}, edge_unroll={self.edge_unroll}")

        # Initialize masks required for the autoregressive flow core (DisGraphAF)
        node_masks, adj_masks, link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(
            max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)

        # Calculate latent space dimensions based on masks
        self.latent_step = node_masks.size(0) # Total number of generation steps (nodes + edges)
        self.latent_node_length = self.max_size * self.node_dim
        num_mask_edge = int(self.latent_step - self.max_size) # Number of edge generation steps
        self.latent_edge_length = num_mask_edge * self.bond_dim

        print(f"GraphFlowModel __init__: latent_step={self.latent_step}, num_mask_edge={num_mask_edge}")

        self.use_gpu = model_conf_dict.get('use_gpu', False)

        # Initialize base distributions (priors) for nodes and edges
        # These are learned during training
        node_base_log_probs_init = torch.randn(self.max_size, self.node_dim)
        edge_base_log_probs_init = torch.randn(num_mask_edge, self.bond_dim)

        print(f"GraphFlowModel __init__: Initializing edge_base_log_probs with shape: {edge_base_log_probs_init.shape}")

        # Instantiate the discrete flow core (DisGraphAF)
        self.flow_core = DisGraphAF(
            mask_node=node_masks,
            mask_edge=adj_masks,
            index_select_edge=link_prediction_index,
            num_flow_layer=model_conf_dict.get('num_flow_layer', 12),
            graph_size=self.max_size,
            num_node_type=self.node_dim,
            num_edge_type=self.bond_dim, # Pass bond_dim (e.g., 3) to DisGraphAF
            num_rgcn_layer=model_conf_dict.get('num_rgcn_layer', 3),
            nhid=model_conf_dict.get('nhid', 128),
            nout=model_conf_dict.get('nout', 128)
        )

        # Handle device placement and DataParallel if multiple GPUs are available
        if self.use_gpu and torch.cuda.is_available():
            target_device = torch.device("cuda")
            if torch.cuda.device_count() > 1:
                print(f"GraphFlowModel: Using nn.DataParallel for flow_core across {torch.cuda.device_count()} GPUs.")
                self.flow_core = nn.DataParallel(self.flow_core)
            else:
                print("GraphFlowModel: Single GPU detected. Not wrapping flow_core.")
            self.node_base_log_probs = nn.Parameter(node_base_log_probs_init.to(target_device), requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs_init.to(target_device), requires_grad=True)
        else:
            target_device = torch.device("cpu")
            print("GraphFlowModel: Using CPU.")
            self.node_base_log_probs = nn.Parameter(node_base_log_probs_init.to(target_device), requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs_init.to(target_device), requires_grad=True)

        # Move the flow core and base distributions to the target device
        self.flow_core.to(target_device)

    def forward(self, inp_node_features, inp_adj_features):
        """
        Forward pass for training the flow model.

        Args:
            inp_node_features (torch.Tensor): Input node features (Batch, N_max, NodeDim).
            inp_adj_features (torch.Tensor): Input adjacency features (Batch, BondDim, N_max, N_max).

        Returns:
            list: Transformed latent representations [z_node, z_edge].
        """
        # Clone inputs as they might be modified by the flow core
        inp_node_features_cont = inp_node_features.clone()

        # Select the relevant edges based on the precomputed mask for the flow core
        # Ensure the mask is boolean type for indexing
        if hasattr(self.flow_core_edge_masks, 'dtype') and self.flow_core_edge_masks.dtype != torch.bool:
             flow_core_edge_masks_bool = self.flow_core_edge_masks.bool()
        else:
             flow_core_edge_masks_bool = self.flow_core_edge_masks

        # Select and reshape adjacency features for edge flow steps
        inp_adj_features_cont = inp_adj_features[:, :, flow_core_edge_masks_bool].clone() # Shape: [B, BondDim, NumEdgesInFlow]
        inp_adj_features_cont = inp_adj_features_cont.permute(0, 2, 1).contiguous() # Shape: [B, NumEdgesInFlow, BondDim]

        # Pass features through the flow core
        z = self.flow_core(inp_node_features, inp_adj_features, inp_node_features_cont, inp_adj_features_cont)
        return z

    def _check_aig_edge_validity(self, u, v, edge_type_idx, current_node_types, current_in_degrees):
        """
        Checks if adding a specific AIG edge type between u and v is valid
        based on current node types and degrees. Focuses on immediate violations.

        Args:
            u (int): Source node index.
            v (int): Target node index.
            edge_type_idx (int): The proposed AIG edge type (0 for REG, 1 for INV).
                                 -1 indicates checking for a NO_EDGE category.
            current_node_types (dict): Maps node index to type string (e.g., "NODE_PI").
            current_in_degrees (defaultdict): Maps node index to current in-degree count.

        Returns:
            bool: True if adding the edge is valid according to local rules, False otherwise.
        """
        # If checking for NO_EDGE, it's always considered "valid" in this context
        if edge_type_idx == -1:
            return True

        source_type = current_node_types.get(u)
        target_type = current_node_types.get(v)

        # Rule 1: Source node cannot be a PO
        if source_type == NODE_PO_STR:
            return False

        # Rule 2: Target node cannot be PI or CONST0
        if target_type == NODE_PI_STR or target_type == NODE_CONST0_STR:
            return False

        # Rule 3: Target node cannot be AND if it already has 2 inputs
        if target_type == NODE_AND_STR:
            if current_in_degrees.get(v, 0) >= 2:
                return False

        # If none of the above rules failed, the edge is considered locally valid
        return True

    def generate_aig_discrete_raw_data(self, max_nodes, temperature_node, temperature_edge, device,
                                       disconnection_patience=5):
        """
        Generates raw node features (one-hot) and a list of typed AIG edges using discrete sampling
        with enhanced validity checking and resampling for edges.
        Edge generation follows the schedule defined by link_prediction_index.
        Constraints: Exactly 1 CONST0, max 8 PI, max 8 PO.

        Args:
            max_nodes (int): Max number of nodes (model's capacity, self.max_size).
            temperature_node (float): Temperature for node type sampling.
            temperature_edge (float): Temperature for edge type sampling.
            device (torch.device): Device for generation.
            disconnection_patience (int): Max consecutive nodes added without connection before stopping.

        Returns:
            tuple:
                - raw_node_features_one_hot (torch.Tensor): Shape (max_nodes, self.node_dim), one-hot.
                - typed_edges_generated (list): List of (source, target, actual_aig_edge_type_idx)
                                                where actual_aig_edge_type_idx is 0 for 'EDGE_REG', 1 for 'EDGE_INV'.
                - actual_num_nodes (int): Actual number of nodes generated.
        """
        if temperature_node <= 1e-9: temperature_node = 1e-9
        if temperature_edge <= 1e-9: temperature_edge = 1e-9

        with torch.no_grad():
            flow_core_model = self.flow_core.module if isinstance(self.flow_core, nn.DataParallel) else self.flow_core
            link_prediction_schedule = flow_core_model.index_select_edge  # Shape: (num_mask_edge, 2)

            cur_node_features_one_hot = torch.zeros([1, max_nodes, self.node_dim], device=device)
            cur_adj_features_one_hot = torch.zeros([1, self.bond_dim, max_nodes, max_nodes], device=device)
            typed_edges_generated = []
            current_node_types = {}
            current_in_degrees = defaultdict(int)
            const0_count = 0
            pi_count = 0
            po_count = 0

            actual_num_nodes = 0
            edge_step_idx = 0  # Global cursor for the edge schedule and edge_base_log_probs
            disconnection_streak = 0

            edge_base_probs_shape_0 = self.edge_base_log_probs.shape[0]
            max_scheduled_edges = link_prediction_schedule.shape[0]

            if edge_base_probs_shape_0 != max_scheduled_edges:
                warnings.warn(
                    f"Mismatch between edge_base_log_probs size ({edge_base_probs_shape_0}) and link_prediction_schedule size ({max_scheduled_edges}). Using smaller one as limit.")
                # This should ideally not happen if initialize_masks and __init__ are correct
                # as num_mask_edge should be consistent.
                limit_edge_steps = min(edge_base_probs_shape_0, max_scheduled_edges)
            else:
                limit_edge_steps = edge_base_probs_shape_0

            print(f"--- Starting Generation (Original DIG-like Edge Schedule) ---")
            print(f"Max nodes to generate: {max_nodes}")
            print(f"Total scheduled edge steps: {limit_edge_steps}")

            for i in range(max_nodes):  # Iterate through potential node indices (target_node_idx for upcoming edges)
                # --- Node Generation Step for node i ---
                node_logits_prior = self.node_base_log_probs[i] / temperature_node
                prior_node_dist = torch.distributions.Categorical(logits=node_logits_prior)
                latent_node_idx_prior = prior_node_dist.sample()
                latent_node_one_hot_prior = F.one_hot(latent_node_idx_prior,
                                                      num_classes=self.node_dim).float().unsqueeze(0).to(device)

                output_node_logits = flow_core_model.reverse(
                    x_cond_onehot=cur_node_features_one_hot,
                    adj_cond_onehot=cur_adj_features_one_hot,
                    z_onehot=latent_node_one_hot_prior,
                    mode=0
                ).view(-1)

                constrained_node_logits = output_node_logits.clone()
                if const0_count >= 1: constrained_node_logits[NODE_CONST0_IDX] = -float('inf')
                if pi_count >= MAX_PI_COUNT: constrained_node_logits[NODE_PI_IDX] = -float('inf')
                if po_count >= MAX_PO_COUNT: constrained_node_logits[NODE_PO_IDX] = -float('inf')

                if torch.isinf(constrained_node_logits).all():
                    warnings.warn(f"Node {i}: All node types disallowed by constraints. Stopping generation.")
                    # actual_num_nodes is already i from the previous successful node.
                    break

                final_node_type_dist = torch.distributions.Categorical(logits=constrained_node_logits)
                final_node_type_idx = final_node_type_dist.sample().item()

                cur_node_features_one_hot[0, i, final_node_type_idx] = 1.0
                # For AIGs, adjacency diagonal for self-loops is not typically 1 unless it's a special no-edge marker
                # cur_adj_features_one_hot[0, :, i, i] = 1.0 # Original DIG had this, check if needed for AIG
                node_type_str = AIG_NODE_TYPE_KEYS[final_node_type_idx]
                current_node_types[i] = node_type_str
                actual_num_nodes = i + 1  # Node i has been successfully added

                if node_type_str == NODE_CONST0_STR:
                    const0_count += 1
                elif node_type_str == NODE_PI_STR:
                    pi_count += 1
                elif node_type_str == NODE_PO_STR:
                    po_count += 1
                # --- End Node Generation Step ---

                # --- Edge Generation Step: Process scheduled edges targeting the newly added node 'i' ---
                is_node_i_connected = False
                if i > 0:  # Edges only form if there's more than one node
                    # Iterate through the global edge schedule as long as the target of the scheduled edge is current node 'i'
                    while edge_step_idx < limit_edge_steps and \
                            link_prediction_schedule[edge_step_idx, 1].item() == i:

                        current_target_node_in_schedule = link_prediction_schedule[edge_step_idx, 1].item()
                        prev_node_idx = link_prediction_schedule[edge_step_idx, 0].item()

                        # This assertion should hold true due to how link_prediction_index is constructed
                        # (edges always point from an existing node to the new node i)
                        if prev_node_idx >= current_target_node_in_schedule:  # current_target_node_in_schedule is i
                            warnings.warn(
                                f"Scheduled edge from {prev_node_idx} to {current_target_node_in_schedule} (node {i}) "
                                f"is not strictly autoregressive or prev_node_idx is invalid. Skipping. edge_step_idx: {edge_step_idx}")
                            edge_step_idx += 1  # Consume this problematic scheduled edge and move to the next
                            continue

                        # --- Resampling Loop for this specific scheduled edge slot (prev_node_idx -> i) ---
                        valid_edge_category_found = False
                        resamples = 0
                        invalid_cats_tried = set()
                        # Use the prior for the current globally scheduled edge
                        base_logits_slot = self.edge_base_log_probs[edge_step_idx].clone().to(device)

                        target_node_idx_for_reverse = i  # The node we are connecting to

                        while not valid_edge_category_found and resamples < 5 and len(
                                invalid_cats_tried) < self.bond_dim:  # max_resamples_per_edge_slot = 5
                            logits_for_sampling = base_logits_slot.clone()
                            for invalid_cat in invalid_cats_tried: logits_for_sampling[invalid_cat] = -float('inf')
                            if torch.isinf(logits_for_sampling).all():
                                final_edge_cat_idx = self.bond_dim - 1  # Force NO_EDGE
                                break

                            prior_dist = torch.distributions.Categorical(logits=logits_for_sampling / temperature_edge)
                            latent_idx = prior_dist.sample()
                            latent_one_hot = F.one_hot(latent_idx, num_classes=self.bond_dim).float().unsqueeze(0).to(
                                device)

                            edge_indices_for_reverse = torch.tensor([[prev_node_idx, target_node_idx_for_reverse]],
                                                                    device=device).long()
                            output_logits = flow_core_model.reverse(cur_node_features_one_hot, cur_adj_features_one_hot,
                                                                    latent_one_hot, mode=1,
                                                                    edge_index=edge_indices_for_reverse).view(-1)

                            final_logits = output_logits.clone()
                            for invalid_cat in invalid_cats_tried: final_logits[invalid_cat] = -float('inf')
                            if torch.isinf(final_logits).all():
                                final_edge_cat_idx = self.bond_dim - 1  # Force NO_EDGE
                                break

                            final_dist = torch.distributions.Categorical(logits=final_logits)
                            final_edge_cat_idx = final_dist.sample().item()

                            aig_edge_type = -1
                            if final_edge_cat_idx == 0:
                                aig_edge_type = 0  # REG
                            elif final_edge_cat_idx == 1:
                                aig_edge_type = 1  # INV
                            # final_edge_cat_idx == 2 (self.bond_dim - 1) is NO_EDGE

                            is_valid = self._check_aig_edge_validity(prev_node_idx, target_node_idx_for_reverse,
                                                                     aig_edge_type, current_node_types,
                                                                     current_in_degrees)

                            if is_valid:
                                valid_edge_category_found = True
                                cur_adj_features_one_hot[
                                    0, final_edge_cat_idx, target_node_idx_for_reverse, prev_node_idx] = 1.0  # v -> u
                                cur_adj_features_one_hot[
                                    0, final_edge_cat_idx, prev_node_idx, target_node_idx_for_reverse] = 1.0  # u -> v (for undirected representation in adj if needed, but AIGs are directed)
                                # For AIG (directed), we typically only set target_node_idx, prev_node_idx for GNNs that expect u->v as adj[v,u]
                                # Let's assume adj[v,u] = 1 for an edge u->v.
                                # The PyG Data object in your processing script sets adj_matrix[edge_channel_index, v_new, u_new] = 1.0
                                # So, for an edge prev_node_idx -> target_node_idx_for_reverse (i.e. u -> v), we set adj[channel, v, u]
                                cur_adj_features_one_hot[
                                    0, final_edge_cat_idx, target_node_idx_for_reverse, prev_node_idx] = 1.0
                                # If your GNN consumes adj[u,v] for u->v, then it should be:
                                # cur_adj_features_one_hot[0, final_edge_cat_idx, prev_node_idx, target_node_idx_for_reverse] = 1.0
                                # Let's stick to your dataset's convention: (v,u) for u->v
                                # Clear the other direction if it was set for symmetry previously
                                # cur_adj_features_one_hot[0, final_edge_cat_idx, prev_node_idx, target_node_idx_for_reverse] = 0.0

                                if aig_edge_type != -1:  # If it's not NO_EDGE
                                    typed_edges_generated.append(
                                        (prev_node_idx, target_node_idx_for_reverse, aig_edge_type))
                                    current_in_degrees[target_node_idx_for_reverse] += 1
                                    is_node_i_connected = True
                            else:
                                invalid_cats_tried.add(final_edge_cat_idx)
                                resamples += 1
                        # --- End Resampling Loop ---

                        if not valid_edge_category_found:  # Default to NO_EDGE if no valid edge found after resampling
                            no_edge_idx = self.bond_dim - 1
                            cur_adj_features_one_hot[0, no_edge_idx, target_node_idx_for_reverse, prev_node_idx] = 1.0
                            # cur_adj_features_one_hot[0, no_edge_idx, prev_node_idx, target_node_idx_for_reverse] = 1.0 # if symmetric

                        edge_step_idx += 1  # Crucial: move to the next globally scheduled edge
                    # --- End While loop for scheduled edges targeting node i ---

                # --- Disconnection Patience Logic for node i ---
                if i == 0:  # First node is by definition "connected" to the graph structure
                    disconnection_streak = 0
                elif is_node_i_connected:
                    disconnection_streak = 0
                else:  # Node i > 0 and was not connected by any scheduled edge
                    disconnection_streak += 1
                    if disconnection_streak >= disconnection_patience:
                        print(
                            f"Terminating generation early at node {actual_num_nodes} (index {i}) due to disconnection patience ({disconnection_patience}). Node {i} was not connected.")
                        # Roll back the last disconnected node i
                        actual_num_nodes = i
                        cur_node_features_one_hot[0, i, :] = 0.0
                        # Clear any adj entries involving node i (though none should have been made if not connected)
                        cur_adj_features_one_hot[:, :, i, :] = 0.0
                        cur_adj_features_one_hot[:, :, :, i] = 0.0
                        if i in current_node_types: del current_node_types[i]
                        if i in current_in_degrees: del current_in_degrees[i]
                        # Adjust counts if node i was rolled back
                        if node_type_str == NODE_CONST0_STR:
                            const0_count -= 1
                        elif node_type_str == NODE_PI_STR:
                            pi_count -= 1
                        elif node_type_str == NODE_PO_STR:
                            po_count -= 1
                        break  # Exit the main node generation loop (for i)
            # --- End Main Node Loop (for i) ---

            raw_node_features_output_one_hot = cur_node_features_one_hot.squeeze(0)
            return raw_node_features_output_one_hot, typed_edges_generated, actual_num_nodes

    # --- initialize_masks and dis_log_prob ---
    # (Keep these methods exactly as they were in the previous version)
    def initialize_masks(self, max_node_unroll, max_edge_unroll):
        print(f"initialize_masks: max_node_unroll={max_node_unroll}, max_edge_unroll={max_edge_unroll}")
        num_masks = int(max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (max_node_unroll - max_edge_unroll) * max_edge_unroll)
        num_mask_edge = int(num_masks - max_node_unroll)
        print(f"initialize_masks: Calculated num_masks={num_masks}, num_mask_edge={num_mask_edge}")

        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).bool()
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).bool()
        node_masks2 = torch.zeros([num_mask_edge, max_node_unroll]).bool()
        adj_masks2 = torch.zeros([num_mask_edge, max_node_unroll, max_node_unroll]).bool()
        link_prediction_index = torch.zeros([num_mask_edge, 2]).long()
        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).bool()

        cnt, cnt_node, cnt_edge = 0, 0, 0
        for i in range(max_node_unroll):
            node_masks1[cnt_node][:i] = 1
            adj_masks1[cnt_node][:i, :i] = 1
            cnt += 1; cnt_node += 1

            edge_total_for_node_i, start_prev_node_for_edges = (i, 0) if i < max_edge_unroll else (max_edge_unroll, i - max_edge_unroll)

            for j in range(edge_total_for_node_i):
                prev_node_connected_idx = start_prev_node_for_edges + j
                node_masks2[cnt_edge][:i + 1] = 1
                if j == 0:
                    adj_masks2[cnt_edge] = adj_masks1[cnt_node - 1].clone()
                    adj_masks2[cnt_edge][i,i] = 1
                else:
                    adj_masks2[cnt_edge] = adj_masks2[cnt_edge - 1].clone()
                    adj_masks2[cnt_edge][i, start_prev_node_for_edges + (j-1)] = 1
                    adj_masks2[cnt_edge][start_prev_node_for_edges + (j-1), i] = 1
                link_prediction_index[cnt_edge][0] = prev_node_connected_idx
                link_prediction_index[cnt_edge][1] = i
                cnt += 1; cnt_edge += 1

        assert cnt == num_masks, f'Masks count mismatch: total {cnt} vs expected {num_masks}'
        assert cnt_node == max_node_unroll, f'Node masks count mismatch: {cnt_node} vs expected {max_node_unroll}'
        assert cnt_edge == num_mask_edge, f'Edge masks count mismatch: {cnt_edge} vs expected {num_mask_edge}'
        print(f"initialize_masks: Assertions passed. Final cnt_edge={cnt_edge}")

        for k_idx in range(num_mask_edge):
            u, v = link_prediction_index[k_idx, 0].item(), link_prediction_index[k_idx, 1].item()
            if 0 <= u < max_node_unroll and 0 <= v < max_node_unroll:
                 flow_core_edge_masks[u, v] = True
            else:
                 warnings.warn(f"WARNING (initialize_masks): Invalid indices ({u},{v}) at step {k_idx} for flow_core_edge_masks.")

        node_masks = torch.cat((node_masks1, node_masks2), dim=0)
        adj_masks = torch.cat((adj_masks1, adj_masks2), dim=0)
        node_masks_param = nn.Parameter(node_masks, requires_grad=False)
        adj_masks_param = nn.Parameter(adj_masks, requires_grad=False)
        link_prediction_index_param = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks_param = nn.Parameter(flow_core_edge_masks, requires_grad=False)
        return node_masks_param, adj_masks_param, link_prediction_index_param, flow_core_edge_masks_param

    def dis_log_prob(self, z_tuple):
        z_node_transformed, z_edge_transformed = z_tuple
        batch_size = z_node_transformed.shape[0]
        if z_node_transformed.ndim == 2: z_node_transformed = z_node_transformed.view(batch_size, self.max_size, self.node_dim)
        num_mask_edge = self.latent_step - self.max_size
        if z_edge_transformed.ndim == 2: z_edge_transformed = z_edge_transformed.view(batch_size, num_mask_edge, self.bond_dim)

        node_base_log_probs_sm = torch.nn.functional.log_softmax(self.node_base_log_probs, dim=-1)
        ll_node = torch.sum(z_node_transformed * node_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))
        edge_base_log_probs_sm = torch.nn.functional.log_softmax(self.edge_base_log_probs, dim=-1)
        ll_edge = torch.sum(z_edge_transformed * edge_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))

        total_dims = self.latent_node_length + self.latent_edge_length
        if total_dims == 0:
            warnings.warn("Total latent dimensions are zero in dis_log_prob. Returning zero loss.")
            return torch.tensor(0.0, device=ll_node.device if hasattr(ll_node, 'device') else torch.device("cpu"), requires_grad=True)
        loss = -(torch.mean(ll_node + ll_edge) / total_dims)
        return loss

