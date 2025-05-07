import time
import os
import copy
import pickle  # For saving generated AIGs
import networkx as nx  # For DiGraph

import torch
from torch.optim import Adam
from tqdm import tqdm

# Assuming energy_func.py and util.py are in the same directory or accessible in PYTHONPATH
from .energy_func import EnergyFunc
from .util import rescale_adj, requires_grad, clip_grad

# Try to import aig_config for type strings, handle if not found.
try:
    # Adjust this import path based on your actual project structure
    from G2PT.configs.aig import aig_config  # Assuming G2PT is accessible

    AIG_NODE_TYPE_KEYS = aig_config.NODE_TYPE_KEYS
    AIG_EDGE_TYPE_KEYS = aig_config.EDGE_TYPE_KEYS
    print("Successfully imported AIG type keys from G2PT.configs.aig_config")
except ImportError:
    print("Warning: G2PT.configs.aig_config not found. Using default AIG type keys.")
    AIG_NODE_TYPE_KEYS = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO']  # Default
    AIG_EDGE_TYPE_KEYS = ['EDGE_REG', 'EDGE_INV']  # Default


class Generator():
    r""" Base class for graph generation methods. """

    def train_rand_gen(self, loader, *args, **kwargs): raise NotImplementedError

    def run_rand_gen(self, *args, **kwargs): raise NotImplementedError

    def train_goal_directed(self, loader, *args, **kwargs): raise NotImplementedError

    def run_prop_opt(self, *args, **kwargs): raise NotImplementedError

    def train_const_prop_opt(self, loader, *args, **kwargs): raise NotImplementedError

    def run_const_prop_opt(self, *args, **kwargs): raise NotImplementedError

    def run_comp_gen(self, *args, **kwargs): raise NotImplementedError


class GraphEBM(Generator):
    r"""
        GraphEBM implementation adapted for AIG generation.
    """

    def __init__(self, n_atom, n_atom_type, n_edge_type, hidden=64, depth=2, swish_act=True, add_self=False,
                 dropout=0.0, n_power_iterations=1, device=None):
        super(GraphEBM, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.energy_function = EnergyFunc(
            n_atom_type=n_atom_type,
            hidden=hidden,
            num_edge_type=n_edge_type,
            swish_act=swish_act,
            depth=depth,
            add_self=add_self,
            dropout=dropout,
            n_power_iterations=n_power_iterations
        ).to(self.device)

        self.n_atom = n_atom  # Max number of nodes
        self.n_atom_type = n_atom_type  # Number of node types (e.g., 4 for AIG)
        self.n_edge_type = n_edge_type  # Number of edge types (e.g., 3 for AIG: REG, INV, VIRTUAL/NO_EDGE)

        # --- Determine indices for virtual/non-virtual types ---
        # Assuming the LAST index corresponds to the virtual/padding node type
        self.virtual_node_type_idx = n_atom_type - 1
        # Assuming the LAST index corresponds to the virtual/no-edge type
        self.virtual_edge_type_idx = n_edge_type - 1
        print(f"GraphEBM assuming virtual node index: {self.virtual_node_type_idx}")
        print(f"GraphEBM assuming virtual edge index: {self.virtual_edge_type_idx}")
        # Ensure AIG_EDGE_TYPE_KEYS aligns with this assumption if used in conversion
        if len(AIG_EDGE_TYPE_KEYS) != self.virtual_edge_type_idx:
            print(f"Warning: AIG_EDGE_TYPE_KEYS length ({len(AIG_EDGE_TYPE_KEYS)}) "
                  f"does not match expected non-virtual edge types ({self.virtual_edge_type_idx}). "
                  "Conversion might be incorrect.")

    def train_rand_gen(self, loader, lr, wd, max_epochs, c, ld_step, ld_noise, ld_step_size,
                       clamp_lgd_grad, alpha, save_interval, save_dir, grad_clip_value=None):
        # --- Training logic as implemented in graphebm_main_script_shape_fix ---
        # This part remains the same.
        parameters = self.energy_function.parameters()
        optimizer = Adam(parameters, lr=lr, betas=(0.0, 0.999), weight_decay=wd)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        for epoch in range(max_epochs):
            t_start = time.time();
            epoch_losses_reg, epoch_losses_en, epoch_losses_total = [], [], []
            self.energy_function.train()
            for _, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{max_epochs}")):
                pos_x_orig = batch.x.to(self.device).to(dtype=torch.float32)
                pos_adj_orig = batch.adj.to(self.device).to(dtype=torch.float32)
                pos_x = pos_x_orig + c * torch.rand_like(pos_x_orig, device=self.device)
                pos_adj = pos_adj_orig + c * torch.rand_like(pos_adj_orig, device=self.device)
                pos_adj_normalized = rescale_adj(pos_adj)
                neg_x = torch.rand_like(pos_x_orig, device=self.device) * (1 + c)
                neg_adj = torch.rand_like(pos_adj_orig, device=self.device)
                neg_x_for_ld = neg_x;
                neg_x_for_ld.requires_grad = True;
                neg_adj.requires_grad = True
                requires_grad(parameters, False);
                self.energy_function.eval()
                for _ in range(ld_step):
                    noise_x_ld = torch.randn_like(neg_x_for_ld);
                    noise_adj_ld = torch.randn_like(neg_adj)
                    noise_x_ld.normal_(0, ld_noise);
                    noise_adj_ld.normal_(0, ld_noise)
                    neg_x_for_ld.data.add_(noise_x_ld.data);
                    neg_adj.data.add_(noise_adj_ld.data)
                    neg_out_ld = self.energy_function(neg_adj, neg_x_for_ld)
                    neg_out_ld.sum().backward()
                    if clamp_lgd_grad:
                        if neg_x_for_ld.grad is not None: neg_x_for_ld.grad.data.clamp_(-0.01, 0.01)
                        if neg_adj.grad is not None: neg_adj.grad.data.clamp_(-0.01, 0.01)
                    if neg_x_for_ld.grad is not None: neg_x_for_ld.data.add_(neg_x_for_ld.grad.data,
                                                                             alpha=-ld_step_size)
                    if neg_adj.grad is not None: neg_adj.data.add_(neg_adj.grad.data, alpha=-ld_step_size)
                    if neg_x_for_ld.grad is not None: neg_x_for_ld.grad.detach_(); neg_x_for_ld.grad.zero_()
                    if neg_adj.grad is not None: neg_adj.grad.detach_(); neg_adj.grad.zero_()
                    neg_x_for_ld.data.clamp_(0, 1 + c);
                    neg_adj.data.clamp_(0, 1)
                neg_x_final = neg_x_for_ld.detach();
                neg_adj_final = neg_adj.detach()
                requires_grad(parameters, True);
                self.energy_function.train();
                optimizer.zero_grad()
                pos_out = self.energy_function(pos_adj_normalized, pos_x)
                neg_out = self.energy_function(neg_adj_final, neg_x_final)
                loss_en = pos_out - neg_out;
                loss_reg = (pos_out ** 2 + neg_out ** 2)
                total_loss = (loss_en + alpha * loss_reg).mean();
                total_loss.backward()
                if grad_clip_value is not None and grad_clip_value > 0:
                    clip_grad(optimizer, grad_clip_value=grad_clip_value)
                optimizer.step()
                epoch_losses_reg.append(loss_reg.mean().item());
                epoch_losses_en.append(loss_en.mean().item());
                epoch_losses_total.append(total_loss.item())
            t_end = time.time()
            avg_total_loss = sum(epoch_losses_total) / len(epoch_losses_total) if epoch_losses_total else float('nan')
            avg_en_loss = sum(epoch_losses_en) / len(epoch_losses_en) if epoch_losses_en else float('nan')
            avg_reg_loss = sum(epoch_losses_reg) / len(epoch_losses_reg) if epoch_losses_reg else float('nan')
            print(
                f'Epoch: {epoch + 1:03d}, Loss: {avg_total_loss:.6f}, Energy Loss: {avg_en_loss:.6f}, Regularizer Loss: {avg_reg_loss:.6f}, Sec/Epoch: {t_end - t_start:.2f}')
            print('==========================================')
            if (epoch + 1) % save_interval == 0:
                save_path = os.path.join(save_dir, f'epoch_{epoch + 1}.pt')
                torch.save(self.energy_function.state_dict(), save_path)
                print(f'Saving checkpoint at epoch {epoch + 1} to {save_path}')
                print('==========================================')

    def run_rand_gen(self, checkpoint_path, n_samples, c, ld_step, ld_noise, ld_step_size, clamp_lgd_grad,
                     num_min_nodes=5,  # Added min nodes requirement
                     aig_node_type_strings=None,  # Added for conversion
                     output_pickle_path="GraphEBM_generated_aigs.pkl"):  # Added output path
        r"""
        Running AIG graph generation for random generation task using GraphEBM.

        Args:
            checkpoint_path (str): Path to the trained model checkpoint (.pt file).
            n_samples (int): Number of AIGs to generate.
            c (float): Scaling hyperparameter for dequantization (t in paper).
            ld_step (int): Number of Langevin dynamics steps (K in paper).
            ld_noise (float): Std dev of Gaussian noise in Langevin dynamics (sigma in paper).
            ld_step_size (float): Step size for Langevin dynamics (lambda/2 in paper).
            clamp_lgd_grad (bool): Whether to clip gradients in Langevin dynamics.
            num_min_nodes (int, optional): Minimum number of actual (non-virtual) nodes required. Defaults to 5.
            aig_node_type_strings (list, optional): List of AIG node type strings. If None, uses defaults.
            output_pickle_path (str, optional): Full path to save the list of generated AIGs.

        Returns:
            list: A list of generated AIGs (networkx.DiGraph objects).
        """
        print(f"Loading parameters from {checkpoint_path}")
        self.energy_function.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        parameters = self.energy_function.parameters()
        requires_grad(parameters, False)
        self.energy_function.eval()

        if aig_node_type_strings is None:
            aig_node_type_strings = AIG_NODE_TYPE_KEYS
            print(f"Using default AIG node type strings: {aig_node_type_strings}")
        if len(aig_node_type_strings) != self.n_atom_type:
            raise ValueError(
                f"Length of aig_node_type_strings ({len(aig_node_type_strings)}) must match model's n_atom_type ({self.n_atom_type}).")

        # Use default AIG edge type strings (REG, INV) for conversion
        aig_edge_type_strings = AIG_EDGE_TYPE_KEYS
        if len(aig_edge_type_strings) != self.virtual_edge_type_idx:  # Should be 2 if virtual is last of 3
            print(f"Warning: Length of AIG_EDGE_TYPE_KEYS ({len(aig_edge_type_strings)}) "
                  f"doesn't match expected non-virtual edge types ({self.virtual_edge_type_idx}).")

        generated_aig_graphs = []
        generated_count = 0
        attempts = 0
        # Adjust max_attempts based on expected success rate
        max_attempts = max(n_samples * 20, 500)

        print(f"Attempting to generate {n_samples} AIGs with GraphEBM (min nodes: {num_min_nodes})...")

        # --- Generation Loop ---
        # Note: Generating one sample at a time for simplicity in post-processing.
        # Batch generation is possible but requires careful indexing in post-processing.
        while generated_count < n_samples and attempts < max_attempts:
            attempts += 1
            if attempts % (max_attempts // 20 if max_attempts >= 20 else 1) == 0:
                print(f"Attempt {attempts}/{max_attempts}, Generated {generated_count}/{n_samples}")

            # Initialize single sample
            gen_x_orig_shape = torch.rand(1, self.n_atom_type, self.n_atom, device=self.device) * (1 + c)  # (1, F, N)
            gen_adj = torch.rand(1, self.n_edge_type, self.n_atom, self.n_atom, device=self.device)  # (1, E, N, N)
            gen_x = gen_x_orig_shape.permute(0, 2, 1)  # (1, N, F)

            gen_x.requires_grad = True
            gen_adj.requires_grad = True

            # Langevin dynamics for the single sample
            for _ in range(ld_step):
                noise_x = torch.randn_like(gen_x);
                noise_adj = torch.randn_like(gen_adj)
                noise_x.normal_(0, ld_noise);
                noise_adj.normal_(0, ld_noise)
                gen_x.data.add_(noise_x.data);
                gen_adj.data.add_(noise_adj.data)
                gen_out = self.energy_function(gen_adj, gen_x)
                gen_out.sum().backward()  # Backward on the single sample energy
                if clamp_lgd_grad:
                    if gen_x.grad is not None: gen_x.grad.data.clamp_(-0.01, 0.01)
                    if gen_adj.grad is not None: gen_adj.grad.data.clamp_(-0.01, 0.01)
                if gen_x.grad is not None: gen_x.data.add_(gen_x.grad.data, alpha=-ld_step_size)
                if gen_adj.grad is not None: gen_adj.data.add_(gen_adj.grad.data, alpha=-ld_step_size)
                if gen_x.grad is not None: gen_x.grad.detach_(); gen_x.grad.zero_()
                if gen_adj.grad is not None: gen_adj.grad.detach_(); gen_adj.grad.zero_()
                gen_x.data.clamp_(0, 1 + c);
                gen_adj.data.clamp_(0, 1)

            gen_x_final = gen_x.detach().squeeze(0)  # Shape: (N, F)
            gen_adj_final = gen_adj.detach().squeeze(0)  # Shape: (E, N, N)

            # Symmetrize adjacency tensor
            gen_adj_final = (gen_adj_final + gen_adj_final.permute(0, 2, 1)) / 2

            # --- Discretization ---
            # Node types: Find the most likely type for each node slot
            node_type_indices = torch.argmax(gen_x_final, dim=1)  # Shape: (N)

            # Edge types: Find the most likely edge type between each pair
            adj_type_indices = torch.argmax(gen_adj_final, dim=0)  # Shape: (N, N)

            # --- Convert to NetworkX ---
            # Determine actual number of nodes (non-virtual)
            actual_node_mask = (node_type_indices != self.virtual_node_type_idx)
            num_actual_nodes = actual_node_mask.sum().item()

            # Map old indices (up to n_atom) to new indices (up to num_actual_nodes)
            original_indices = torch.where(actual_node_mask)[0]
            if len(original_indices) != num_actual_nodes:
                print(f"Warning: Mismatch in actual node count logic. Skipping attempt {attempts}.")
                continue  # Should not happen

            index_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(original_indices)}

            if num_actual_nodes >= num_min_nodes:
                aig_graph = nx.DiGraph()

                # Add actual nodes with correct types
                for old_idx_tensor in original_indices:
                    old_idx = old_idx_tensor.item()
                    new_idx = index_map[old_idx]
                    node_type_idx = node_type_indices[old_idx].item()
                    if 0 <= node_type_idx < len(aig_node_type_strings):
                        node_type_label = aig_node_type_strings[node_type_idx]
                        aig_graph.add_node(new_idx, type=node_type_label)
                    else:
                        print(f"Warning: Invalid node type index {node_type_idx} for node {old_idx}. Skipping graph.")
                        aig_graph = None  # Mark graph as invalid
                        break

                if aig_graph is None: continue  # Skip if node conversion failed

                # Add actual edges (non-virtual type) between actual nodes
                for u_old_tensor in original_indices:
                    for v_old_tensor in original_indices:
                        u_old = u_old_tensor.item()
                        v_old = v_old_tensor.item()

                        edge_type_idx = adj_type_indices[u_old, v_old].item()

                        # Check if it's NOT a virtual/no-edge type
                        if edge_type_idx != self.virtual_edge_type_idx:
                            u_new = index_map[u_old]
                            v_new = index_map[v_old]

                            if 0 <= edge_type_idx < len(aig_edge_type_strings):
                                edge_type_label = aig_edge_type_strings[edge_type_idx]
                                aig_graph.add_edge(u_new, v_new, type=edge_type_label)
                            else:
                                print(
                                    f"Warning: Invalid edge type index {edge_type_idx} for edge ({u_new}->{v_new}). Adding without type.")
                                aig_graph.add_edge(u_new, v_new)

                # Optional: Add AIG validity checks here (e.g., DAG, PO/PI constraints)
                # if not is_valid_aig(aig_graph): continue

                generated_aig_graphs.append(aig_graph)
                generated_count += 1
                if generated_count % 10 == 0 or generated_count == n_samples:
                    print(f"GraphEBM: Successfully generated {generated_count}/{n_samples} AIGs.")
            # else: # Optional log for graphs below min size
            # print(f"Attempt {attempts}: Generated graph with {num_actual_nodes} actual nodes (min required: {num_min_nodes}). Skipping.")

        if generated_count < n_samples:
            print(
                f"Warning: GraphEBM generated only {generated_count} AIGs after {max_attempts} attempts (target was {n_samples}).")

        # Save the generated graphs
        try:
            output_dir = os.path.dirname(output_pickle_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created directory for GraphEBM output: {output_dir}")

            with open(output_pickle_path, 'wb') as f:
                pickle.dump(generated_aig_graphs, f)
            print(f"Saved {len(generated_aig_graphs)} AIG DiGraphs from GraphEBM to {output_pickle_path}")
        except Exception as e:
            print(f"Error saving AIGs from GraphEBM to pickle file '{output_pickle_path}': {e}")

        return generated_aig_graphs

    # --- Other methods (train_goal_directed, run_comp_gen, etc.) remain the same ---
    # They would need similar post-processing in their respective run_* methods if used for generation.
    # ... (train_goal_directed, run_comp_gen implementations from graphebm_main_script_shape_fix) ...
    def train_goal_directed(self, loader, lr, wd, max_epochs, c, ld_step, ld_noise, ld_step_size,
                            clamp_lgd_grad, alpha, save_interval, save_dir, grad_clip_value=None):
        # (Keep implementation from graphebm_main_script_shape_fix)
        parameters = self.energy_function.parameters()
        optimizer = Adam(parameters, lr=lr, betas=(0.0, 0.999), weight_decay=wd)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        for epoch in range(max_epochs):
            t_start = time.time();
            epoch_losses_reg, epoch_losses_en, epoch_losses_total = [], [], []
            self.energy_function.train()
            for _, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{max_epochs} (Goal-Directed)")):
                pos_x_orig = batch.x.to(self.device).to(dtype=torch.float32);
                pos_adj_orig = batch.adj.to(self.device).to(dtype=torch.float32)
                pos_y = batch.y.to(self.device).to(dtype=torch.float32)
                pos_x = pos_x_orig + c * torch.rand_like(pos_x_orig, device=self.device);
                pos_adj = pos_adj_orig + c * torch.rand_like(pos_adj_orig, device=self.device)
                pos_adj_normalized = rescale_adj(pos_adj)
                neg_x = torch.rand_like(pos_x_orig, device=self.device) * (1 + c);
                neg_adj = torch.rand_like(pos_adj_orig, device=self.device)
                neg_x_for_ld = neg_x;
                neg_x_for_ld.requires_grad = True;
                neg_adj.requires_grad = True
                requires_grad(parameters, False);
                self.energy_function.eval()
                for _ in range(ld_step):
                    noise_x_ld = torch.randn_like(neg_x_for_ld);
                    noise_adj_ld = torch.randn_like(neg_adj)
                    noise_x_ld.normal_(0, ld_noise);
                    noise_adj_ld.normal_(0, ld_noise)
                    neg_x_for_ld.data.add_(noise_x_ld.data);
                    neg_adj.data.add_(noise_adj_ld.data)
                    neg_out_ld = self.energy_function(neg_adj, neg_x_for_ld);
                    neg_out_ld.sum().backward()
                    if clamp_lgd_grad:
                        if neg_x_for_ld.grad is not None: neg_x_for_ld.grad.data.clamp_(-0.01, 0.01)
                        if neg_adj.grad is not None: neg_adj.grad.data.clamp_(-0.01, 0.01)
                    if neg_x_for_ld.grad is not None: neg_x_for_ld.data.add_(neg_x_for_ld.grad.data,
                                                                             alpha=-ld_step_size)
                    if neg_adj.grad is not None: neg_adj.data.add_(neg_adj.grad.data, alpha=-ld_step_size)
                    if neg_x_for_ld.grad is not None: neg_x_for_ld.grad.detach_(); neg_x_for_ld.grad.zero_()
                    if neg_adj.grad is not None: neg_adj.grad.detach_(); neg_adj.grad.zero_()
                    neg_x_for_ld.data.clamp_(0, 1 + c);
                    neg_adj.data.clamp_(0, 1)
                neg_x_final = neg_x_for_ld.detach();
                neg_adj_final = neg_adj.detach()
                requires_grad(parameters, True);
                self.energy_function.train();
                optimizer.zero_grad()
                pos_out = self.energy_function(pos_adj_normalized, pos_x);
                neg_out = self.energy_function(neg_adj_final, neg_x_final)
                f_y = 1 + torch.exp(pos_y);
                loss_en = f_y * pos_out - neg_out;
                loss_reg = (pos_out ** 2 + neg_out ** 2)
                total_loss = (loss_en + alpha * loss_reg).mean();
                total_loss.backward()
                if grad_clip_value is not None and grad_clip_value > 0: clip_grad(optimizer,
                                                                                  grad_clip_value=grad_clip_value)
                optimizer.step()
                epoch_losses_reg.append(loss_reg.mean().item());
                epoch_losses_en.append(loss_en.mean().item());
                epoch_losses_total.append(total_loss.item())
            t_end = time.time()
            avg_total_loss = sum(epoch_losses_total) / len(epoch_losses_total) if epoch_losses_total else float('nan')
            avg_en_loss = sum(epoch_losses_en) / len(epoch_losses_en) if epoch_losses_en else float('nan')
            avg_reg_loss = sum(epoch_losses_reg) / len(epoch_losses_reg) if epoch_losses_reg else float('nan')
            print(
                f'Epoch (Goal-Directed): {epoch + 1:03d}, Loss: {avg_total_loss:.6f}, EL: {avg_en_loss:.6f}, RL: {avg_reg_loss:.6f}, Sec: {t_end - t_start:.2f}')
            if (epoch + 1) % save_interval == 0:
                save_path = os.path.join(save_dir, f'epoch_goal_directed_{epoch + 1}.pt')
                torch.save(self.energy_function.state_dict(), save_path)
                print(f'Saving checkpoint: {save_path}')

    def run_comp_gen(self, checkpoint_path_prop1, checkpoint_path_prop2, n_samples, c, ld_step, ld_noise, ld_step_size,
                     clamp_lgd_grad, weight_prop1=0.5, weight_prop2=0.5, atomic_num_list=None, correct_validity=True):
        # (Keep implementation from graphebm_main_script_shape_fix)
        # Note: This would also need discretization and conversion to NetworkX added if used for AIG generation.
        print(f"Loading model for property 1 from {checkpoint_path_prop1}")
        energy_function_prop1 = copy.deepcopy(self.energy_function);
        energy_function_prop1.load_state_dict(torch.load(checkpoint_path_prop1, map_location=self.device))
        energy_function_prop1.eval();
        requires_grad(energy_function_prop1.parameters(), False)
        print(f"Loading model for property 2 from {checkpoint_path_prop2}")
        energy_function_prop2 = copy.deepcopy(self.energy_function);
        energy_function_prop2.load_state_dict(torch.load(checkpoint_path_prop2, map_location=self.device))
        energy_function_prop2.eval();
        requires_grad(energy_function_prop2.parameters(), False)
        print("Initializing samples for compositional generation...")
        gen_x_orig_shape = torch.rand(n_samples, self.n_atom_type, self.n_atom, device=self.device) * (1 + c)
        gen_adj = torch.rand(n_samples, self.n_edge_type, self.n_atom, self.n_atom, device=self.device)
        gen_x = gen_x_orig_shape.permute(0, 2, 1)
        gen_x.requires_grad = True;
        gen_adj.requires_grad = True
        print("Generating samples via Langevin dynamics (Compositional)...")
        for i in tqdm(range(ld_step), desc="Langevin Dynamics (Compositional)"):
            noise_x = torch.randn_like(gen_x);
            noise_adj = torch.randn_like(gen_adj)
            noise_x.normal_(0, ld_noise);
            noise_adj.normal_(0, ld_noise)
            gen_x.data.add_(noise_x.data);
            gen_adj.data.add_(noise_adj.data)
            e_prop1 = energy_function_prop1(gen_adj, gen_x);
            e_prop2 = energy_function_prop2(gen_adj, gen_x)
            combined_energy = weight_prop1 * e_prop1 + weight_prop2 * e_prop2;
            combined_energy.sum().backward()
            if clamp_lgd_grad:
                if gen_x.grad is not None: gen_x.grad.data.clamp_(-0.01, 0.01)
                if gen_adj.grad is not None: gen_adj.grad.data.clamp_(-0.01, 0.01)
            if gen_x.grad is not None: gen_x.data.add_(gen_x.grad.data, alpha=-ld_step_size)
            if gen_adj.grad is not None: gen_adj.data.add_(gen_adj.grad.data, alpha=-ld_step_size)
            if gen_x.grad is not None: gen_x.grad.detach_(); gen_x.grad.zero_()
            if gen_adj.grad is not None: gen_adj.grad.detach_(); gen_adj.grad.zero_()
            gen_x.data.clamp_(0, 1 + c);
            gen_adj.data.clamp_(0, 1)
        gen_x_final = gen_x.detach();
        gen_adj_final = gen_adj.detach()
        gen_adj_final = (gen_adj_final + gen_adj_final.permute(0, 1, 3, 2)) / 2
        print("Compositional generation complete. Returning raw tensors.")
        # Add discretization and conversion to NetworkX here if needed
        return gen_x_final, gen_adj_final  # Currently returns tensors

