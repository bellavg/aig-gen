import time
import os
import copy

import torch
from torch.optim import Adam
from tqdm import tqdm
# from rdkit import Chem # Uncomment if RDKit dependent functions are used

# Assuming energy_func.py and util.py are in the same directory or accessible in PYTHONPATH
from .energy_func import EnergyFunc
from .util import rescale_adj, requires_grad, clip_grad  # clip_grad might be from a different util


class Generator():
    r"""
    The method base class for graph generation. To write a new graph generation method, create a new class
    inheriting from this class and implement the functions.
    """

    def train_rand_gen(self, loader, *args, **kwargs):
        r"""
        Running training for random generation task.

        Args:
            loader: The data loader for loading training samples.
        """
        raise NotImplementedError("The function train_rand_gen is not implemented!")

    def run_rand_gen(self, *args, **kwargs):
        r"""
        Running graph generation for random generation task.
        """
        raise NotImplementedError("The function run_rand_gen is not implemented!")

    def train_goal_directed(self, loader, *args, **kwargs):  # Changed from train_prop_opt to match paper's terminology
        r"""
        Running training for goal-directed generation task (property optimization).
        Args:
            loader: The data loader for loading training samples.
        """
        raise NotImplementedError("The function train_goal_directed is not implemented!")

    def run_prop_opt(self, *args, **kwargs):
        r"""
        Running graph generation for property optimization task.
        """
        raise NotImplementedError("The function run_prop_opt is not implemented!")

    def train_const_prop_opt(self, loader, *args, **kwargs):
        r"""
        Running training for constrained optimization task.

        Args:
            loader: The data loader for loading training samples.
        """
        raise NotImplementedError("The function train_const_prop_opt is not implemented!")

    def run_const_prop_opt(self, *args, **kwargs):
        r"""
        Running molecule optimization for constrained optimization task.
        """
        raise NotImplementedError("The function run_const_prop_opt is not implemented!")

    def run_comp_gen(self, *args, **kwargs):
        r"""
        Running graph generation for compositional generation task.
        """
        raise NotImplementedError("The function run_comp_gen is not implemented!")


class GraphEBM(Generator):
    r"""
        The method class for GraphEBM algorithm proposed in the paper `GraphEBM: Molecular Graph Generation with Energy-Based Models <https://arxiv.org/abs/2102.00546>`_.
        This class provides interfaces for running random generation, goal-directed generation
        (including property optimization and constrained optimization), and compositional generation.

        Args:
            n_atom (int): Maximum number of atoms (nodes).
            n_atom_type (int): Number of possible atom types (node feature dimension).
            n_edge_type (int): Number of possible bond types (edge feature dimension).
            hidden (int): Hidden dimension for GraphConv layers in EnergyFunc. (Paper Appendix D: d=64)
            depth (int): Number of additional GraphConv layers after the first one in EnergyFunc.
                         (Paper Appendix D: L=3 total layers, so depth=2).
            swish_act (bool): Whether to use Swish activation in EnergyFunc. (Paper Appendix D: Swish is used).
            add_self (bool): Whether to add self-connections in GraphConv layers.
            dropout (float): Dropout rate in EnergyFunc.
            n_power_iterations (int): Number of power iterations for spectral normalization.
            device (torch.device, optional): The device where the model is deployed.
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

        self.n_atom = n_atom  # Max number of atoms (nodes)
        self.n_atom_type = n_atom_type  # Number of atom types (node features dimension)
        self.n_edge_type = n_edge_type  # Number of edge types

    def train_rand_gen(self, loader, lr, wd, max_epochs, c, ld_step, ld_noise, ld_step_size, clamp_lgd_grad, alpha,
                       save_interval, save_dir):
        r"""
            Running training for random generation task. (Corresponds to Section 2.3 of the paper)

            Args:
                loader: DataLoader for training samples. Assumes batch.x is (batch, features, nodes)
                        and batch.adj is (batch, edge_types, nodes, nodes).
                lr (float): Learning rate. (Paper Appendix D: 0.0001)
                wd (float): Weight decay.
                max_epochs (int): Maximum training epochs. (Paper Appendix D: up to 20)
                c (float): Scaling hyperparameter for dequantization (t in paper Eq.8). (Paper Appendix D: t in [0,1])
                ld_step (int): Number of Langevin dynamics steps (K in paper). (Paper Appendix D: K in [30, 300])
                ld_noise (float): Std dev of Gaussian noise in Langevin dynamics (sigma in paper Eq.7). (Paper Appendix D: sigma=0.005)
                ld_step_size (float): Step size for Langevin dynamics (lambda/2 in paper Eq.7). (Paper Appendix D: lambda/2 in [10,50])
                clamp_lgd_grad (bool): Whether to clip gradients in Langevin dynamics. (Paper Appendix D: clip grad magnitude < 0.01)
                alpha (float): Weight for regularization loss term. (Paper Appendix D: alpha=1)
                save_interval (int): Frequency to save model checkpoints.
                save_dir (str): Directory to save model checkpoints.
        """
        parameters = self.energy_function.parameters()
        optimizer = Adam(parameters, lr=lr, betas=(0.0, 0.999), weight_decay=wd)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for epoch in range(max_epochs):
            t_start = time.time()
            epoch_losses_reg = []
            epoch_losses_en = []
            epoch_losses_total = []

            self.energy_function.train()  # Ensure model is in training mode for dropout, etc.

            for _, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{max_epochs}")):
                ### Dequantization of positive samples (Section 2.3, Eq. 8)
                # Assuming batch.x from loader is (batch_size, num_atom_types, num_nodes)
                # Assuming batch.adj from loader is (batch_size, num_edge_types, num_nodes, num_nodes)
                pos_x_orig = batch.x.to(self.device).to(dtype=torch.float32)
                pos_adj_orig = batch.adj.to(self.device).to(dtype=torch.float32)

                pos_x = pos_x_orig + c * torch.rand_like(pos_x_orig, device=self.device)
                pos_adj = pos_adj_orig + c * torch.rand_like(pos_adj_orig, device=self.device)

                # Permute pos_x to (batch_size, num_nodes, num_atom_types) for EnergyFunc
                # This aligns with typical GNN input and how gen_x is structured in run_rand_gen
                pos_x = pos_x.permute(0, 2, 1)

                ### Adjacency matrix normalization for positive samples (Section 2.3, Eq. 9)
                # rescale_adj should implement: A_{(:,:,k)}^{\oplus} = D^{-1}A_{(:,:,k)}^{\prime}
                # where D is diagonal degree matrix D_{(i,i)}=\sum_{j,k}A_{(i,j,k)}^{\prime}
                pos_adj_normalized = rescale_adj(pos_adj)  # Critical: util.rescale_adj must match paper's Eq.9

                ### Langevin Dynamics for generating negative samples (Section 2.3, Eq. 7, 10)
                # Initialize negative samples
                neg_x = torch.rand_like(pos_x_orig, device=self.device) * (1 + c)  # Shape (B, features, N_nodes)
                neg_adj = torch.rand_like(pos_adj_orig, device=self.device)  # Shape (B, edge_types, N_nodes, N_nodes)

                # Permute neg_x for consistency if it's used by energy_function directly in LD
                # However, energy_function expects (B, N_nodes, features)
                neg_x_for_ld = neg_x.permute(0, 2, 1)  # Shape (B, N_nodes, features) for energy_function input

                neg_x_for_ld.requires_grad = True
                neg_adj.requires_grad = True  # Assuming neg_adj is already in (B, E, N, N) format for energy_function

                # Langevin dynamics loop
                requires_grad(parameters, False)  # Detach EBM parameters from LD gradient computation
                self.energy_function.eval()  # Use EBM in eval mode for LD (e.g. no dropout)

                for _ in range(ld_step):
                    noise_x_ld = torch.randn_like(neg_x_for_ld, device=self.device)  # Noise for (B, N_nodes, features)
                    noise_adj_ld = torch.randn_like(neg_adj, device=self.device)  # Noise for (B, E, N_nodes, N_nodes)

                    noise_x_ld.normal_(0, ld_noise)
                    noise_adj_ld.normal_(0, ld_noise)

                    # Add noise to current negative samples
                    neg_x_for_ld.data.add_(noise_x_ld.data)
                    neg_adj.data.add_(noise_adj_ld.data)

                    # Compute energy and gradients w.r.t. negative samples
                    neg_out_ld = self.energy_function(neg_adj, neg_x_for_ld)
                    neg_out_ld.sum().backward()  # Sum over batch for independent gradients

                    if clamp_lgd_grad:  # As per Appendix D
                        if neg_x_for_ld.grad is not None:
                            neg_x_for_ld.grad.data.clamp_(-0.01, 0.01)
                        if neg_adj.grad is not None:
                            neg_adj.grad.data.clamp_(-0.01, 0.01)

                    # Update negative samples (gradient descent on energy)
                    if neg_x_for_ld.grad is not None:
                        neg_x_for_ld.data.add_(neg_x_for_ld.grad.data, alpha=-ld_step_size)
                    if neg_adj.grad is not None:
                        neg_adj.data.add_(neg_adj.grad.data, alpha=-ld_step_size)

                    # Zero gradients for next LD step
                    if neg_x_for_ld.grad is not None:
                        neg_x_for_ld.grad.detach_()
                        neg_x_for_ld.grad.zero_()
                    if neg_adj.grad is not None:
                        neg_adj.grad.detach_()
                        neg_adj.grad.zero_()

                    # Clamp negative samples to valid range (Section 2.3, after Eq. 10)
                    neg_x_for_ld.data.clamp_(0, 1 + c)  # Clamp (B, N_nodes, features)
                    neg_adj.data.clamp_(0, 1)  # Clamp (B, E, N_nodes, N_nodes)

                # Detach negative samples from computation graph for EBM training
                neg_x_final = neg_x_for_ld.detach()
                neg_adj_final = neg_adj.detach()

                ### Training EBM by backpropagation (Section 2.3, Eq. 11, 12)
                requires_grad(parameters, True)  # Re-enable gradients for EBM parameters
                self.energy_function.train()  # Set EBM to train mode
                optimizer.zero_grad()

                pos_out = self.energy_function(pos_adj_normalized, pos_x)  # pos_x is (B, N_nodes, features)
                neg_out = self.energy_function(neg_adj_final, neg_x_final)  # neg_x_final is (B, N_nodes, features)

                loss_en = pos_out - neg_out  # Energy shaping loss (Eq. 11)
                loss_reg = (pos_out ** 2 + neg_out ** 2)  # Energy magnitude regularization (Eq. 12)

                total_loss = (loss_en + alpha * loss_reg).mean()  # Average over batch
                total_loss.backward()

                if optimizer.param_groups[0]['params'][0].grad is not None:  # Check if grads exist
                    clip_grad(optimizer)  # Custom gradient clipping if needed
                optimizer.step()

                epoch_losses_reg.append(loss_reg.mean().item())
                epoch_losses_en.append(loss_en.mean().item())
                epoch_losses_total.append(total_loss.item())

            t_end = time.time()
            avg_total_loss = sum(epoch_losses_total) / len(epoch_losses_total) if epoch_losses_total else float('nan')
            avg_en_loss = sum(epoch_losses_en) / len(epoch_losses_en) if epoch_losses_en else float('nan')
            avg_reg_loss = sum(epoch_losses_reg) / len(epoch_losses_reg) if epoch_losses_reg else float('nan')

            print(f'Epoch: {epoch + 1:03d}, Loss: {avg_total_loss:.6f}, Energy Loss: {avg_en_loss:.6f}, '
                  f'Regularizer Loss: {avg_reg_loss:.6f}, Sec/Epoch: {t_end - t_start:.2f}')
            print('==========================================')

            if (epoch + 1) % save_interval == 0:
                save_path = os.path.join(save_dir, f'epoch_{epoch + 1}.pt')
                torch.save(self.energy_function.state_dict(), save_path)
                print(f'Saving checkpoint at epoch {epoch + 1} to {save_path}')
                print('==========================================')

    def run_rand_gen(self, checkpoint_path, n_samples, c, ld_step, ld_noise, ld_step_size, clamp_lgd_grad,
                     atomic_num_list=None, correct_validity=True):
        r"""
            Running graph generation for random generation task. (Corresponds to Section 2.4 of the paper)

            Args:
                checkpoint_path (str): Path to the trained model checkpoint (.pt file).
                n_samples (int): Number of molecules to generate.
                c (float): Scaling hyperparameter for dequantization (t in paper).
                ld_step (int): Number of Langevin dynamics steps (K in paper).
                ld_noise (float): Std dev of Gaussian noise in Langevin dynamics (sigma in paper).
                ld_step_size (float): Step size for Langevin dynamics (lambda/2 in paper).
                clamp_lgd_grad (bool): Whether to clip gradients in Langevin dynamics.
                atomic_num_list (list, optional): List to map atom type indices to atomic numbers (for RDKit conversion).
                correct_validity (bool): Whether to apply validity correction (e.g. RDKit based).

            Returns:
                list: A list of generated molecules (e.g., RDKit Mol objects if conversion is implemented).
                      Currently returns detached tensors gen_x, gen_adj.
        """
        print(f"Loading parameters from {checkpoint_path}")
        self.energy_function.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        parameters = self.energy_function.parameters()

        ### Initialization of samples for generation (Section 2.3, Eq. 10)
        print("Initializing samples for generation...")
        # Initialize gen_x with shape (n_samples, n_atom_type, n_atom)
        gen_x_orig_shape = torch.rand(n_samples, self.n_atom_type, self.n_atom, device=self.device) * (1 + c)
        # Initialize gen_adj with shape (n_samples, n_edge_type, n_atom, n_atom)
        gen_adj = torch.rand(n_samples, self.n_edge_type, self.n_atom, self.n_atom, device=self.device)

        # Permute gen_x to (n_samples, n_atom, n_atom_type) for EnergyFunc input
        gen_x = gen_x_orig_shape.permute(0, 2, 1)

        gen_x.requires_grad = True
        gen_adj.requires_grad = True

        requires_grad(parameters, False)  # Detach EBM parameters
        self.energy_function.eval()  # EBM in eval mode

        ### Langevin dynamics for generation (Section 2.4, reusing Eq. 7)
        print("Generating samples via Langevin dynamics...")
        for i in tqdm(range(ld_step), desc="Langevin Dynamics for Generation"):
            noise_x = torch.randn_like(gen_x, device=self.device)  # Noise for (B, N_nodes, features)
            noise_adj = torch.randn_like(gen_adj, device=self.device)  # Noise for (B, E, N_nodes, N_nodes)

            noise_x.normal_(0, ld_noise)
            noise_adj.normal_(0, ld_noise)

            gen_x.data.add_(noise_x.data)
            gen_adj.data.add_(noise_adj.data)

            gen_out = self.energy_function(gen_adj, gen_x)
            gen_out.sum().backward()

            if clamp_lgd_grad:
                if gen_x.grad is not None:
                    gen_x.grad.data.clamp_(-0.01, 0.01)
                if gen_adj.grad is not None:
                    gen_adj.grad.data.clamp_(-0.01, 0.01)

            if gen_x.grad is not None:
                gen_x.data.add_(gen_x.grad.data, alpha=-ld_step_size)
            if gen_adj.grad is not None:
                gen_adj.data.add_(gen_adj.grad.data, alpha=-ld_step_size)

            if gen_x.grad is not None:
                gen_x.grad.detach_()
                gen_x.grad.zero_()
            if gen_adj.grad is not None:
                gen_adj.grad.detach_()
                gen_adj.grad.zero_()

            gen_x.data.clamp_(0, 1 + c)  # Clamp to [0, 1+t)
            gen_adj.data.clamp_(0, 1)  # Clamp to [0, 1)

        gen_x_final = gen_x.detach()
        gen_adj_final = gen_adj.detach()

        ### Post-processing (Section 2.4)
        # Symmetrize adjacency tensor
        gen_adj_final = (gen_adj_final + gen_adj_final.permute(0, 1, 3, 2)) / 2  # Average for symmetry

        # Discretization (argmax) would happen here.
        # gen_x_discrete = torch.argmax(gen_x_final.permute(0,2,1), dim=1) # if gen_x_final is (B,N,F) -> (B,F,N) then argmax
        # gen_adj_discrete = torch.argmax(gen_adj_final, dim=1)

        # The commented RDKit lines below would handle conversion and validity correction.
        # For now, returning raw tensors.
        # gen_mols = gen_mol_from_one_shot_tensor(gen_adj_final, gen_x_final.permute(0,2,1), atomic_num_list, correct_validity=correct_validity)
        # return gen_mols

        print("Generation complete. Returning raw tensors.")
        # Return gen_x_final in (B, N_nodes, Features) and gen_adj_final in (B, E_types, N_nodes, N_nodes)
        return gen_x_final, gen_adj_final

    def train_goal_directed(self, loader, lr, wd, max_epochs, c, ld_step, ld_noise, ld_step_size, clamp_lgd_grad, alpha,
                            save_interval, save_dir):
        r"""
            Running training for goal-directed generation task. (Corresponds to Section 2.5 of the paper)
            This function is analogous to train_rand_gen but uses a modified loss.

            Args:
                loader: DataLoader. Assumes batch contains batch.x, batch.adj, and batch.y (normalized property).
                (Other parameters similar to train_rand_gen)
                pos_y (torch.Tensor): Normalized property values for positive samples, shape (batch_size, 1).
        """
        parameters = self.energy_function.parameters()
        optimizer = Adam(parameters, lr=lr, betas=(0.0, 0.999), weight_decay=wd)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for epoch in range(max_epochs):
            t_start = time.time()
            epoch_losses_reg = []
            epoch_losses_en = []
            epoch_losses_total = []

            self.energy_function.train()

            for _, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{max_epochs} (Goal-Directed)")):
                pos_x_orig = batch.x.to(self.device).to(dtype=torch.float32)
                pos_adj_orig = batch.adj.to(self.device).to(dtype=torch.float32)
                pos_y = batch.y.to(self.device).to(dtype=torch.float32)  # Normalized property

                pos_x = pos_x_orig + c * torch.rand_like(pos_x_orig, device=self.device)
                pos_adj = pos_adj_orig + c * torch.rand_like(pos_adj_orig, device=self.device)
                pos_x = pos_x.permute(0, 2, 1)  # (B, N_nodes, Features)

                pos_adj_normalized = rescale_adj(pos_adj)

                # Langevin Dynamics (same as in train_rand_gen)
                neg_x = torch.rand_like(pos_x_orig, device=self.device) * (1 + c)
                neg_adj = torch.rand_like(pos_adj_orig, device=self.device)
                neg_x_for_ld = neg_x.permute(0, 2, 1)
                neg_x_for_ld.requires_grad = True
                neg_adj.requires_grad = True

                requires_grad(parameters, False)
                self.energy_function.eval()

                for _ in range(ld_step):
                    noise_x_ld = torch.randn_like(neg_x_for_ld, device=self.device)
                    noise_adj_ld = torch.randn_like(neg_adj, device=self.device)
                    noise_x_ld.normal_(0, ld_noise)
                    noise_adj_ld.normal_(0, ld_noise)
                    neg_x_for_ld.data.add_(noise_x_ld.data)
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
                    neg_x_for_ld.data.clamp_(0, 1 + c)
                    neg_adj.data.clamp_(0, 1)

                neg_x_final = neg_x_for_ld.detach()
                neg_adj_final = neg_adj.detach()

                # Training EBM
                requires_grad(parameters, True)
                self.energy_function.train()
                optimizer.zero_grad()

                pos_out = self.energy_function(pos_adj_normalized, pos_x)
                neg_out = self.energy_function(neg_adj_final, neg_x_final)

                # Goal-directed loss (Section 2.5, Eq. 13)
                # f(y) = 1 + e^y. Ensure pos_y is normalized property [0,1]
                f_y = 1 + torch.exp(pos_y)
                loss_en = f_y * pos_out - neg_out
                loss_reg = (pos_out ** 2 + neg_out ** 2)

                total_loss = (loss_en + alpha * loss_reg).mean()
                total_loss.backward()
                if optimizer.param_groups[0]['params'][0].grad is not None:
                    clip_grad(optimizer)
                optimizer.step()

                epoch_losses_reg.append(loss_reg.mean().item())
                epoch_losses_en.append(loss_en.mean().item())
                epoch_losses_total.append(total_loss.item())

            t_end = time.time()
            avg_total_loss = sum(epoch_losses_total) / len(epoch_losses_total) if epoch_losses_total else float('nan')
            avg_en_loss = sum(epoch_losses_en) / len(epoch_losses_en) if epoch_losses_en else float('nan')
            avg_reg_loss = sum(epoch_losses_reg) / len(epoch_losses_reg) if epoch_losses_reg else float('nan')

            print(
                f'Epoch (Goal-Directed): {epoch + 1:03d}, Loss: {avg_total_loss:.6f}, Energy Loss: {avg_en_loss:.6f}, '
                f'Regularizer Loss: {avg_reg_loss:.6f}, Sec/Epoch: {t_end - t_start:.2f}')
            print('==========================================')

            if (epoch + 1) % save_interval == 0:
                save_path = os.path.join(save_dir, f'epoch_goal_directed_{epoch + 1}.pt')
                torch.save(self.energy_function.state_dict(), save_path)
                print(f'Saving checkpoint at epoch {epoch + 1} to {save_path}')
                print('==========================================')

    # Placeholder for run_prop_opt - depends on RDKit utils for molecule processing
    # def run_prop_opt(self, checkpoint_path, initialization_loader, c, ld_step, ld_noise, ld_step_size, clamp_lgd_grad, atomic_num_list, train_smiles):
    #     # ... (Implementation would be similar to run_rand_gen but might involve property calculation)
    #     raise NotImplementedError("run_prop_opt requires RDKit utilities for molecule processing and property calculation.")

    # Placeholder for run_const_prop_opt
    # def run_const_prop_opt(self, checkpoint_path, initialization_loader, c, ld_step, ld_noise, ld_step_size, clamp_lgd_grad, atomic_num_list, train_smiles):
    #     # ...
    #     raise NotImplementedError("run_const_prop_opt requires RDKit utilities.")

    def run_comp_gen(self, checkpoint_path_prop1, checkpoint_path_prop2, n_samples, c, ld_step, ld_noise, ld_step_size,
                     clamp_lgd_grad, weight_prop1=0.5, weight_prop2=0.5, atomic_num_list=None, correct_validity=True):
        r"""
            Running graph generation for compositional generation task. (Corresponds to Section 2.6 of the paper)

            Args:
                checkpoint_path_prop1 (str): Path to model trained for property 1.
                checkpoint_path_prop2 (str): Path to model trained for property 2.
                weight_prop1 (float): Weight for energy from model 1.
                weight_prop2 (float): Weight for energy from model 2.
                (Other parameters similar to run_rand_gen)

            Returns:
                list: A list of generated molecules (e.g., RDKit Mol objects if conversion is implemented).
        """
        print(f"Loading model for property 1 from {checkpoint_path_prop1}")
        energy_function_prop1 = copy.deepcopy(self.energy_function)  # Create a new instance
        energy_function_prop1.load_state_dict(torch.load(checkpoint_path_prop1, map_location=self.device))
        energy_function_prop1.eval()
        requires_grad(energy_function_prop1.parameters(), False)

        print(f"Loading model for property 2 from {checkpoint_path_prop2}")
        energy_function_prop2 = copy.deepcopy(self.energy_function)  # Create another new instance
        energy_function_prop2.load_state_dict(torch.load(checkpoint_path_prop2, map_location=self.device))
        energy_function_prop2.eval()
        requires_grad(energy_function_prop2.parameters(), False)

        print("Initializing samples for compositional generation...")
        gen_x_orig_shape = torch.rand(n_samples, self.n_atom_type, self.n_atom, device=self.device) * (1 + c)
        gen_adj = torch.rand(n_samples, self.n_edge_type, self.n_atom, self.n_atom, device=self.device)
        gen_x = gen_x_orig_shape.permute(0, 2, 1)  # (B, N_nodes, Features)

        gen_x.requires_grad = True
        gen_adj.requires_grad = True

        print("Generating samples via Langevin dynamics (Compositional)...")
        for i in tqdm(range(ld_step), desc="Langevin Dynamics (Compositional)"):
            noise_x = torch.randn_like(gen_x, device=self.device)
            noise_adj = torch.randn_like(gen_adj, device=self.device)
            noise_x.normal_(0, ld_noise)
            noise_adj.normal_(0, ld_noise)

            gen_x.data.add_(noise_x.data)
            gen_adj.data.add_(noise_adj.data)

            # Combined energy (Section 2.6, Eq. 14)
            e_prop1 = energy_function_prop1(gen_adj, gen_x)
            e_prop2 = energy_function_prop2(gen_adj, gen_x)
            combined_energy = weight_prop1 * e_prop1 + weight_prop2 * e_prop2  # Paper uses sum, can be weighted sum
            combined_energy.sum().backward()

            if clamp_lgd_grad:
                if gen_x.grad is not None: gen_x.grad.data.clamp_(-0.01, 0.01)
                if gen_adj.grad is not None: gen_adj.grad.data.clamp_(-0.01, 0.01)

            if gen_x.grad is not None: gen_x.data.add_(gen_x.grad.data, alpha=-ld_step_size)
            if gen_adj.grad is not None: gen_adj.data.add_(gen_adj.grad.data, alpha=-ld_step_size)

            if gen_x.grad is not None: gen_x.grad.detach_(); gen_x.grad.zero_()
            if gen_adj.grad is not None: gen_adj.grad.detach_(); gen_adj.grad.zero_()

            gen_x.data.clamp_(0, 1 + c)
            gen_adj.data.clamp_(0, 1)

        gen_x_final = gen_x.detach()
        gen_adj_final = gen_adj.detach()
        gen_adj_final = (gen_adj_final + gen_adj_final.permute(0, 1, 3, 2)) / 2

        # gen_mols = gen_mol_from_one_shot_tensor(gen_adj_final, gen_x_final.permute(0,2,1), atomic_num_list, correct_validity=correct_validity)
        # return gen_mols
        print("Compositional generation complete. Returning raw tensors.")
        return gen_x_final, gen_adj_final

