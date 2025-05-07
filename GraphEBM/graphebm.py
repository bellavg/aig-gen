import time
import os
import copy

import torch
from torch.optim import Adam
from tqdm import tqdm
# from rdkit import Chem # Uncomment if RDKit dependent functions are used

# Assuming energy_func.py and util.py are in the same directory or accessible in PYTHONPATH
from .energy_func import EnergyFunc  # This should be the one from graphebm_energy_func_updated
from .util import rescale_adj, requires_grad, clip_grad


class Generator():
    r"""
    The method base class for graph generation.
    """

    def train_rand_gen(self, loader, *args, **kwargs):
        raise NotImplementedError("The function train_rand_gen is not implemented!")

    def run_rand_gen(self, *args, **kwargs):
        raise NotImplementedError("The function run_rand_gen is not implemented!")

    def train_goal_directed(self, loader, *args, **kwargs):
        raise NotImplementedError("The function train_goal_directed is not implemented!")

    def run_prop_opt(self, *args, **kwargs):
        raise NotImplementedError("The function run_prop_opt is not implemented!")

    def train_const_prop_opt(self, loader, *args, **kwargs):
        raise NotImplementedError("The function train_const_prop_opt is not implemented!")

    def run_const_prop_opt(self, *args, **kwargs):
        raise NotImplementedError("The function run_const_prop_opt is not implemented!")

    def run_comp_gen(self, *args, **kwargs):
        raise NotImplementedError("The function run_comp_gen is not implemented!")


class GraphEBM(Generator):
    r"""
        The method class for GraphEBM algorithm.
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

        self.n_atom = n_atom
        self.n_atom_type = n_atom_type
        self.n_edge_type = n_edge_type

    def train_rand_gen(self, loader, lr, wd, max_epochs, c, ld_step, ld_noise, ld_step_size, clamp_lgd_grad, alpha,
                       save_interval, save_dir):
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

            for _, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{max_epochs}")):
                # batch.x from DenseDataLoader is expected to be (batch_size, num_nodes, num_features)
                pos_x_orig = batch.x.to(self.device).to(dtype=torch.float32)
                pos_adj_orig = batch.adj.to(self.device).to(dtype=torch.float32)

                # Dequantization
                pos_x = pos_x_orig + c * torch.rand_like(pos_x_orig, device=self.device)  # Shape: (B, N, F)
                pos_adj = pos_adj_orig + c * torch.rand_like(pos_adj_orig, device=self.device)

                # IMPORTANT FIX: REMOVED INCORRECT PERMUTATION
                # pos_x is already (B, N, F), which is what EnergyFunc expects for 'h'.
                # Original incorrect line: pos_x = pos_x.permute(0, 2, 1)

                pos_adj_normalized = rescale_adj(pos_adj)

                # Initialize negative samples
                # pos_x_orig is (B, N, F), so neg_x will also be (B, N, F)
                neg_x = torch.rand_like(pos_x_orig, device=self.device) * (1 + c)
                neg_adj = torch.rand_like(pos_adj_orig, device=self.device)

                # IMPORTANT FIX: REMOVED INCORRECT PERMUTATION FOR NEGATIVE SAMPLES
                # neg_x is already (B, N, F), which is what EnergyFunc expects for 'h'.
                neg_x_for_ld = neg_x  # Original incorrect line: neg_x_for_ld = neg_x.permute(0, 2, 1)
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

                requires_grad(parameters, True)
                self.energy_function.train()
                optimizer.zero_grad()

                pos_out = self.energy_function(pos_adj_normalized, pos_x)  # pos_x is (B, N, F)
                neg_out = self.energy_function(neg_adj_final, neg_x_final)  # neg_x_final is (B, N, F)

                loss_en = pos_out - neg_out
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
        print(f"Loading parameters from {checkpoint_path}")
        self.energy_function.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        parameters = self.energy_function.parameters()

        print("Initializing samples for generation...")
        # Initialize gen_x with shape (n_samples, n_atom_type, n_atom) -> (B, F, N)
        gen_x_orig_shape = torch.rand(n_samples, self.n_atom_type, self.n_atom, device=self.device) * (1 + c)
        gen_adj = torch.rand(n_samples, self.n_edge_type, self.n_atom, self.n_atom, device=self.device)

        # Permute gen_x to (n_samples, n_atom, n_atom_type) -> (B, N, F) for EnergyFunc input
        # This permutation is CORRECT for this specific initialization.
        gen_x = gen_x_orig_shape.permute(0, 2, 1)

        gen_x.requires_grad = True
        gen_adj.requires_grad = True
        requires_grad(parameters, False)
        self.energy_function.eval()

        print("Generating samples via Langevin dynamics...")
        for i in tqdm(range(ld_step), desc="Langevin Dynamics for Generation"):
            noise_x = torch.randn_like(gen_x, device=self.device)
            noise_adj = torch.randn_like(gen_adj, device=self.device)
            noise_x.normal_(0, ld_noise)
            noise_adj.normal_(0, ld_noise)
            gen_x.data.add_(noise_x.data)
            gen_adj.data.add_(noise_adj.data)

            gen_out = self.energy_function(gen_adj, gen_x)
            gen_out.sum().backward()
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

        print("Generation complete. Returning raw tensors.")
        return gen_x_final, gen_adj_final

    def train_goal_directed(self, loader, lr, wd, max_epochs, c, ld_step, ld_noise, ld_step_size, clamp_lgd_grad, alpha,
                            save_interval, save_dir):
        parameters = self.energy_function.parameters()
        optimizer = Adam(parameters, lr=lr, betas=(0.0, 0.999), weight_decay=wd)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        for epoch in range(max_epochs):
            t_start = time.time()
            epoch_losses_reg, epoch_losses_en, epoch_losses_total = [], [], []
            self.energy_function.train()

            for _, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{max_epochs} (Goal-Directed)")):
                pos_x_orig = batch.x.to(self.device).to(dtype=torch.float32)  # (B, N, F)
                pos_adj_orig = batch.adj.to(self.device).to(dtype=torch.float32)
                pos_y = batch.y.to(self.device).to(dtype=torch.float32)

                pos_x = pos_x_orig + c * torch.rand_like(pos_x_orig, device=self.device)  # (B, N, F)
                pos_adj = pos_adj_orig + c * torch.rand_like(pos_adj_orig, device=self.device)
                # REMOVED INCORRECT PERMUTATION for pos_x

                pos_adj_normalized = rescale_adj(pos_adj)

                neg_x = torch.rand_like(pos_x_orig, device=self.device) * (1 + c)  # (B, N, F)
                neg_adj = torch.rand_like(pos_adj_orig, device=self.device)
                neg_x_for_ld = neg_x  # REMOVED INCORRECT PERMUTATION
                neg_x_for_ld.requires_grad = True
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
                f_y = 1 + torch.exp(pos_y)
                loss_en = f_y * pos_out - neg_out
                loss_reg = (pos_out ** 2 + neg_out ** 2)
                total_loss = (loss_en + alpha * loss_reg).mean()
                total_loss.backward()
                if optimizer.param_groups[0]['params'][0].grad is not None: clip_grad(optimizer)
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
        print(f"Loading model for property 1 from {checkpoint_path_prop1}")
        energy_function_prop1 = copy.deepcopy(self.energy_function)
        energy_function_prop1.load_state_dict(torch.load(checkpoint_path_prop1, map_location=self.device))
        energy_function_prop1.eval();
        requires_grad(energy_function_prop1.parameters(), False)

        print(f"Loading model for property 2 from {checkpoint_path_prop2}")
        energy_function_prop2 = copy.deepcopy(self.energy_function)
        energy_function_prop2.load_state_dict(torch.load(checkpoint_path_prop2, map_location=self.device))
        energy_function_prop2.eval();
        requires_grad(energy_function_prop2.parameters(), False)

        print("Initializing samples for compositional generation...")
        gen_x_orig_shape = torch.rand(n_samples, self.n_atom_type, self.n_atom, device=self.device) * (1 + c)  # (B,F,N)
        gen_adj = torch.rand(n_samples, self.n_edge_type, self.n_atom, self.n_atom, device=self.device)
        gen_x = gen_x_orig_shape.permute(0, 2, 1)  # (B, N, F) - This is correct for this init

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
            e_prop1 = energy_function_prop1(gen_adj, gen_x)
            e_prop2 = energy_function_prop2(gen_adj, gen_x)
            combined_energy = weight_prop1 * e_prop1 + weight_prop2 * e_prop2
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
        return gen_x_final, gen_adj_final

