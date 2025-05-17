import time
import os
import warnings  # Added for wandb import warning

import torch
from torch.optim import Adam
from tqdm import tqdm
from typing import Optional
from generator import Generator
from .energy_func import EnergyFunc
from .util import rescale_adj, requires_grad, clip_grad
# Assuming aig_config.py provides necessary constants if not passed directly
# For GraphEBM, constants like NUM_NODE_ATTRIBUTES, NUM_EDGE_ATTRIBUTES are used in its __init__
# and MAX_NODE_COUNT, NUM_EXPLICIT_NODE_FEATURES, NUM_EXPLICIT_EDGE_FEATURES are used in train_graphs.py for instantiation.
from aig_config import (
    MAX_NODE_COUNT,
    NUM_EXPLICIT_NODE_FEATURES,  # Used for n_atom_type_actual
    NUM_NODE_ATTRIBUTES,  # Total node features including padding (n_atom_type for EnergyFunc)
    NUM_EXPLICIT_EDGE_FEATURES,  # Used for n_edge_type_actual
    NUM_EDGE_ATTRIBUTES,  # Total edge features including no-edge (n_edge_type for EnergyFunc)
    # NUM_ADJ_CHANNELS is also relevant for GraphEBM's internal logic if it differs from NUM_EDGE_ATTRIBUTES
    # The provided __init__ uses n_edge_type directly for EnergyFunc, which should be NUM_EDGE_ATTRIBUTES
)
from .generate import gen_mol_from_one_shot_tensor  # For run_rand_gen

# wandb import
try:
    import wandb
except ImportError:
    wandb = None
    # Warning will be issued in train_rand_gen if wandb_active is True but wandb is not installed


class GraphEBM(Generator):
    r"""
        The method class for GraphEBM algorithm.
        Args:
            hidden (int): Hidden dimensions for the EnergyFunc.
            n_atom (int): Maximum number of nodes (MAX_NODE_COUNT).
            n_atom_type_actual (int): Number of ACTUAL node types (e.g., NUM_EXPLICIT_NODE_FEATURES).
                                      The EnergyFunc will be initialized with NUM_NODE_ATTRIBUTES.
            n_edge_type_actual (int): Number of ACTUAL edge types (e.g., NUM_EXPLICIT_EDGE_FEATURES).
                                      The EnergyFunc will be initialized with NUM_EDGE_ATTRIBUTES.
            device (torch.device, optional): The device where the model is deployed.
    """

    def __init__(self, hidden: int,
                 n_atom: int = MAX_NODE_COUNT,
                 n_atom_type_actual: int = NUM_EXPLICIT_NODE_FEATURES,  # As passed by train_graphs.py
                 n_edge_type_actual: int = NUM_EXPLICIT_EDGE_FEATURES,  # As passed by train_graphs.py
                 device: Optional[torch.device] = None):
        super(GraphEBM, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # The EnergyFunc expects the total number of node/edge types,
        # which includes any padding or virtual/no-edge channels.
        # These total counts are NUM_NODE_ATTRIBUTES and NUM_EDGE_ATTRIBUTES from aig_config.py
        self.n_atom_type_total_for_energy_func = NUM_NODE_ATTRIBUTES
        self.n_edge_type_total_for_energy_func = NUM_EDGE_ATTRIBUTES

        self.energy_function = EnergyFunc(
            n_atom_type=self.n_atom_type_total_for_energy_func,  # Total node features
            hidden=hidden,
            num_edge_type=self.n_edge_type_total_for_energy_func  # Total edge/adj channels
        ).to(self.device)

        self.n_atom = n_atom  # Max nodes (for generating samples)

        # For consistency in run_rand_gen, store the total types the model operates on.
        self.n_atom_type = self.n_atom_type_total_for_energy_func
        self.n_edge_type = self.n_edge_type_total_for_energy_func

        print(f"GraphEBM Initialized on device: {self.device}")
        print(f"  Max Nodes (self.n_atom): {self.n_atom}")
        print(f"  EnergyFunc - n_atom_type (input features): {self.n_atom_type_total_for_energy_func}")
        print(f"  EnergyFunc - num_edge_type (adj channels): {self.n_edge_type_total_for_energy_func}")
        print(f"  (Input data 'x' should have {self.n_atom_type_total_for_energy_func} features per node)")
        print(f"  (Input data 'adj' should have {self.n_edge_type_total_for_energy_func} channels)")

    def train_rand_gen(self, loader, lr, wd, max_epochs, model_conf_dict,
                       save_interval, save_dir, wandb_active=False):  # Added wandb_active
        r"""
            Running training for random generation task.
            ... (rest of docstring) ...
            Args:
                ...
                wandb_active (bool, optional): Flag to enable/disable wandb logging.
        """
        # Parameters for Langevin dynamics from model_conf_dict
        # These should be part of base_conf['model'] in aig_config.py
        c = model_conf_dict.get('c', 0.05)
        ld_step = model_conf_dict.get('ld_step', 60)
        ld_noise = model_conf_dict.get('ld_noise', 0.005)
        ld_step_size = model_conf_dict.get('ld_step_size', 30.0)
        clamp = model_conf_dict.get('clamp', True)
        alpha = model_conf_dict.get('alpha', 1.0)

        parameters = self.energy_function.parameters()
        optimizer = Adam(parameters, lr=lr, betas=(0.0, 0.999), weight_decay=wd)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if wandb_active and wandb is None:
            warnings.warn("wandb_active is True, but wandb library is not installed. Logging will be skipped.")
            wandb_active = False

        model_device = self.device  # Device model is on
        print(f"Starting GraphEBM training for {max_epochs} epochs on device {model_device}...")

        for epoch in range(max_epochs):  # Original code was range(max_epochs), usually range(1, max_epochs + 1)
            epoch_display = epoch + 1  # For display and logging
            t_start = time.time()

            epoch_losses_reg = []
            epoch_losses_en = []
            epoch_losses_total = []  # Renamed from 'losses' for clarity

            for batch_idx, data_batch in enumerate(tqdm(loader, desc=f"Epoch {epoch_display}/{max_epochs}")):
                # Data from AIGPaddedInMemoryDataset is already on CPU by default from torch.load
                # Move to model's device
                pos_x = data_batch.x.to(model_device)  # Shape (B, MAX_NODE_COUNT, NUM_NODE_ATTRIBUTES)
                pos_adj = data_batch.adj.to(
                    model_device)  # Shape (B, NUM_EDGE_ATTRIBUTES, MAX_NODE_COUNT, MAX_NODE_COUNT)

                # Dequantization (as per original GraphEBM logic)
                pos_x = pos_x + c * torch.rand_like(pos_x, device=model_device)
                pos_adj = pos_adj + c * torch.rand_like(pos_adj, device=model_device)

                # Langevin dynamics for negative samples
                # Shapes should match pos_x and pos_adj
                neg_x = torch.rand_like(pos_x, device=model_device) * (1 + c)
                neg_adj = torch.rand_like(pos_adj, device=model_device)

                # Adjacency matrix normalization (GraphEBM specific utility)
                pos_adj = rescale_adj(pos_adj)

                neg_x.requires_grad = True
                neg_adj.requires_grad = True

                requires_grad(parameters, False)
                self.energy_function.eval()

                noise_x_ld = torch.randn_like(neg_x, device=model_device)
                noise_adj_ld = torch.randn_like(neg_adj, device=model_device)
                for _ in range(ld_step):
                    noise_x_ld.normal_(0, ld_noise)
                    noise_adj_ld.normal_(0, ld_noise)
                    neg_x.data.add_(noise_x_ld.data)
                    neg_adj.data.add_(noise_adj_ld.data)

                    # EnergyFunc expects h as (B, F, N) and adj as (B, E, N, N)
                    # Input pos_x is (B,N,F), pos_adj is (B,E,N,N)
                    # EnergyFunc's forward permutes h: h.permute(0, 2, 1)
                    # So, we pass neg_x as (B,N,F) and neg_adj as (B,E,N,N)
                    neg_out_ld = self.energy_function(neg_adj, neg_x)  # adj, h
                    neg_out_ld.sum().backward()

                    if clamp:
                        if neg_x.grad is not None: neg_x.grad.data.clamp_(-0.01, 0.01)
                        if neg_adj.grad is not None: neg_adj.grad.data.clamp_(-0.01, 0.01)

                    if neg_x.grad is not None: neg_x.data.add_(neg_x.grad.data, alpha=-ld_step_size)
                    if neg_adj.grad is not None: neg_adj.data.add_(neg_adj.grad.data, alpha=-ld_step_size)

                    if neg_x.grad is not None: neg_x.grad.detach_(); neg_x.grad.zero_()
                    if neg_adj.grad is not None: neg_adj.grad.detach_(); neg_adj.grad.zero_()

                    neg_x.data.clamp_(0, 1 + c)
                    neg_adj.data.clamp_(0, 1)  # Adjacency values typically [0,1]

                # Training by backprop
                neg_x_final = neg_x.detach()
                neg_adj_final = neg_adj.detach()

                requires_grad(parameters, True)
                self.energy_function.train()
                optimizer.zero_grad()

                pos_out = self.energy_function(pos_adj, pos_x)
                neg_out = self.energy_function(neg_adj_final, neg_x_final)

                loss_reg = (pos_out ** 2 + neg_out ** 2)
                loss_en = pos_out - neg_out
                loss = (loss_en + alpha * loss_reg).mean()  # Mean over batch

                loss.backward()
                clip_grad(optimizer,
                          grad_clip_value=model_conf_dict.get('grad_clip_value', None))  # Use configured clip value
                optimizer.step()

                epoch_losses_reg.append(loss_reg.mean().item())
                epoch_losses_en.append(loss_en.mean().item())
                epoch_losses_total.append(loss.item())

            t_end = time.time()

            avg_total_loss_epoch = sum(epoch_losses_total) / len(epoch_losses_total) if epoch_losses_total else float(
                'nan')
            avg_en_loss_epoch = sum(epoch_losses_en) / len(epoch_losses_en) if epoch_losses_en else float('nan')
            avg_reg_loss_epoch = sum(epoch_losses_reg) / len(epoch_losses_reg) if epoch_losses_reg else float('nan')

            print(
                f'Epoch: {epoch_display:03d}/{max_epochs}, Avg Total Loss: {avg_total_loss_epoch:.6f}, '
                f'Avg Energy Loss: {avg_en_loss_epoch:.6f}, Avg Regularizer Loss: {avg_reg_loss_epoch:.6f}, '
                f'Sec/Epoch: {t_end - t_start:.2f}'
            )
            print('===================================================================================')

            # --- Wandb Logging (Per Epoch for GraphEBM) ---
            if wandb_active and wandb.run:
                try:
                    wandb.log({
                        "epoch": epoch_display,
                        "ebm_avg_total_loss": avg_total_loss_epoch,
                        "ebm_avg_energy_loss": avg_en_loss_epoch,
                        "ebm_avg_regularizer_loss": avg_reg_loss_epoch,
                        "learning_rate": lr
                        # Assuming lr is constant per run, or get from optimizer: optimizer.param_groups[0]['lr']
                    })
                except Exception as e:
                    warnings.warn(f"Failed to log to wandb for epoch {epoch_display}: {e}")
            # --- End Wandb Logging ---

            if epoch_display % save_interval == 0:
                ckpt_path = os.path.join(save_dir, 'GraphEBM_epoch_{}.pt'.format(epoch_display))
                torch.save(self.energy_function.state_dict(), ckpt_path)
                print(f'Saving checkpoint to {ckpt_path}')
                print('===================================================================================')

        print("GraphEBM training finished.")

    def run_rand_gen(self, model_conf_dict, checkpoint_path, n_samples):
        r"""
            Running graph generation for random generation task.
            ... (rest of docstring) ...
        """
        # Parameters for Langevin dynamics from model_conf_dict
        c = model_conf_dict.get('c', 0.05)
        ld_step = model_conf_dict.get('ld_step', 60)
        ld_noise = model_conf_dict.get('ld_noise', 0.005)
        ld_step_size = model_conf_dict.get('ld_step_size', 30.0)
        clamp = model_conf_dict.get('clamp', True)
        # atomic_num_list is handled by gen_mol_from_one_shot_tensor for AIGs

        print(f"Loading parameters from {checkpoint_path}")
        try:
            self.energy_function.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"Successfully loaded checkpoint onto {self.device}")
        except Exception as e:
            warnings.warn(f"Could not load checkpoint from {checkpoint_path}: {e}. Using randomly initialized model.")

        parameters = self.energy_function.parameters()

        # Initialization of samples
        print("Initializing samples for generation...")
        # self.n_atom_type and self.n_edge_type are total channels (e.g., NUM_NODE_ATTRIBUTES, NUM_EDGE_ATTRIBUTES)
        # self.n_atom is MAX_NODE_COUNT
        gen_x = torch.rand(n_samples, self.n_atom, self.n_atom_type, device=self.device) * (
                    1 + c)  # Shape (B, N, F_total)
        gen_adj = torch.rand(n_samples, self.n_edge_type, self.n_atom, self.n_atom,
                             device=self.device)  # Shape (B, E_total, N, N)
        # Note: EnergyFunc expects h (node features) as (B,F,N) after permute. So gen_x (B,N,F) is correct here.

        gen_x.requires_grad = True
        gen_adj.requires_grad = True
        requires_grad(parameters, False)
        self.energy_function.eval()

        noise_x = torch.randn_like(gen_x, device=self.device)
        noise_adj = torch.randn_like(gen_adj, device=self.device)

        # Langevin dynamics for generation
        print(f"Generating {n_samples} samples using Langevin dynamics ({ld_step} steps)...")
        for step_idx in tqdm(range(ld_step), desc="Langevin Dynamics Steps"):
            noise_x.normal_(0, ld_noise)
            noise_adj.normal_(0, ld_noise)
            gen_x.data.add_(noise_x.data)
            gen_adj.data.add_(noise_adj.data)

            gen_out = self.energy_function(gen_adj, gen_x)  # adj, h
            gen_out.sum().backward()

            if clamp:
                if gen_x.grad is not None: gen_x.grad.data.clamp_(-0.01, 0.01)
                if gen_adj.grad is not None: gen_adj.grad.data.clamp_(-0.01, 0.01)

            if gen_x.grad is not None: gen_x.data.add_(gen_x.grad.data, alpha=-ld_step_size)
            if gen_adj.grad is not None: gen_adj.data.add_(gen_adj.grad.data, alpha=-ld_step_size)

            if gen_x.grad is not None: gen_x.grad.detach_(); gen_x.grad.zero_()
            if gen_adj.grad is not None: gen_adj.grad.detach_(); gen_adj.grad.zero_()

            gen_x.data.clamp_(0, 1 + c)
            gen_adj.data.clamp_(0, 1)

        gen_x = gen_x.detach()  # Shape (B, N, F_total)
        gen_adj = gen_adj.detach()  # Shape (B, E_total, N, N)

        # Symmetrize adjacency matrix (important for undirected interpretation)
        gen_adj = (gen_adj + gen_adj.permute(0, 1, 3, 2)) / 2

        # Convert tensors to AIGs (NetworkX graphs)
        # gen_mol_from_one_shot_tensor needs to be adapted for AIGs.
        # It expects gen_x as (B, F_total, N) and gen_adj as (B, E_total, N, N)
        # Our gen_x is (B, N, F_total), so permute it.
        print("Converting generated tensors to NetworkX AIGs...")
        # The generate.py script's gen_mol_from_one_shot_tensor expects x shape (B, F, N)
        # and adj shape (B, E, N, N).
        # Our gen_x is (B, N, F_total), so permute it.
        # Our gen_adj is (B, E_total, N, N), which is correct.
        generated_aigs, pure_valids_flags = gen_mol_from_one_shot_tensor(gen_adj, gen_x.permute(0, 2, 1))
        print(f"Finished generation. Produced {len(generated_aigs)} AIG objects.")

        return generated_aigs, pure_valids_flags
