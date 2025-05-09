import time
import os
import copy

import networkx as nx
import torch
from torch.optim import Adam
from tqdm import tqdm
from generator import Generator
#from dig.ggraph.utils import gen_mol_from_one_shot_tensor
from .energy_func import EnergyFunc
from .util import rescale_adj, requires_grad, clip_grad
from aig_config import *
from .generate import gen_mol_from_one_shot_tensor


class GraphEBM(Generator):
    r"""
        Adapted for AIG generation with on-the-fly virtual node channel handling.
        Args:
            n_atom (int): Maximum number of nodes (MAX_NODE_COUNT for AIGs).
            n_atom_type (int): Number of ACTUAL node types (e.g., NUM_NODE_FEATURES from aig_config).
                               The model will internally add +1 for the virtual channel.
            n_edge_type (int): Number of ACTUAL edge types (e.g., NUM_EXPLICIT_EDGE_TYPES from aig_config).
                               The model will use NUM_ADJ_CHANNELS (actual + virtual) internally.
            hidden (int): Hidden dimensions for the EnergyFunc.
            device (torch.device, optional): The device where the model is deployed.
    """

    def __init__(self, n_atom, n_atom_type_actual, n_edge_type_actual, hidden, device=None):
        super(GraphEBM, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Node feature setup
        self.num_actual_node_features = n_atom_type_actual  # e.g., 4 (NUM_NODE_FEATURES)
        # self.n_atom_type from user's original code was n_atom_type_actual + 1
        # This is the total number of channels the EnergyFunc's node input will have.
        self.model_total_node_channels = n_atom_type_actual + 1  # e.g., 5
        self.virtual_node_channel_idx = self.num_actual_node_features  # Index for the virtual channel (0-indexed)

        # Edge feature setup
        # n_edge_type_actual is NUM_EXPLICIT_EDGE_TYPES (e.g., 2)
        # The EnergyFunc and data use NUM_ADJ_CHANNELS (e.g., 3 from aig_config)
        self.model_total_edge_channels = NUM_ADJ_CHANNELS  # This is what EnergyFunc needs for num_edge_type
        self.virtual_edge_channel_idx = self.model_total_edge_channels - 1

        # Initialize EnergyFunc with total channels
        self.energy_function = EnergyFunc(self.model_total_node_channels, hidden, self.model_total_edge_channels).to(
            self.device)

        self.n_atom = n_atom  # MAX_NODE_COUNT

        # For consistency and use in methods like run_rand_gen,
        # self.n_atom_type and self.n_edge_type will store the *total model channels*.
        self.n_atom_type = self.model_total_node_channels  # Consistent with user's self.n_atom_type = n_atom_type + 1
        self.n_edge_type = self.model_total_edge_channels  # Corrected to use total edge channels

        print(f"GraphEBM Initialized (User Structure Adapted & Corrected):")
        print(f"  Max Nodes (self.n_atom): {self.n_atom}")
        print(f"  Model Node Channels (self.n_atom_type): {self.n_atom_type}")
        print(f"  Model Edge Channels (self.n_edge_type): {self.n_edge_type}")
        print(
            f"  Actual Node Features from input data (self.num_actual_node_features): {self.num_actual_node_features}")
        print(f"  Virtual Node Channel Index used in transform: {self.virtual_node_channel_idx}")
        print(f"  Virtual Edge Channel Index for AIG conversion: {self.virtual_edge_channel_idx}")

    def _transform_node_features_add_virtual_channel(self, x_features_actual_BNF, device):
        """
        Transforms node features from (B, N, NUM_ACTUAL_FEATURES)
        to (B, MODEL_TOTAL_NODE_CHANNELS, N) by adding a virtual node channel.
        Assumes padded rows in x_features_actual_BNF are all zeros.
        The output shape is (B, F_total, N) as expected by the original EnergyFunc's 'h' input.
        """
        batch_size, num_nodes, num_actual_feats = x_features_actual_BNF.shape

        if num_actual_feats != self.num_actual_node_features:
            raise ValueError(
                f"Input x_features_actual_BNF has {num_actual_feats} features, "
                f"but model was initialized for {self.num_actual_node_features} actual features."
            )

        # Target shape for EnergyFunc input h: (B, self.n_atom_type, N)
        # self.n_atom_type is self.model_total_node_channels
        x_transformed_BFN = torch.zeros(batch_size, self.n_atom_type, num_nodes, device=device)

        # Copy actual features. Input is (B,N,F_actual), permute to (B,F_actual,N) for assignment.
        x_transformed_BFN[:, :self.num_actual_node_features, :] = x_features_actual_BNF.permute(0, 2, 1)

        is_padding_node_BN = (x_features_actual_BNF.abs().sum(dim=2) < 1e-6)

        for b in range(batch_size):
            padding_node_indices_in_sample = torch.where(is_padding_node_BN[b])[0]
            if len(padding_node_indices_in_sample) > 0:
                # Use self.virtual_node_channel_idx (which is self.num_actual_node_features)
                x_transformed_BFN[b, self.virtual_node_channel_idx, padding_node_indices_in_sample] = 1.0
                x_transformed_BFN[b, :self.num_actual_node_features, padding_node_indices_in_sample] = 0.0

        return x_transformed_BFN

    def train_rand_gen(self, loader, lr, wd, max_epochs, model_conf_dict,
                       save_interval, save_dir):
        c = model_conf_dict['c']
        ld_step = model_conf_dict['ld_step']
        ld_noise = model_conf_dict['ld_noise']
        ld_step_size = model_conf_dict['ld_step_size']
        clamp = model_conf_dict['clamp']
        alpha = model_conf_dict['alpha']

        parameters = self.energy_function.parameters()
        optimizer = Adam(parameters, lr=lr, betas=(0.0, 0.999), weight_decay=wd)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for epoch in range(max_epochs):
            t_start = time.time()
            losses_reg, losses_en, losses = [], [], []
            for i_batch, batch_data in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{max_epochs}")):

                pos_x_actual_feats_BNF = batch_data.x.to(self.device).to(dtype=torch.float32)
                pos_x_BFN = self._transform_node_features_add_virtual_channel(pos_x_actual_feats_BNF, self.device)
                pos_x_BFN += c * torch.rand_like(pos_x_BFN, device=self.device)

                pos_adj_BEFN = batch_data.adj.to(self.device).to(dtype=torch.float32)
                pos_adj_BEFN += c * torch.rand_like(pos_adj_BEFN, device=self.device)

                pos_adj_normalized_BEFN = rescale_adj(pos_adj_BEFN)

                neg_x_BFN = torch.rand(pos_x_BFN.shape[0], self.n_atom_type, self.n_atom, device=self.device) * (1 + c)
                neg_adj_BEFN = torch.rand_like(pos_adj_BEFN, device=self.device)

                neg_x_BFN.requires_grad = True
                neg_adj_BEFN.requires_grad = True

                requires_grad(parameters, False)
                self.energy_function.eval()

                noise_x_ld = torch.randn_like(neg_x_BFN, device=self.device)
                noise_adj_ld = torch.randn_like(neg_adj_BEFN, device=self.device)
                for _ in range(ld_step):
                    noise_x_ld.normal_(0, ld_noise)
                    noise_adj_ld.normal_(0, ld_noise)
                    neg_x_BFN.data.add_(noise_x_ld.data)
                    neg_adj_BEFN.data.add_(noise_adj_ld.data)

                    # EnergyFunc expects h as (B, F, N) and adj as (B, E, N, N)
                    # Your EnergyFunc (original) has h.permute(0,2,1) at the start of its forward.
                    # So, it expects input h to be (B, F_total, N)
                    # And input adj to be (B, E_total, N, N)
                    neg_out_ld = self.energy_function(neg_adj_BEFN, neg_x_BFN)
                    neg_out_ld.sum().backward()
                    if clamp:
                        if neg_x_BFN.grad is not None: neg_x_BFN.grad.data.clamp_(-0.01, 0.01)
                        if neg_adj_BEFN.grad is not None: neg_adj_BEFN.grad.data.clamp_(-0.01, 0.01)

                    if neg_x_BFN.grad is not None: neg_x_BFN.data.add_(neg_x_BFN.grad.data, alpha=-ld_step_size)
                    if neg_adj_BEFN.grad is not None: neg_adj_BEFN.data.add_(neg_adj_BEFN.grad.data,
                                                                             alpha=-ld_step_size)

                    if neg_x_BFN.grad is not None: neg_x_BFN.grad.detach_(); neg_x_BFN.grad.zero_()
                    if neg_adj_BEFN.grad is not None: neg_adj_BEFN.grad.detach_(); neg_adj_BEFN.grad.zero_()

                    neg_x_BFN.data.clamp_(0, 1 + c)
                    neg_adj_BEFN.data.clamp_(0, 1)

                neg_x_final_BFN = neg_x_BFN.detach()
                neg_adj_final_BEFN = neg_adj_BEFN.detach()

                requires_grad(parameters, True)
                self.energy_function.train()
                optimizer.zero_grad()

                pos_out = self.energy_function(pos_adj_normalized_BEFN, pos_x_BFN)
                neg_out = self.energy_function(neg_adj_final_BEFN, neg_x_final_BFN)

                loss_reg = (pos_out ** 2 + neg_out ** 2)
                loss_en = pos_out - neg_out
                loss = (loss_en + alpha * loss_reg).mean()
                loss.backward()
                clip_grad(optimizer)
                optimizer.step()

                losses_reg.append(loss_reg.mean().item())
                losses_en.append(loss_en.mean().item())
                losses.append(loss.item())

            t_end = time.time()
            if (epoch + 1) % save_interval == 0:
                torch.save(self.energy_function.state_dict(), os.path.join(save_dir, f'epoch_{epoch + 1}.pt'))
                print(f'Saving checkpoint at epoch {epoch + 1}')
            avg_total_loss = sum(losses) / len(losses) if losses else float('nan')
            avg_en_loss = sum(losses_en) / len(losses_en) if losses_en else float('nan')
            avg_reg_loss = sum(losses_reg) / len(losses_reg) if losses_reg else float('nan')
            print(
                f'Epoch: {epoch + 1:03d}, Loss: {avg_total_loss:.6f}, Energy Loss: {avg_en_loss:.6f}, Regularizer Loss: {avg_reg_loss:.6f}, Sec/Epoch: {t_end - t_start:.2f}')
            print('==========================================')

    def run_rand_gen(self, model_conf_dict, checkpoint_path, n_samples):
        r"""
            Running graph generation for random generation task.

            Args:
                checkpoint_path (str): The path of the trained model, *i.e.*, the .pt file.
                n_samples (int): the number of molecules to generate.
                c (float): The scaling hyperparameter for dequantization.
                ld_step (int): The number of iteration steps of Langevin dynamics.
                ld_noise (float): The standard deviation of the added noise in Langevin dynamics.
                ld_step_size (int): The step size of Langevin dynamics.
                clamp (bool): Whether to use gradient clamp in Langevin dynamics.
                atomic_num_list (list): The list used to indicate atom types.

            :rtype:
                gen_mols (list): A list of generated molecules represented by rdkit Chem.Mol objects;

        """
        c = model_conf_dict['c']
        ld_step = model_conf_dict['ld_step']
        ld_noise = model_conf_dict['ld_noise']
        ld_step_size = model_conf_dict['ld_step_size']
        clamp = model_conf_dict['clamp']

        print("Loading paramaters from {}".format(checkpoint_path))
        self.energy_function.load_state_dict(torch.load(checkpoint_path))
        parameters = self.energy_function.parameters()

        ### Initialization
        print("Initializing samples...")
        gen_x = torch.rand(n_samples, self.n_atom_type, self.n_atom, device=self.device) * (1 + c)
        gen_adj = torch.rand(n_samples, self.n_edge_type, self.n_atom, self.n_atom, device=self.device)

        gen_x.requires_grad = True
        gen_adj.requires_grad = True
        requires_grad(parameters, False)
        self.energy_function.eval()

        noise_x = torch.randn_like(gen_x, device=self.device)
        noise_adj = torch.randn_like(gen_adj, device=self.device)

        ### Langevin dynamics
        print("Generating samples...")
        for _ in range(ld_step):
            noise_x.normal_(0, ld_noise)
            noise_adj.normal_(0, ld_noise)
            gen_x.data.add_(noise_x.data)
            gen_adj.data.add_(noise_adj.data)

            gen_out = self.energy_function(gen_adj, gen_x)
            gen_out.sum().backward()
            if clamp:
                gen_x.grad.data.clamp_(-0.01, 0.01)
                gen_adj.grad.data.clamp_(-0.01, 0.01)

            gen_x.data.add_(gen_x.grad.data, alpha=-ld_step_size)
            gen_adj.data.add_(gen_adj.grad.data, alpha=-ld_step_size)

            gen_x.grad.detach_()
            gen_x.grad.zero_()
            gen_adj.grad.detach_()
            gen_adj.grad.zero_()

            gen_x.data.clamp_(0, 1 + c)
            gen_adj.data.clamp_(0, 1)

        gen_x = gen_x.detach()
        gen_adj = gen_adj.detach()
        gen_adj = (gen_adj + gen_adj.permute(0, 1, 3, 2)) / 2

        gen_mols, pure_valids = gen_mol_from_one_shot_tensor(gen_adj, gen_x)

        return gen_mols, pure_valids


