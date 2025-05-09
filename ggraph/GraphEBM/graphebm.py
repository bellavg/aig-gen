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
        The method class for GraphEBM algorithm proposed in the paper `GraphEBM: Molecular Graph Generation with Energy-Based Models <https://arxiv.org/abs/2102.00546>`_. This class provides interfaces for running random generation, goal-directed generation (including property
        optimization and constrained optimization), and compositional generation with GraphEBM algorithm. Please refer to the `example codes <https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph/GraphEBM>`_ for usage examples.

        Args:
            n_atom (int): Maximum number of atoms.
            n_atom_type (int): Number of possible atom types.
            n_edge_type (int): Number of possible bond types.
            hidden (int): Hidden dimensions.
            device (torch.device, optional): The device where the model is deployed.

    """

    def __init__(self, n_atom, n_atom_type, n_edge_type, hidden, device=None):
        super(GraphEBM, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.n_atom_type = n_atom_type + 1 # for virtual
        self.energy_function = EnergyFunc(self.n_atom_type, hidden, n_edge_type).to(self.device)
        self.n_atom = n_atom
        self.n_edge_type = n_edge_type
        self.num_actual_node_features = n_atom_type

    def _transform_node_features_add_virtual_channel(self, x_features_actual_BNF, device):
        """
        Transforms node features from (B, N, NUM_ACTUAL_FEATURES)
        to (B, MODEL_N_ATOM_TYPE, N) by adding a virtual node channel.
        MODEL_N_ATOM_TYPE = NUM_ACTUAL_FEATURES + 1.
        Assumes padded rows in x_features_actual_BNF are all zeros.
        The output shape is (B, F_total, N) as expected by the original EnergyFunc's 'h' input.
        """
        batch_size, num_nodes, num_actual_feats = x_features_actual_BNF.shape

        if num_actual_feats != self.num_actual_node_features:
            raise ValueError(
                f"Input x_features_actual_BNF has {num_actual_feats} features, "
                f"but model expects {self.num_actual_node_features} actual features."
            )

        # Target shape for EnergyFunc input h: (B, self.model_n_atom_type, N)
        x_transformed_BFN = torch.zeros(batch_size, self.n_atom_type, num_nodes, device=device)

        # Copy actual features. Input is (B,N,F_actual), permute to (B,F_actual,N) for assignment.
        x_transformed_BFN[:, :self.num_actual_node_features, :] = x_features_actual_BNF.permute(0, 2, 1)

        # Identify padded nodes. A node is padding if all its actual features are zero.
        # x_features_actual_BNF is (B, N, F_actual)
        is_padding_node_BN = (
                    x_features_actual_BNF.abs().sum(dim=2) < 1e-6)  # Check if sum of actual features is close to zero

        # For padded nodes, set the virtual channel (at virtual_node_channel_idx) to 1.
        # x_transformed_BFN is (B, F_total, N)
        # is_padding_node_BN is (B, N)
        for b in range(batch_size):
            padding_node_indices_in_sample = torch.where(is_padding_node_BN[b])[0]
            if len(padding_node_indices_in_sample) > 0:
                x_transformed_BFN[b, self.n_atom_type, padding_node_indices_in_sample] = 1.0
                # Ensure other channels are zero for these virtual nodes
                x_transformed_BFN[b, :self.num_actual_node_features, padding_node_indices_in_sample] = 0.0

        return x_transformed_BFN


    def train_rand_gen(self, loader, lr, wd, max_epochs, model_conf_dict,
                       save_interval, save_dir):
        r"""
            Running training for random generation task.

            Args:
                loader: The data loader for loading training samples. It is supposed to use dig.ggraph.dataset.QM9/ZINC250k
                    as the dataset class, and apply torch_geometric.data.DenseDataLoader to it to form the data loader.
                lr (float): The learning rate for training.
                wd (float): The weight decay factor for training.
                max_epochs (int): The maximum number of training epochs.
                c (float): The scaling hyperparameter for dequantization.
                ld_step (int): The number of iteration steps of Langevin dynamics.
                ld_noise (float): The standard deviation of the added noise in Langevin dynamics.
                ld_step_size (int): The step size of Langevin dynamics.
                clamp (bool): Whether to use gradient clamp in Langevin dynamics.
                alpha (float): The weight coefficient for loss function.
                save_interval (int): The frequency to save the model parameters to .pt files,
                    *e.g.*, if save_interval=2, the model parameters will be saved for every 2 training epochs.
                save_dir (str): the directory to save the model parameters.
        """
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
            losses_reg = []
            losses_en = []
            losses = []
            for _, batch in enumerate(tqdm(loader)):
                ### Dequantization
                pos_x = batch.x.to(self.device).to(dtype=torch.float32)
                pos_x = self._transform_node_features_add_virtual_channel(pos_x, self.device)
                pos_x += c * torch.rand_like(pos_x, device=self.device)
                pos_adj = batch.adj.to(self.device).to(dtype=torch.float32)
                pos_adj += c * torch.rand_like(pos_adj, device=self.device)

                ### Langevin dynamics
                neg_x = torch.rand_like(pos_x, device=self.device) * (1 + c)
                neg_adj = torch.rand_like(pos_adj, device=self.device)

                pos_adj = rescale_adj(pos_adj)
                neg_x.requires_grad = True
                neg_adj.requires_grad = True

                requires_grad(parameters, False)
                self.energy_function.eval()

                noise_x = torch.randn_like(neg_x, device=self.device)
                noise_adj = torch.randn_like(neg_adj, device=self.device)
                for _ in range(ld_step):

                    noise_x.normal_(0, ld_noise)
                    noise_adj.normal_(0, ld_noise)
                    neg_x.data.add_(noise_x.data)
                    neg_adj.data.add_(noise_adj.data)

                    neg_out = self.energy_function(neg_adj, neg_x)
                    neg_out.sum().backward()
                    if clamp:
                        neg_x.grad.data.clamp_(-0.01, 0.01)
                        neg_adj.grad.data.clamp_(-0.01, 0.01)

                    neg_x.data.add_(neg_x.grad.data, alpha=-ld_step_size)
                    neg_adj.data.add_(neg_adj.grad.data, alpha=-ld_step_size)

                    neg_x.grad.detach_()
                    neg_x.grad.zero_()
                    neg_adj.grad.detach_()
                    neg_adj.grad.zero_()

                    neg_x.data.clamp_(0, 1 + c)
                    neg_adj.data.clamp_(0, 1)

                ### Training by backprop
                neg_x = neg_x.detach()
                neg_adj = neg_adj.detach()
                requires_grad(parameters, True)
                self.energy_function.train()

                self.energy_function.zero_grad()

                pos_out = self.energy_function(pos_adj, pos_x)
                neg_out = self.energy_function(neg_adj, neg_x)

                loss_reg = (pos_out ** 2 + neg_out ** 2)  # energy magnitudes regularizer
                loss_en = pos_out - neg_out  # loss for shaping energy function
                loss = loss_en + alpha * loss_reg
                loss = loss.mean()
                loss.backward()
                clip_grad(optimizer)
                optimizer.step()

                losses_reg.append(loss_reg.mean())
                losses_en.append(loss_en.mean())
                losses.append(loss)

            t_end = time.time()

            ### Save checkpoints
            if (epoch + 1) % save_interval == 0:
                torch.save(self.energy_function.state_dict(), os.path.join(save_dir, 'epoch_{}.pt'.format(epoch + 1)))
                print('Saving checkpoint at epoch ', epoch + 1)
                print('==========================================')
            print(
                'Epoch: {:03d}, Loss: {:.6f}, Energy Loss: {:.6f}, Regularizer Loss: {:.6f}, Sec/Epoch: {:.2f}'.format(
                    epoch + 1, (sum(losses) / len(losses)).item(), (sum(losses_en) / len(losses_en)).item(),
                    (sum(losses_reg) / len(losses_reg)).item(), t_end - t_start))
            print('==========================================')

    def run_rand_gen(self, model_conf_dict, checkpoint_path, n_samples, atomic_num_list):
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

        gen_mols = gen_mol_from_one_shot_tensor(gen_adj, gen_x, atomic_num_list, correct_validity=True)

        return gen_mols


