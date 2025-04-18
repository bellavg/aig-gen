import os
import torch
from .generator import Generator
from .model import GraphFlowModel


class GraphDF(Generator):
    r"""
        The method class for GraphDF algorithm proposed in the paper `GraphDF: A Discrete Flow Model for Molecular Graph Generation <https://arxiv.org/abs/2102.01189>`_. This class provides interfaces for running random generation, property
        optimization, and constrained optimization with GraphDF algorithm. Please refer to the `example codes <https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph/GraphDF>`_ for usage examples.
    """

    def __init__(self):
        super(GraphDF, self).__init__()
        self.model = None

    def get_model(self, task, args, checkpoint_path=None):
        if task == 'rand_gen':
            self.model = GraphFlowModel(args)
        else:
            raise ValueError('Task {} is not supported in GraphDF!'.format(task))
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))

    def load_pretrain_model(self, path):
        load_key = torch.load(path)
        for key in load_key.keys():
            if key in self.model.state_dict().keys():
                self.model.state_dict()[key].copy_(load_key[key].detach().clone())


    def train_rand_gen(self, loader, args):
        """
        Running training for random generation task.
        """
        self.get_model('rand_gen', args)
        self.model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                     weight_decay=args.wd)
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)

        for epoch in range(1, args.max_epochs + 1):
            total_loss = 0

            for batch, data_batch in enumerate(loader):
                optimizer.zero_grad()

                # Extract input features and adjacency matrices
                inp_node_features = data_batch.x  # (B, N = 100, node_dim)
                inp_adj_features = data_batch.adj  # (B, edge_dim=2, N, N)

                # Forward pass
                out_z = self.model(inp_node_features, inp_adj_features)

                # Mask the loss to exclude padded regions
                loss = self.model.dis_log_prob(out_z)  # (B, N)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                print(f'Training iteration {batch} | loss {loss.item()}')

            avg_loss = total_loss / (batch + 1)
            print(f"Epoch {epoch} | Average loss {avg_loss}")

            if epoch % args.save_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(args.save_dir, f'rand_gen_ckpt_{epoch}.pth'))




    # def run_rand_gen(self, model_conf_dict, checkpoint_path, n_mols=100, num_min_node=7, num_max_node=25, temperature=[0.3, 0.3], atomic_num_list=[6, 7, 8, 9]):
    #     r"""
    #         Running graph generation for random generation task.
    #
    #         Args:
    #             model_conf_dict (dict): The python dict for configuring the model hyperparameters.
    #             checkpoint_path (str): The path to the saved model checkpoint file.
    #             n_mols (int, optional): The number of molecules to generate. (default: :obj:`100`)
    #             num_min_node (int, optional): The minimum number of nodes in the generated molecular graphs. (default: :obj:`7`)
    #             num_max_node (int, optional): the maximum number of nodes in the generated molecular graphs. (default: :obj:`25`)
    #             temperature (list, optional): a list of two float numbers, the temperature parameter of prior distribution. (default: :obj:`[0.3, 0.3]`)
    #             atomic_num_list (list, optional): a list of integers, the list of atomic numbers indicating the node types in the generated molecular graphs. (default: :obj:`[6, 7, 8, 9]`)
    #
    #         :rtype:
    #             (all_mols, pure_valids),
    #             all_mols is a list of generated molecules represented by rdkit Chem.Mol objects;
    #             pure_valids is a list of integers, all are 0 or 1, indicating whether bond resampling happens.
    #     """
    #
    #     self.get_model('rand_gen', model_conf_dict, checkpoint_path)
    #     self.model.eval()
    #     all_mols, pure_valids = [], []
    #     cnt_mol = 0
    #
    #     while cnt_mol < n_mols:
    #         mol, no_resample, num_atoms = self.model.generate(atom_list=atomic_num_list, min_atoms=num_min_node, max_atoms=num_max_node, temperature=temperature)
    #         if (num_atoms >= num_min_node):
    #             cnt_mol += 1
    #             all_mols.append(mol)
    #             pure_valids.append(no_resample)
    #             if cnt_mol % 10 == 0:
    #                 print('Generated {} molecules'.format(cnt_mol))
    #
    #     assert cnt_mol == n_mols, 'number of generated molecules does not equal num'
    #     return all_mols, pure_valids


