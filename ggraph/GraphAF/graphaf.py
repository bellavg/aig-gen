import os
import torch
from generator import Generator
from .model import GraphFlowModel



class GraphAF(Generator):
    r"""
        The method class for GraphAF algorithm proposed in the paper `GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation <https://arxiv.org/abs/2001.09382>`_. This class provides interfaces for running random generation, property
        optimization, and constrained optimization with GraphAF. Please refer to the `example codes <https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph/GraphAF>`_ for usage examples.
    """

    def __init__(self):
        super(GraphAF, self).__init__()
        self.model = None

    def get_model(self, task, model_conf_dict, checkpoint_path=None):
        if model_conf_dict['use_gpu'] and not torch.cuda.is_available():
            model_conf_dict['use_gpu'] = False
        if task == 'rand_gen':
            self.model = GraphFlowModel(model_conf_dict)
        else:
            raise ValueError('Task {} is not supported in GraphDF!'.format(task))
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))

    def load_pretrain_model(self, path):
        load_key = torch.load(path)
        for key in load_key.keys():
            if key in self.model.state_dict().keys():
                self.model.state_dict()[key].copy_(load_key[key].detach().clone())

    def train_rand_gen(self, loader, lr, wd, max_epochs, model_conf_dict, save_interval, save_dir):
        r"""
            Running training for random generation task.

            Args:
                loader: The data loader for loading training samples. It is supposed to use dig.ggraph.dataset.QM9/ZINC250k
                    as the dataset class, and apply torch_geometric.data.DenseDataLoader to it to form the data loader.
                lr (float): The learning rate for training.
                wd (float): The weight decay factor for training.
                max_epochs (int): The maximum number of training epochs.
                model_conf_dict (dict): The python dict for configuring the model hyperparameters.
                save_interval (int): Indicate the frequency to save the model parameters to .pth files,
                    *e.g.*, if save_interval=2, the model parameters will be saved for every 2 training epochs.
                save_dir (str): The directory to save the model parameters.
        """

        self.get_model('rand_gen', model_conf_dict)
        self.model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for epoch in range(1, max_epochs + 1):
            total_loss = 0
            for batch, data_batch in enumerate(loader):
                optimizer.zero_grad()
                inp_node_features = data_batch.x  # (B, N, node_dim)
                inp_adj_features = data_batch.adj  # (B, 4, N, N)
                if model_conf_dict['use_gpu']:
                    inp_node_features = inp_node_features.cuda()
                    inp_adj_features = inp_adj_features.cuda()

                out_z, out_logdet = self.model(inp_node_features, inp_adj_features)
                loss = self.model.log_prob(out_z, out_logdet)
                loss.backward()
                optimizer.step()

                total_loss += loss.to('cpu').item()
                if batch % 100 == 0:
                    print('Training iteration {} | loss {}'.format(batch, loss.to('cpu').item()))

            avg_loss = total_loss / (batch + 1)

            print("Epoch| Average loss {}".format(avg_loss))

            if epoch % save_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'rand_gen_ckpt_{}.pth'.format(epoch)))

    def run_rand_gen(self, model_conf_dict, checkpoint_path, n_mols=1000, num_min_node=5, num_max_node=64,
                     temperature=0.75):
        r"""
            Running graph generation for random generation task.

            Args:
                model_conf_dict (dict): The python dict for configuring the model hyperparameters.
                checkpoint_path (str): The path to the saved model checkpoint file.
                n_mols (int, optional): The number of molecules to generate. (default: :obj:`100`)
                num_min_node (int, optional): The minimum number of nodes in the generated molecular graphs. (default: :obj:`7`)
                num_max_node (int, optional): The maximum number of nodes in the generated molecular graphs. (default: :obj:`25`)
                temperature (float, optional): A float numbers, the temperature parameter of prior distribution. (default: :obj:`0.75`)
                atomic_num_list (list, optional): A list of integers, the list of atomic numbers indicating the node types in the generated molecular graphs. (default: :obj:`[6, 7, 8, 9]`)

            :rtype:
                (all_mols, pure_valids),
                all_mols is a list of generated molecules represented by rdkit Chem.Mol objects;
                pure_valids is a list of integers, all are 0 or 1, indicating whether bond resampling happens.
        """

        self.get_model('rand_gen', model_conf_dict, checkpoint_path)
        self.model.eval()
        all_mols, pure_valids = [], []
        cnt_mol = 0

        while cnt_mol < n_mols:
            mol, no_resample, num_atoms = self.model.generate(min_atoms=num_min_node,
                                                              max_atoms=num_max_node, temperature=temperature)
            if (num_atoms >= num_min_node):
                cnt_mol += 1
                all_mols.append(mol)
                pure_valids.append(no_resample)
                if cnt_mol % 10 == 0:
                    print('Generated {} molecules'.format(cnt_mol))

        assert cnt_mol == n_mols, 'number of generated molecules does not equal num'
        return all_mols, pure_valids