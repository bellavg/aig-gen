import os, torch, json, ast
import os.path as osp
import ssl
from itertools import repeat
import numpy as np
import pandas as pd
import networkx as nx
from six.moves import urllib
from torch_geometric.data import Data, InMemoryDataset, download_url

# bond_type_to_int = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}
# zinc_atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53]
# qm9_atom_list = [6, 7, 8, 9]
class PygDataset(InMemoryDataset):
    """
    A PyTorch Geometric data interface for datasets used in molecule generation.

    Args:
        file (str): Path to the raw data file.
        conf_dict (dict, optional): Configuration dictionary for dataset parameters.
        processed_filename (str, optional): Name of the processed data file. Default is 'data.pt'.
    """

    def __init__(self,
                 file,
                 conf_dict=None,
                 processed_filename='data.pt'):
        self.processed_filename = processed_filename
        self.raw_file = file

        # Use the provided configuration dictionary or set defaults
        self.conf_dict = conf_dict or {
            "num_max_nodes": 120,
            "node_dim": 4,
            "edge_dim": 2,
            "max_pis": 8,
            "edge_unroll": 14,
            "pad_value": -1
        }

        # Assign configurations
        self.num_max_nodes = self.conf_dict["num_max_nodes"]
        self.node_dim = self.conf_dict["node_dim"]
        self.edge_dim = self.conf_dict["edge_dim"]
        self.max_pis = self.conf_dict["max_pis"]
        self.edge_unroll = self.conf_dict["edge_unroll"]
        self.pad_value = self.conf_dict["pad_value"]

        super(PygDataset, self).__init__(file)
        self.process()

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return self.raw_file

    @property
    def processed_file_names(self):
        return self.processed_filename

    def download(self):
        print('Making raw files:', self.raw_dir)
        if not osp.exists(self.raw_dir):
            os.makedirs(self.raw_dir)

    def process(self):
        """Processes the dataset from raw data to the processed directory."""
        print('Processing...')
        self.data, self.slices = self.pre_process()

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        print('Making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        torch.save((self.data, self.slices), self.processed_paths[0])
        print('Done!')

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

    def get(self, idx):
        """
        Gets the data object at the specified index.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            Data: A PyTorch Geometric Data object.
        """
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data

    def get_split_idx(self):
        """
        Gets the train-validation split indices of the dataset.

        Returns:
            dict: A dictionary with keys 'train_idx' and 'valid_idx'.
        """
        if 'zinc250k' in self.name:
            split_file = 'valid_idx_zinc250k.json'
        elif 'qm9' in self.name:
            split_file = 'valid_idx_qm9.json'
        else:
            print('No available split file for this dataset, please check.')
            return None

        split_path = osp.join(self.root, 'raw', split_file)

        if not osp.exists(split_path):
            url = f'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/{split_file}'
            context = ssl._create_unverified_context()
            data = urllib.request.urlopen(url, context=context)
            with open(split_path, 'wb') as f:
                f.write(data.read())

        with open(split_path) as f:
            valid_idx = json.load(f)
            if 'qm9' in self.name:
                valid_idx = list(map(int, valid_idx['valid_idxs']))

        train_idx = list(set(np.arange(len(self))).difference(valid_idx))

        return {
            'train_idx': torch.tensor(train_idx, dtype=torch.long),
            'valid_idx': torch.tensor(valid_idx, dtype=torch.long)
        }

    def _bfs_seq(self, G, start_id):
        """
        Returns the BFS sequence starting from a specific node.

        Args:
            G (networkx.Graph): Input graph.
            start_id (int): Starting node for BFS.

        Returns:
            list: BFS sequence of node indices.
        """
        dictionary = dict(nx.bfs_successors(G, start_id))
        sequence = [start_id]
        queue = [start_id]

        while queue:
            current = queue.pop(0)
            neighbors = dictionary.get(current, [])
            sequence.extend(neighbors)
            queue.extend(neighbors)

        return sequence

#
#
# class PygDataset(InMemoryDataset):
#     """
#         A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for datasets used in molecule generation.
#
#         .. note::
#             Some datasets may not come with any node labels, like :obj:`moses`.
#             Since they don't have any properties in the original data file. The process of the
#             dataset can only save the current input property and will load the same  property
#             label when the processed dataset is used. You can change the augment :obj:`processed_filename`
#             to re-process the dataset with intended property.
#
#         Args:
#             root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
#             name (string, optional): The name of the dataset.  Available dataset names are as follows:
#                                     :obj:`zinc250k`, :obj:`zinc_800_graphaf`, :obj:`zinc_800_jt`, :obj:`zinc250k_property`,
#                                     :obj:`qm9_property`, :obj:`qm9`, :obj:`moses`. (default: :obj:`qm9`)
#
#             conf_dict (dictionary, optional): dictionary that stores all the configuration for the corresponding dataset. Default is None, but when something is passed, it uses its information. Useful for debugging and customizing for external contributers. (default: :obj:`False`)
#             transform (callable, optional): A function/transform that takes in an
#                 :obj:`torch_geometric.data.Data` object and returns a transformed
#                 version. The data object will be transformed before every access.
#                 (default: :obj:`None`)
#             pre_transform (callable, optional): A function/transform that takes in
#                 an :obj:`torch_geometric.data.Data` object and returns a
#                 transformed version. The data object will be transformed before
#                 being saved to disk. (default: :obj:`None`)
#             pre_filter (callable, optional): A function that takes in an
#                 :obj:`torch_geometric.data.Data` object and returns a boolean
#                 value, indicating whether the data object should be included in the
#                 final dataset. (default: :obj:`None`)
#             use_aug (bool, optional): If :obj:`True`, data augmentation will be used. (default: :obj:`False`)
#             one_shot (bool, optional): If :obj:`True`, the returned data will use one-shot format with an extra dimension of virtual node and edge feature. (default: :obj:`False`)
#         """
#
#     def __init__(self,
#                  file,
#                  conf_dict=None,
#                  processed_filename='data.pt',
#                  ):
#
#         self.processed_filename = processed_filename
#         self.raw_file = file
#
#         if conf_dict is None:
#             config_file = pd.read_csv(os.path.join(os.path.dirname(__file__), 'config.csv'), index_col = 0)
#             if not self.name in config_file:
#                 error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
#                 error_mssg += 'Available datasets are as follows:\n'
#                 error_mssg += '\n'.join(config_file.keys())
#                 raise ValueError(error_mssg)
#             config = config_file[self.name]
#
#         else:
#             config = conf_dict
#
#         self.num_max_node = int(config['num_max_nodes'])
#         self.node_type_dim = int(config['node_types'])  # Node types: PI, AND, PO, 0 (one-hot)
#         self.edge_type_dim = int(config['edge_types'])
#         self.max_pis = int(config['max_pis'])
#
#         super(PygDataset, self).__init__(file)
#         self.process()
#
#
#     @property
#     def raw_dir(self):
#         name = 'raw'
#         return osp.join(self.root, name)
#
#     @property
#     def processed_dir(self):
#         name = 'processed'
#         return osp.join(self.root, self.name, name)
#
#     @property
#     def raw_file_names(self):
#         return self.raw_file
#
#     @property
#     def processed_file_names(self):
#         return self.processed_filename
#
#     def download(self):
#         print('making raw files:', self.raw_dir)
#         if not osp.exists(self.raw_dir):
#             os.makedirs(self.raw_dir)
#
#
#     def process(self):
#         r"""Processes the dataset from raw data file to the :obj:`self.processed_dir` folder.
#
#             If one-hot format is required, the processed data type will include an extra dimension of virtual node and edge feature.
#         """
#
#         print('Processing...')
#         self.data, self.slices = self.pre_process()
#
#         if self.pre_filter is not None:
#             data_list = [self.get(idx) for idx in range(len(self))]
#             data_list = [data for data in data_list if self.pre_filter(data)]
#             self.data, self.slices = self.collate(data_list)
#
#         if self.pre_transform is not None:
#             data_list = [self.get(idx) for idx in range(len(self))]
#             data_list = [self.pre_transform(data) for data in data_list]
#             self.data, self.slices = self.collate(data_list)
#
#         print('making processed files:', self.processed_dir)
#         if not osp.exists(self.processed_dir):
#             os.makedirs(self.processed_dir)
#
#         torch.save((self.data, self.slices, self.all_smiles), self.processed_paths[0])
#         print('Done!')
#
#     def __repr__(self):
#         return '{}({})'.format(self.name, len(self))
#
#     def get(self, idx):
#         r"""Gets the data object at index :idx:.
#
#         Args:
#             idx: The index of the data that you want to reach.
#         :rtype: A data object corresponding to the input index :obj:`idx` .
#         """
#         data = self.data.__class__()
#
#         if hasattr(self.data, '__num_nodes__'):
#             data.num_nodes = self.data.__num_nodes__[idx]
#
#         for key in self.data.keys():
#             item, slices = self.data[key], self.slices[key]
#             if torch.is_tensor(item):
#                 s = list(repeat(slice(None), item.dim()))
#                 s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
#             else:
#                 s = slice(slices[idx], slices[idx + 1])
#             data[key] = item[s]
#
#         data['smile'] = self.all_smiles[idx]
#
#         if not self.one_shot:
#             # bfs-searching order
#             mol_size = data.num_atom.numpy()[0]
#             pure_adj = np.sum(data.adj[:3].numpy(), axis=0)[:mol_size, :mol_size]
#             if self.use_aug:
#                 local_perm = np.random.permutation(mol_size)
#                 adj_perm = pure_adj[np.ix_(local_perm, local_perm)]
#                 G = nx.from_numpy_matrix(np.asmatrix(adj_perm))
#                 start_idx = np.random.randint(adj_perm.shape[0])
#             else:
#                 local_perm = np.arange(mol_size)
#                 G = nx.from_numpy_matrix(np.asmatrix(pure_adj))
#                 start_idx = 0
#
#             bfs_perm = np.array(self._bfs_seq(G, start_idx))
#             bfs_perm_origin = local_perm[bfs_perm]
#             bfs_perm_origin = np.concatenate([bfs_perm_origin, np.arange(mol_size, self.num_max_node)])
#             data.x = data.x[bfs_perm_origin]
#             for i in range(4):
#                 data.adj[i] = data.adj[i][bfs_perm_origin][:,bfs_perm_origin]
#
#             data['bfs_perm_origin'] = torch.Tensor(bfs_perm_origin).long()
#
#         return data
#
#     # def pre_process(self):
#     #     input_path = self.raw_paths[0]
#     #     input_df = pd.read_csv(input_path, sep=',', dtype='str')
#     #     smile_list = list(input_df[self.smile_col])
#     #     if self.available_prop:
#     #             prop_list = list(input_df[self.prop_name])
#     #
#     #     self.all_smiles = smile_list
#     #     data_list = []
#     #
#     #     for i in range(len(smile_list)):
#     #         smile = smile_list[i]
#     #         mol = Chem.MolFromSmiles(smile)
#     #         Chem.Kekulize(mol)
#     #         num_atom = mol.GetNumAtoms()
#     #         if num_atom > self.num_max_node:
#     #             continue
#     #         else:
#     #             # atoms
#     #             atom_array = np.zeros((self.num_max_node, len(self.atom_list)), dtype=np.float32)
#     #
#     #             atom_idx = 0
#     #             for atom in mol.GetAtoms():
#     #                 atom_feature = atom.GetAtomicNum()
#     #                 atom_array[atom_idx, self.atom_list.index(atom_feature)] = 1
#     #                 atom_idx += 1
#     #
#     #             x = torch.tensor(atom_array)
#     #
#     #             # bonds
#     #             adj_array = np.zeros([4, self.num_max_node, self.num_max_node], dtype=np.float32)
#     #             for bond in mol.GetBonds():
#     #                 bond_type = bond.GetBondType()
#     #                 ch = bond_type_to_int[bond_type]
#     #                 i = bond.GetBeginAtomIdx()
#     #                 j = bond.GetEndAtomIdx()
#     #                 adj_array[ch, i, j] = 1.0
#     #                 adj_array[ch, j, i] = 1.0
#     #             adj_array[-1, :, :] = 1 - np.sum(adj_array, axis=0)
#     #             adj_array += np.eye(self.num_max_node)
#     #
#     #             data = Data(x=x)
#     #             data.adj = torch.tensor(adj_array)
#     #             data.num_atom = num_atom
#     #             if self.available_prop:
#     #                 data.y = torch.tensor([float(prop_list[i])])
#     #             data_list.append(data)
#     #
#     #     data, slices = self.collate(data_list)
#     #     return data, slices
#
# #     def one_shot_process(self):
# #         input_path = self.raw_paths[0]
# #         input_df = pd.read_csv(input_path, sep=',', dtype='str')
# #         smile_list = list(input_df[self.smile_col])
# #         if self.available_prop:
# #                 prop_list = list(input_df[self.prop_name])
# #
# #         self.all_smiles = smile_list
# #         data_list = []
# #
# #         for i in range(len(smile_list)):
# #             smile = smile_list[i]
# #             mol = Chem.MolFromSmiles(smile)
# #             Chem.Kekulize(mol)
# #             num_atom = mol.GetNumAtoms()
# #             if num_atom > self.num_max_node:
# #                 continue
# #             else:
# #                 # atoms
# #                 atom_array = np.zeros((len(self.atom_list), self.num_max_node), dtype=np.int32)
# #                 if self.one_shot:
# #                     virtual_node = np.ones((1, self.num_max_node), dtype=np.int32)
# #
# #                 atom_idx = 0
# #                 for atom in mol.GetAtoms():
# #                     atom_feature = atom.GetAtomicNum()
# # #                     print('self.atom_list','atom_feature', 'index')
# # #                     print(self.atom_list, atom_feature, self.atom_list.index(atom_feature))
# #                     atom_array[self.atom_list.index(atom_feature), atom_idx] = 1
# #                     if self.one_shot:
# #                         virtual_node[0, atom_idx] = 0
# #                     atom_idx += 1
# #
# #                 if self.one_shot:
# #                     x = torch.tensor(np.concatenate((atom_array, virtual_node), axis=0))
# #                 else:
# #                     x = torch.tensor(atom_array)
# #
# #                 # bonds
# #                 adj_array = np.zeros([4, self.num_max_node, self.num_max_node], dtype=np.float32)
# #                 for bond in mol.GetBonds():
# #                     bond_type = bond.GetBondType()
# #                     ch = bond_type_to_int[bond_type]
# #                     i = bond.GetBeginAtomIdx()
# #                     j = bond.GetEndAtomIdx()
# #                     adj_array[ch, i, j] = 1.0
# #                     adj_array[ch, j, i] = 1.0
# #                 adj_array[-1, :, :] = 1 - np.sum(adj_array, axis=0)
# #
# #                 data = Data(x=x)
# #                 data.adj = torch.tensor(adj_array)
# #                 data.num_atom = num_atom
# #                 if self.available_prop:
# #                     data.y = torch.tensor([float(prop_list[i])])
# #                 data_list.append(data)
# #
# #         data, slices = self.collate(data_list)
# #         return data, slices
#
#     def get_split_idx(self):
#         r"""Gets the train-valid set split indices of the dataset.
#
#         :rtype: A dictionary for training-validation split with key :obj:`train_idx` and :obj:`valid_idx`.
#         """
#         if self.name.find('zinc250k') != -1:
#             path = os.path.join(self.root, 'raw/valid_idx_zinc250k.json')
#
#             if not osp.exists(path):
#                 url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/valid_idx_zinc250k.json'
#                 context = ssl._create_unverified_context()
#                 data = urllib.request.urlopen(url, context=context)
#                 with open(path, 'wb') as f:
#                     f.write(data.read())
#             with open(path) as f:
#                 valid_idx = json.load(f)
#
#         elif self.name.find('qm9') != -1:
#             path = os.path.join(self.root, 'raw/valid_idx_qm9.json')
#
#             if not osp.exists(path):
#                 url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/valid_idx_qm9.json'
#                 context = ssl._create_unverified_context()
#                 data = urllib.request.urlopen(url, context=context)
#                 with open(path, 'wb') as f:
#                     f.write(data.read())
#             with open(path) as f:
#                 valid_idx = json.load(f)['valid_idxs']
#                 valid_idx = list(map(int, valid_idx))
#         else:
#             print('No available split file for this dataset, please check.')
#             return None
#
#         train_idx = list(set(np.arange(self.__len__())).difference(set(valid_idx)))
#
#         return {'train_idx': torch.tensor(train_idx, dtype = torch.long), 'valid_idx': torch.tensor(valid_idx, dtype = torch.long)}
#
#     def _bfs_seq(self, G, start_id):
#         dictionary = dict(nx.bfs_successors(G, start_id))
#         start = [start_id]
#         output = [start_id]
#         while len(start) > 0:
#             next_vertex = []
#             while len(start) > 0:
#                 current = start.pop(0)
#                 neighbor = dictionary.get(current)
#                 if neighbor is not None:
#                     next_vertex = next_vertex + neighbor
#             output = output + next_vertex
#             start = next_vertex
#         return output
#
#
#
#
