import os
import torch
import torch.nn as nn
# from rdkit import Chem
from .model import GraphFlowModel
from .train_utils import adjust_learning_rate, DataIterator
import pickle

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

    def train_prop_opt(self, *args, **kwargs):
        r"""
        Running training for property optimization task.
        """

        raise NotImplementedError("The function train_prop_opt is not implemented!")

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

        for epoch in range(1, max_epochs+1):
            total_loss = 0
            for batch, data_batch in enumerate(loader):
                optimizer.zero_grad()
                inp_node_features = data_batch.x #(B, N, node_dim)
                inp_adj_features = data_batch.adj #(B, 4, N, N)
                if model_conf_dict['use_gpu']:
                    inp_node_features = inp_node_features.cuda()
                    inp_adj_features = inp_adj_features.cuda()
                
                out_z, out_logdet = self.model(inp_node_features, inp_adj_features)
                loss = self.model.log_prob(out_z, out_logdet)
                loss.backward()
                optimizer.step()

                total_loss += loss.to('cpu').item()
                if batch % 500 == 0:
                    print('Training iteration {} | loss {}'.format(batch, loss.to('cpu').item()))

            avg_loss = total_loss / (batch + 1)
            print("Training | Average loss {}".format(avg_loss))
            
            if epoch % save_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'rand_gen_ckpt_{}.pth'.format(epoch)))


    def run_rand_gen(self, model_conf_dict, checkpoint_path,
                     num_samples=100, num_min_nodes=5,
                     # num_max_nodes_config is implicitly model_conf_dict['max_size'] used by GraphFlowModel
                     temperature=0.75, aig_node_types=None,
                     output_pickle_path="GraphAF_generated_aigs.pkl"):  # Default filename changed
        r"""
        Running AIG graph generation for random generation task using GraphAF.

        Args:
            model_conf_dict (dict): Python dict for configuring model hyperparameters.
                                    Must include 'node_dim' and 'max_size'.
            checkpoint_path (str): Path to the saved model checkpoint file.
            num_samples (int, optional): Number of AIGs to generate. (default: 100)
            num_min_nodes (int, optional): Minimum number of actual nodes in generated AIGs. (default: 5)
            temperature (float, optional): Temperature for sampling in the internal generation. (default: 0.75)
            aig_node_types (list, optional): List of AIG node type strings.
                                             Length must match model_conf_dict['node_dim'].
                                             Example: ['CONST0', 'PI', 'AND', 'PO'].
                                             If None, defaults to generic types based on node_dim.
            output_pickle_path (str, optional): Full path to save the list of generated AIGs.
                                                (default: "GraphAF_generated_aigs.pkl")

        :rtype:
            list: A list of generated AIGs, where each AIG is a networkx.DiGraph object.
        """
        node_dim_config = model_conf_dict.get('node_dim')
        max_size_config = model_conf_dict.get('max_size')

        if node_dim_config is None or max_size_config is None:
            raise ValueError("model_conf_dict must contain 'node_dim' and 'max_size'.")

        if aig_node_types is None:
            aig_node_types = [f"TYPE_{i}" for i in range(node_dim_config)]
            print(
                f"Warning: aig_node_types not provided. Using generic types based on node_dim={node_dim_config}: {aig_node_types}")

        if len(aig_node_types) != node_dim_config:
            raise ValueError(
                f"Length of aig_node_types ({len(aig_node_types)}) must match model_conf_dict['node_dim'] ({node_dim_config}).")

        # Initialize or get the model, ensuring it's on the correct device
        self.get_model('rand_gen_aig', model_conf_dict, checkpoint_path)
        self.model.eval()  # Set model to evaluation mode

        generated_aig_graphs = []
        generated_count = 0
        attempts = 0
        # Increase max_attempts if generation is often filtered out by num_min_nodes
        max_attempts = num_samples * 10

        print(f"Attempting to generate {num_samples} AIGs (min nodes: {num_min_nodes})...")

        # Determine device for generation calls from the model itself
        try:
            generation_device = next(self.model.parameters()).device
        except StopIteration:  # Model has no parameters
            generation_device = torch.device(
                "cuda" if model_conf_dict.get('use_gpu', False) and torch.cuda.is_available() else "cpu")
        print(f"Generation will run on device: {generation_device}")

        while generated_count < num_samples and attempts < max_attempts:
            attempts += 1

            if not hasattr(self.model, 'generate_aig_raw_data'):
                raise NotImplementedError(
                    "The internal model (GraphFlowModel) does not have the required "
                    "'generate_aig_raw_data(self, max_nodes, temperature, device)' method for AIG generation. "
                    "This method needs to be implemented in GraphFlowModel to return "
                    "(raw_node_features_tensor, raw_adj_matrix_tensor, actual_num_nodes_generated)."
                )

            try:
                # self.model.max_size comes from model_conf_dict['max_size']
                # raw_node_features: (max_size, node_dim)
                # raw_adj_matrix: (max_size, max_size) - representing directed edges
                # actual_nodes: int
                raw_node_features, raw_adj_matrix, actual_nodes = \
                    self.model.generate_aig_raw_data(max_nodes=max_size_config,  # Model's capacity
                                                     temperature=temperature,
                                                     device=generation_device)
            except Exception as e:
                print(f"Error during self.model.generate_aig_raw_data (attempt {attempts}): {e}")
                if attempts % (max_attempts // 10 if max_attempts > 10 else 1) == 0:  # Log progress periodically
                    print(f"Generation progress: {generated_count}/{num_samples} after {attempts} attempts.")
                continue

            if actual_nodes >= num_min_nodes:
                aig_graph = self._convert_raw_to_aig_digraph(raw_node_features, raw_adj_matrix, actual_nodes,
                                                             aig_node_types)
                if aig_graph is not None:
                    generated_aig_graphs.append(aig_graph)
                    generated_count += 1
                    if generated_count % 10 == 0 or generated_count == num_samples:
                        print(f"Successfully generated {generated_count}/{num_samples} AIGs.")

            if attempts > 0 and attempts % (max_attempts // 20 if max_attempts > 20 else 1) == 0:  # Log progress
                print(f"Generation progress: {generated_count}/{num_samples} after {attempts} attempts.")

        if generated_count < num_samples:
            print(
                f"Warning: Generated only {generated_count} AIGs after {max_attempts} attempts (target was {num_samples}).")

        try:
            output_dir = os.path.dirname(output_pickle_path)
            if output_dir and not os.path.exists(output_dir):  # Create directory if it doesn't exist
                os.makedirs(output_dir)
                print(f"Created directory for output: {output_dir}")

            with open(output_pickle_path, 'wb') as f:
                pickle.dump(generated_aig_graphs, f)
            print(f"Saved {len(generated_aig_graphs)} AIG DiGraphs to {output_pickle_path}")
        except Exception as e:
            print(f"Error saving AIGs to pickle file '{output_pickle_path}': {e}")

        return generated_aig_graphs


    def _convert_raw_to_aig_digraph(self, node_features_tensor, adj_matrix_tensor, num_actual_nodes, aig_node_types):
        """
        Converts raw model output (node features and adjacency matrix for a single graph)
        to a NetworkX DiGraph for AIGs.

        Args:
            node_features_tensor (torch.Tensor): Tensor of shape (max_nodes, node_dim)
                                                 representing node type probabilities or one-hot encodings.
            adj_matrix_tensor (torch.Tensor): Tensor of shape (max_nodes, max_nodes)
                                              representing the directed adjacency matrix (binary or scores).
            num_actual_nodes (int): The actual number of nodes in the graph (up to max_nodes).
            aig_node_types (list): List of AIG node type strings (e.g., ['CONST0', 'PI', 'AND', 'PO']).

        Returns:
            nx.DiGraph: A NetworkX directed graph representing the AIG, or None if conversion fails.
        """
        graph = nx.DiGraph()

        node_features = node_features_tensor.cpu().detach()
        adj_matrix = adj_matrix_tensor.cpu().detach()  # Shape: (max_nodes, max_nodes)

        # Add nodes up to num_actual_nodes
        for i in range(num_actual_nodes):
            try:
                node_type_idx = torch.argmax(node_features[i]).item()
                if 0 <= node_type_idx < len(aig_node_types):
                    node_type_label = aig_node_types[node_type_idx]
                else:
                    print(
                        f"Warning: Node {i} has invalid type index {node_type_idx}. Max index: {len(aig_node_types) - 1}. Setting type to UNKNOWN.")
                    node_type_label = "UNKNOWN"
                graph.add_node(i, type=node_type_label)
            except IndexError:
                print(f"Warning: Index error accessing node_features for node {i}. Skipping node.")
                return None  # Or handle more gracefully

        # Add directed edges among the actual nodes
        for i in range(num_actual_nodes):
            for j in range(num_actual_nodes):
                # Assuming adj_matrix contains scores/probabilities that can be thresholded,
                # or is already binary.
                if adj_matrix[i, j] > 0.5:
                    graph.add_edge(i, j)
        return graph

    #
    # def train_prop_optim(self, lr, wd, max_iters, warm_up, model_conf_dict, pretrain_path, save_interval, save_dir):
    #     r"""
    #         Running fine-tuning for property optimization task.
    #
    #         Args:
    #             lr (float): The learning rate for fine-tuning.
    #             wd (float): The weight decay factor for training.
    #             max_iters (int): The maximum number of training iters.
    #             warm_up (int): The number of linear warm-up iters.
    #             model_conf_dict (dict): The python dict for configuring the model hyperparameters.
    #             pretrain_path (str): The path to the saved pretrained model file.
    #             save_interval (int): Indicate the frequency to save the model parameters to .pth files,
    #                 *e.g.*, if save_interval=20, the model parameters will be saved for every 20 training iters.
    #             save_dir (str): The directory to save the model parameters.
    #     """
    #
    #
    #     self.get_model('prop_opt', model_conf_dict)
    #     self.load_pretrain_model(pretrain_path)
    #     self.model.train()
    #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
    #     if not os.path.isdir(save_dir):
    #         os.mkdir(save_dir)
    #
    #     print('start finetuning model(reinforce)')
    #     moving_baseline = None
    #     for cur_iter in range(max_iters):
    #         optimizer.zero_grad()
    #         loss, per_mol_reward, per_mol_property_score, moving_baseline = self.model.reinforce_forward_optim(in_baseline=moving_baseline, cur_iter=cur_iter)
    #
    #         num_mol = len(per_mol_reward)
    #         avg_reward = sum(per_mol_reward) / num_mol
    #         avg_score = sum(per_mol_property_score) / num_mol
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.flow_core.parameters()), 1.0)
    #         adjust_learning_rate(optimizer, cur_iter, lr, warm_up)
    #         optimizer.step()
    #
    #         print('Iter {} | reward {}, score {}, loss {}'.format(cur_iter, avg_reward, avg_score, loss.item()))
    #
    #         if cur_iter % save_interval == save_interval - 1:
    #             torch.save(self.model.state_dict(), os.path.join(save_dir, 'prop_opt_net_{}.pth'.format(cur_iter)))
    #
    #     print("Finetuning (Reinforce) Finished!")
    #
    #
    # def run_prop_optim(self, model_conf_dict, checkpoint_path, n_mols=100, num_min_node=7, num_max_node=25, temperature=0.75, atomic_num_list=[6, 7, 8, 9]):
    #     r"""
    #         Running graph generation for property optimization task.
    #
    #         Args:
    #             model_conf_dict (dict): The python dict for configuring the model hyperparameters.
    #             checkpoint_path (str): The path to the saved model checkpoint file.
    #             n_mols (int, optional): The number of molecules to generate. (default: :obj:`100`)
    #             num_min_node (int, optional): The minimum number of nodes in the generated molecular graphs. (default: :obj:`7`)
    #             num_max_node (int, optional): The maximum number of nodes in the generated molecular graphs. (default: :obj:`25`)
    #             temperature (float, optional): A float numbers, the temperature parameter of prior distribution. (default: :obj:`0.75`)
    #             atomic_num_list (list, optional): A list of integers, the list of atomic numbers indicating the node types in the generated molecular graphs. (default: :obj:`[6, 7, 8, 9]`)
    #
    #         :rtype:
    #             all_mols, a list of generated molecules represented by rdkit Chem.Mol objects.
    #     """
    #
    #     self.get_model('prop_opt', model_conf_dict, checkpoint_path)
    #     self.model.eval()
    #     all_mols, all_smiles = [], []
    #     cnt_mol = 0
    #
    #     while cnt_mol < n_mols:
    #         mol, num_atoms = self.model.reinforce_optim_one_mol(atom_list=atomic_num_list, max_size_rl=num_max_node, temperature=temperature)
    #         if mol is not None:
    #             smile = Chem.MolToSmiles(mol)
    #             if num_atoms >= num_min_node and not smile in all_smiles:
    #                 all_mols.append(mol)
    #                 all_smiles.append(smile)
    #                 cnt_mol += 1
    #                 if cnt_mol % 10 == 0:
    #                     print('Generated {} molecules'.format(cnt_mol))
    #
    #     return all_mols
    #
    #
    # def train_cons_optim(self, loader, lr, wd, max_iters, warm_up, model_conf_dict, pretrain_path, save_interval, save_dir):
    #     r"""
    #         Running fine-tuning for constrained optimization task.
    #
    #         Args:
    #             loader: The data loader for loading training samples. It is supposed to use dig.ggraph.dataset.ZINC800
    #                 as the dataset class, and apply torch_geometric.data.DenseDataLoader to it to form the data loader.
    #             lr (float): The learning rate for training.
    #             wd (float): The weight decay factor for training.
    #             max_iters (int): The maximum number of training iters.
    #             warm_up (int): The number of linear warm-up iters.
    #             model_conf_dict (dict): The python dict for configuring the model hyperparameters.
    #             pretrain_path (str): The path to the saved pretrained model parameters file.
    #             save_interval (int): Indicate the frequency to save the model parameters to .pth files,
    #                 *e.g.*, if save_interval=20, the model parameters will be saved for every 20 training iters.
    #             save_dir (str): The directory to save the model parameters.
    #     """
    #
    #     self.get_model('const_prop_opt', model_conf_dict)
    #     self.load_pretrain_model(pretrain_path)
    #     self.model.train()
    #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
    #     if not os.path.isdir(save_dir):
    #         os.mkdir(save_dir)
    #     loader = DataIterator(loader)
    #
    #     print('start finetuning model(reinforce)')
    #     moving_baseline = None
    #     for cur_iter in range(max_iters):
    #         optimizer.zero_grad()
    #         batch_data = next(loader)
    #         mol_xs = batch_data.x
    #         mol_adjs = batch_data.adj
    #         mol_sizes = batch_data.num_atom
    #         bfs_perm_origin = batch_data.bfs_perm_origin
    #         raw_smiles = batch_data.smile
    #
    #         loss, per_mol_reward, per_mol_property_score, moving_baseline = self.model.reinforce_forward_constrained_optim(
    #                                                 mol_xs=mol_xs, mol_adjs=mol_adjs, mol_sizes=mol_sizes, raw_smiles=raw_smiles,
    #                                                 bfs_perm_origin=bfs_perm_origin, in_baseline=moving_baseline, cur_iter=cur_iter)
    #
    #         num_mol = len(per_mol_reward)
    #         avg_reward = sum(per_mol_reward) / num_mol
    #         avg_score = sum(per_mol_property_score) / num_mol
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.flow_core.parameters()), 1.0)
    #         adjust_learning_rate(optimizer, cur_iter, lr, warm_up)
    #         optimizer.step()
    #
    #         print('Iter {} | reward {}, score {}, loss {}'.format(cur_iter, avg_reward, avg_score, loss.item()))
    #
    #         if cur_iter % save_interval == save_interval - 1:
    #             torch.save(self.model.state_dict(), os.path.join(save_dir, 'const_prop_opt_net_{}.pth'.format(cur_iter)))
    #
    #     print("Finetuning (Reinforce) Finished!")
    #
    #
    # def run_cons_optim_one_mol(self, adj, x, org_smile, mol_size, bfs_perm_origin, max_size_rl=38, temperature=0.70, atom_list=[6, 7, 8, 9]):
    #
    #     best_mol0 = None
    #     best_mol2 = None
    #     best_mol4 = None
    #     best_mol6 = None
    #     best_imp0 = -100.
    #     best_imp2 = -100.
    #     best_imp4 = -100.
    #     best_imp6 = -100.
    #     final_sim0 = -1.
    #     final_sim2 = -1.
    #     final_sim4 = -1.
    #     final_sim6 = -1.
    #
    #     mol_org = Chem.MolFromSmiles(org_smile)
    #     mol_org_size = mol_org.GetNumAtoms()
    #     assert mol_org_size == mol_size
    #
    #     cur_mols, cur_mol_imps, cur_mol_sims = self.model.reinforce_constrained_optim_one_mol(x, adj, mol_size, org_smile, bfs_perm_origin,
    #                                                                     atom_list=atom_list, temperature=temperature, max_size_rl=max_size_rl)
    #     num_success = len(cur_mol_imps)
    #     for i in range(num_success):
    #         cur_mol = cur_mols[i]
    #         cur_imp = cur_mol_imps[i]
    #         cur_sim = cur_mol_sims[i]
    #         assert cur_imp > 0
    #         if cur_sim > 0:
    #             if cur_imp > best_imp0:
    #                 best_mol0 = cur_mol
    #                 best_imp0 = cur_imp
    #                 final_sim0 = cur_sim
    #         if cur_sim > 0.2:
    #             if cur_imp > best_imp2:
    #                 best_mol2 = cur_mol
    #                 best_imp2 = cur_imp
    #                 final_sim2 = cur_sim
    #         if cur_sim > 0.4:
    #             if cur_imp > best_imp4:
    #                 best_mol4 = cur_mol
    #                 best_imp4 = cur_imp
    #                 final_sim4 = cur_sim
    #         if cur_sim > 0.6:
    #             if cur_imp > best_imp6:
    #                 best_mol6 = cur_mol
    #                 best_imp6 = cur_imp
    #                 final_sim6 = cur_sim
    #
    #     return [best_mol0, best_mol2, best_mol4, best_mol6], [best_imp0, best_imp2, best_imp4, best_imp6], [final_sim0, final_sim2, final_sim4, final_sim6]
    #
    #
    # def run_cons_optim(self, dataset, model_conf_dict, checkpoint_path, repeat_time=200, min_optim_time=50, num_max_node=25, temperature=0.7, atomic_num_list=[6, 7, 8, 9]):
    #     r"""
    #         Running molecule optimization for constrained optimization task.
    #
    #         Args:
    #             dataset: The dataset class for loading molecules to be optimized. It is supposed to use dig.ggraph.dataset.ZINC800 as the dataset class.
    #             model_conf_dict (dict): The python dict for configuring the model hyperparameters.
    #             checkpoint_path (str): The path to the saved model checkpoint file.
    #             repeat_time (int, optional): The maximum number of optimization times for each molecule before successfully optimizing it under the threshold 0.6.  (default: :obj:`200`)
    #             min_optim_time (int, optional): The minimum number of optimization times for each molecule. (default: :obj:`50`)
    #             num_max_node (int, optional): The maximum number of nodes in the optimized molecular graphs. (default: :obj:`25`)
    #             temperature (float, optional): A float numbers, the temperature parameter of prior distribution. (default: :obj:`0.75`)
    #             atomic_num_list (list, optional): A list of integers, the list of atomic numbers indicating the node types in the optimized molecular graphs. (default: :obj:`[6, 7, 8, 9]`)
    #
    #         :rtype:
    #             (mols_0, mols_2, mols_4, mols_6), they are lists of optimized molecules (represented by rdkit Chem.Mol objects) under the threshold 0.0, 0.2, 0.4, 0.6, respectively.
    #     """
    #
    #
    #     self.get_model('const_prop_opt', model_conf_dict, checkpoint_path)
    #     self.model.eval()
    #
    #     data_len = len(dataset)
    #     optim_success_dict = {}
    #     mols_0, mols_2, mols_4, mols_6 = [], [], [], []
    #     for batch_cnt in range(data_len):
    #         best_mol = [None, None, None, None]
    #         best_score = [-100., -100., -100., -100.]
    #         final_sim = [-1., -1., -1., -1.]
    #
    #         batch_data = dataset[batch_cnt] # dataloader is dataset object
    #
    #         inp_node_features = batch_data.x.unsqueeze(0) #(1, N, node_dim)
    #         inp_adj_features = batch_data.adj.unsqueeze(0) #(1, 4, N, N)
    #
    #         raw_smile = batch_data.smile  #(1)
    #         mol_size = batch_data.num_atom
    #         bfs_perm_origin = batch_data.bfs_perm_origin
    #
    #         for cur_iter in range(repeat_time):
    #             if raw_smile not in optim_success_dict:
    #                 optim_success_dict[raw_smile] = [0, -1] #(try_time, imp)
    #             if optim_success_dict[raw_smile][0] > min_optim_time and optim_success_dict[raw_smile][1] > 0: # reach min time and imp is positive
    #                 continue # not optimize this one
    #
    #             best_mol0246, best_score0246, final_sim0246 = self.run_cons_optim_one_mol(inp_adj_features,
    #                                                                 inp_node_features, raw_smile, mol_size, bfs_perm_origin, num_max_node, temperature, atomic_num_list)
    #             if best_score0246[0] > best_score[0]:
    #                 best_score[0] = best_score0246[0]
    #                 best_mol[0] = best_mol0246[0]
    #                 final_sim[0] = final_sim0246[0]
    #
    #             if best_score0246[1] > best_score[1]:
    #                 best_score[1] = best_score0246[1]
    #                 best_mol[1] = best_mol0246[1]
    #                 final_sim[1] = final_sim0246[1]
    #
    #             if best_score0246[2] > best_score[2]:
    #                 best_score[2] = best_score0246[2]
    #                 best_mol[2] = best_mol0246[2]
    #                 final_sim[2] = final_sim0246[2]
    #
    #             if best_score0246[3] > best_score[3]:
    #                 best_score[3] = best_score0246[3]
    #                 best_mol[3] = best_mol0246[3]
    #                 final_sim[3] = final_sim0246[3]
    #
    #             if best_score[3] > 0: #imp > 0
    #                 optim_success_dict[raw_smile][1] = best_score[3]
    #             optim_success_dict[raw_smile][0] += 1 # try time + 1
    #
    #         mols_0.append(best_mol[0])
    #         mols_2.append(best_mol[1])
    #         mols_4.append(best_mol[2])
    #         mols_6.append(best_mol[3])
    #
    #         if batch_cnt % 1 == 0:
    #             print('Optimized {} molecules'.format(batch_cnt+1))
    #
    #     return mols_0, mols_2, mols_4, mols_6
