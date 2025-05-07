import os
import torch
import torch.nn as nn
# from rdkit import Chem # Not needed for AIGs
from .model import GraphFlowModel  # Assuming .model points to the directory containing graphflow.py
from .train_utils import adjust_learning_rate, DataIterator
import pickle
import networkx as nx

# Try to import aig_config for type strings, handle if not found.
try:
    from G2PT.configs import aig_config  # Adjust this import path if necessary

    AIG_NODE_TYPE_KEYS = aig_config.NODE_TYPE_KEYS
    AIG_EDGE_TYPE_KEYS = aig_config.EDGE_TYPE_KEYS
except ImportError:
    print("Warning: G2PT.configs.aig_config not found. Using default AIG type keys.")
    AIG_NODE_TYPE_KEYS = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO']  # Default
    AIG_EDGE_TYPE_KEYS = ['EDGE_REG', 'EDGE_INV']  # Default


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
        # Determine the device based on model_conf_dict and availability
        use_gpu_config = model_conf_dict.get('use_gpu', False)
        if use_gpu_config and not torch.cuda.is_available():
            print("Warning: 'use_gpu' is True in config, but CUDA is not available. Falling back to CPU.")
            model_conf_dict['use_gpu'] = False  # Ensure model_conf_dict reflects actual device use
        elif not use_gpu_config:
            model_conf_dict['use_gpu'] = False  # Explicitly set to False if not specified or False

        # Instantiate the model
        # The task string 'rand_gen' should correctly instantiate GraphFlowModel
        if task == 'rand_gen':  # Or other tasks if GraphFlowModel handles them
            self.model = GraphFlowModel(model_conf_dict)
        else:
            # If you have different model classes for different tasks, handle them here
            raise ValueError('Task {} is not supported or model instantiation is not defined for it.'.format(task))

        # Load checkpoint if provided
        if checkpoint_path is not None:
            try:
                # Determine map_location based on where the model will run
                device_to_load_on = torch.device("cuda" if model_conf_dict['use_gpu'] else "cpu")
                state_dict = torch.load(checkpoint_path, map_location=device_to_load_on)

                # Handle DataParallel wrapper if present in checkpoint
                if isinstance(self.model, nn.DataParallel) or any(key.startswith('module.') for key in state_dict):
                    # If current model is not DataParallel but checkpoint is, strip 'module.'
                    if not isinstance(self.model, nn.DataParallel) and all(
                            key.startswith('module.') for key in state_dict):
                        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
                        self.model.load_state_dict(new_state_dict)
                    # If current model is DataParallel and checkpoint is not, it might load fine or need adjustment
                    # If both are DataParallel or both are not (and no 'module.' prefix), load directly
                    else:
                        self.model.load_state_dict(state_dict)
                else:  # Neither model nor checkpoint seems to be from DataParallel, or model is not yet wrapped
                    self.model.load_state_dict(state_dict)
                print(f"Successfully loaded checkpoint from {checkpoint_path} to device {device_to_load_on}")

            except Exception as e:
                print(f"Error loading checkpoint from {checkpoint_path}: {e}")
                raise  # Re-raise the exception to halt if loading fails

        # After loading state_dict, explicitly move the model to the target device
        # This is important especially if the checkpoint was saved on a different device type
        # or if DataParallel is used.
        final_device = torch.device("cuda" if model_conf_dict['use_gpu'] and torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model.to(final_device)
            print(f"GraphAF model assigned to device: {final_device}")
        else:
            print("Error: Model is None after get_model call.")

    def load_pretrain_model(self, path):
        # This method might be redundant if get_model handles checkpoint loading robustly.
        # For safety, ensuring device consistency here too.
        try:
            # Determine device from model if it exists, otherwise default to CPU for loading
            current_model_device = next(self.model.parameters()).device if self.model and len(
                list(self.model.parameters())) > 0 else torch.device("cpu")
            load_key = torch.load(path, map_location=current_model_device)

            # If model is wrapped in DataParallel
            is_model_dataparallel = isinstance(self.model, nn.DataParallel)
            actual_model_state_dict = self.model.module.state_dict() if is_model_dataparallel else self.model.state_dict()

            # Check if checkpoint keys are prefixed with 'module.'
            checkpoint_is_dataparallel = any(key.startswith('module.') for key in load_key.keys())

            temp_load_dict = {}
            for key_ckpt, value_ckpt in load_key.items():
                key_model = key_ckpt
                if checkpoint_is_dataparallel and not is_model_dataparallel:  # ckpt has 'module.', model doesn't
                    key_model = key_ckpt[7:] if key_ckpt.startswith('module.') else key_ckpt
                elif not checkpoint_is_dataparallel and is_model_dataparallel:  # ckpt no 'module.', model has
                    # This case is tricky, usually load_state_dict handles it if strict=False,
                    # or one might need to prefix model keys. For simplicity, assume direct match or model handles it.
                    pass  # Keep key_model as key_ckpt

                if key_model in actual_model_state_dict:
                    if actual_model_state_dict[key_model].shape == value_ckpt.shape:
                        temp_load_dict[key_model] = value_ckpt.detach().clone()
                    else:
                        print(
                            f"Shape mismatch for key {key_model}: model {actual_model_state_dict[key_model].shape}, checkpoint {value_ckpt.shape}. Skipping.")
                else:
                    print(f"Key {key_model} (from ckpt key {key_ckpt}) not found in model state_dict. Skipping.")

            if is_model_dataparallel:
                self.model.module.load_state_dict(temp_load_dict, strict=False)
            else:
                self.model.load_state_dict(temp_load_dict, strict=False)
            print(f"Loaded pretrain model from {path} with {len(temp_load_dict)} matching keys.")

        except Exception as e:
            print(f"Error in load_pretrain_model from {path}: {e}")

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

        self.get_model('rand_gen', model_conf_dict)  # Model is moved to device in get_model
        self.model.train()

        # Determine device from model_conf_dict for optimizer and data
        device = torch.device("cuda" if model_conf_dict.get('use_gpu', False) and torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)  # Use exist_ok=True

        for epoch in range(1, max_epochs + 1):
            total_loss = 0
            batch_count = 0
            for batch_idx, data_batch in enumerate(loader):
                optimizer.zero_grad()
                # Move data to the same device as the model
                inp_node_features = data_batch.x.to(device)  # (B, N, node_dim)
                inp_adj_features = data_batch.adj.to(device)  # (B, bond_dim, N, N)
                # For AIG, bond_dim is 3 (REG, INV, NO_EDGE)

                out_z, out_logdet = self.model(inp_node_features, inp_adj_features)
                loss = self.model.log_prob(out_z, out_logdet)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()  # No need to move to CPU if already scalar
                batch_count += 1
                if batch_idx % 500 == 0:  # Use batch_idx for iteration number
                    print('Training Epoch {} | Iteration {} | loss {}'.format(epoch, batch_idx, loss.item()))

            if batch_count > 0:
                avg_loss = total_loss / batch_count
                print("Training Epoch {} | Average loss {}".format(epoch, avg_loss))
            else:
                print(f"Training Epoch {epoch} | No data processed.")

            if epoch % save_interval == 0:
                # Save model state dict (handles DataParallel correctly)
                model_state_to_save = self.model.module.state_dict() if isinstance(self.model,
                                                                                   nn.DataParallel) else self.model.state_dict()
                torch.save(model_state_to_save, os.path.join(save_dir, 'rand_gen_ckpt_{}.pth'.format(epoch)))
                print(
                    f"Saved model checkpoint at epoch {epoch} to {os.path.join(save_dir, 'rand_gen_ckpt_{}.pth'.format(epoch))}")

    def run_rand_gen(self, model_conf_dict, checkpoint_path,
                     num_samples=100, num_min_nodes=5,
                     temperature=0.75, aig_node_type_strings=None,  # Renamed for clarity
                     output_pickle_path="GraphAF_generated_aigs.pkl"):
        r"""
        Running AIG graph generation for random generation task using GraphAF.

        Args:
            model_conf_dict (dict): Python dict for configuring model hyperparameters.
                                    Must include 'node_dim' (e.g., 4 for AIGs) and 'max_size'.
                                    'bond_dim' (e.g., 3 for AIGs) is also crucial for the model.
            checkpoint_path (str): Path to the saved model checkpoint file.
            num_samples (int, optional): Number of AIGs to generate. (default: 100)
            num_min_nodes (int, optional): Minimum number of actual nodes in generated AIGs. (default: 5)
            temperature (float, optional): Temperature for sampling. (default: 0.75)
            aig_node_type_strings (list, optional): List of AIG node type strings.
                                             Length must match model_conf_dict['node_dim'].
                                             Example: ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO'].
                                             If None, uses defaults from AIG_NODE_TYPE_KEYS.
            output_pickle_path (str, optional): Full path to save the list of generated AIGs.
                                                (default: "GraphAF_generated_aigs.pkl")

        :rtype:
            list: A list of generated AIGs, where each AIG is a networkx.DiGraph object.
        """
        node_dim_config = model_conf_dict.get('node_dim')
        max_size_config = model_conf_dict.get('max_size')
        bond_dim_config = model_conf_dict.get('bond_dim')  # Important for AIG generation

        if node_dim_config is None or max_size_config is None or bond_dim_config is None:
            raise ValueError("model_conf_dict must contain 'node_dim', 'max_size', and 'bond_dim'.")

        if aig_node_type_strings is None:
            aig_node_type_strings = AIG_NODE_TYPE_KEYS  # Use imported or default
            print(f"Using AIG node type strings: {aig_node_type_strings}")

        # Define AIG edge type strings based on your config (these are for the final NetworkX graph)
        # The model internally uses bond_dim (e.g., 3 channels: REG, INV, NO_EDGE)
        # The conversion function will map generated edge category indices to these strings.
        # Typically, index 0 -> 'EDGE_REG', index 1 -> 'EDGE_INV'
        aig_edge_type_strings = AIG_EDGE_TYPE_KEYS  # Should be ['EDGE_REG', 'EDGE_INV']

        if len(aig_node_type_strings) != node_dim_config:
            raise ValueError(
                f"Length of aig_node_type_strings ({len(aig_node_type_strings)}) must match model_conf_dict['node_dim'] ({node_dim_config}).")

        # Initialize or get the model. 'rand_gen' task should instantiate GraphFlowModel.
        # The model is moved to the correct device within get_model.
        self.get_model('rand_gen', model_conf_dict, checkpoint_path)
        self.model.eval()

        generated_aig_graphs = []
        generated_count = 0
        attempts = 0
        max_attempts = num_samples * 20  # Increased attempts for better chance of meeting num_min_nodes

        print(f"Attempting to generate {num_samples} AIGs (min nodes: {num_min_nodes}, max_size: {max_size_config})...")

        # Determine device for generation calls from the model itself (it's already on this device)
        try:
            # Check if model has parameters and get device of the first one
            if len(list(self.model.parameters())) > 0:
                generation_device = next(self.model.parameters()).device
            else:  # Model has no parameters, fallback to config (should not happen for GraphFlowModel)
                generation_device = torch.device(
                    "cuda" if model_conf_dict.get('use_gpu', False) and torch.cuda.is_available() else "cpu")
        except StopIteration:
            generation_device = torch.device(
                "cuda" if model_conf_dict.get('use_gpu', False) and torch.cuda.is_available() else "cpu")

        print(f"Generation will run on device: {generation_device}")

        while generated_count < num_samples and attempts < max_attempts:
            attempts += 1
            if attempts % (max_attempts // 20 if max_attempts >= 20 else 1) == 0:
                print(f"Attempt {attempts}/{max_attempts}, Generated {generated_count}/{num_samples}")

            if not hasattr(self.model, 'generate_aig_raw_data'):
                raise NotImplementedError(
                    "The internal model (GraphFlowModel) does not have the required "
                    "'generate_aig_raw_data(self, max_nodes, temperature, device)' method. "
                    "This method should return (raw_node_features_tensor, typed_edges_list, actual_num_nodes)."
                )

            try:
                # generate_aig_raw_data now returns (raw_node_features, typed_edges_list, actual_nodes)
                raw_node_features, typed_edges_list, actual_nodes = \
                    self.model.generate_aig_raw_data(max_nodes=max_size_config,
                                                     temperature=temperature,
                                                     device=generation_device)
            except Exception as e:
                print(f"Error during self.model.generate_aig_raw_data (attempt {attempts}): {e}")
                # Optionally add a small delay or specific error handling here
                continue

            if actual_nodes >= num_min_nodes:
                # Pass both node and edge type strings to the conversion function
                aig_graph = self._convert_raw_to_aig_digraph(raw_node_features,
                                                             typed_edges_list,
                                                             actual_nodes,
                                                             aig_node_type_strings,
                                                             aig_edge_type_strings)
                if aig_graph is not None:
                    generated_aig_graphs.append(aig_graph)
                    generated_count += 1
                    if generated_count % 10 == 0 or generated_count == num_samples:
                        print(f"Successfully generated {generated_count}/{num_samples} AIGs.")

        if generated_count < num_samples:
            print(
                f"Warning: Generated only {generated_count} AIGs after {max_attempts} attempts (target was {num_samples}).")

        try:
            output_dir = os.path.dirname(output_pickle_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created directory for output: {output_dir}")

            with open(output_pickle_path, 'wb') as f:
                pickle.dump(generated_aig_graphs, f)
            print(f"Saved {len(generated_aig_graphs)} AIG DiGraphs to {output_pickle_path}")
        except Exception as e:
            print(f"Error saving AIGs to pickle file '{output_pickle_path}': {e}")

        return generated_aig_graphs

    def _convert_raw_to_aig_digraph(self, node_features_tensor, typed_edges_list,
                                    num_actual_nodes, aig_node_type_strings, aig_edge_type_strings):
        """
        Converts raw model output (node features and a list of typed edges)
        to a NetworkX DiGraph for AIGs.

        Args:
            node_features_tensor (torch.Tensor): Tensor of shape (max_nodes, node_dim)
                                                 representing node type probabilities/scores.
            typed_edges_list (list): List of tuples (source_idx, target_idx, edge_type_idx),
                                     where edge_type_idx is 0 for 'EDGE_REG', 1 for 'EDGE_INV'.
            num_actual_nodes (int): The actual number of nodes in the graph (up to max_nodes).
            aig_node_type_strings (list): List of AIG node type strings (e.g., ['NODE_CONST0', ...]).
            aig_edge_type_strings (list): List of AIG edge type strings (e.g., ['EDGE_REG', 'EDGE_INV']).

        Returns:
            nx.DiGraph: A NetworkX directed graph representing the AIG, or None if conversion fails.
        """
        graph = nx.DiGraph()
        node_features = node_features_tensor.cpu().detach()  # Ensure on CPU for numpy/python operations

        # Add nodes up to num_actual_nodes
        for i in range(num_actual_nodes):
            try:
                # Determine node type by taking argmax of the feature vector for node i
                node_type_idx = torch.argmax(node_features[i]).item()
                if 0 <= node_type_idx < len(aig_node_type_strings):
                    node_type_label = aig_node_type_strings[node_type_idx]
                else:
                    print(
                        f"Warning: Node {i} has invalid type index {node_type_idx} (max allowed: {len(aig_node_type_strings) - 1}). Setting type to 'UNKNOWN_NODE_TYPE'.")
                    node_type_label = "UNKNOWN_NODE_TYPE"  # Or handle as an error
                graph.add_node(i, type=node_type_label)
            except IndexError:
                print(
                    f"Error: Index out of bounds when accessing node_features for node {i}. num_actual_nodes={num_actual_nodes}, node_features.shape={node_features.shape}. Skipping graph.")
                return None
            except Exception as e:
                print(f"Error processing node {i}: {e}. Skipping graph.")
                return None

        # Add typed directed edges from the typed_edges_list
        for u, v, edge_type_idx in typed_edges_list:
            # Ensure source and target nodes are within the actual number of nodes
            if u < num_actual_nodes and v < num_actual_nodes:
                if 0 <= edge_type_idx < len(aig_edge_type_strings):
                    edge_type_label = aig_edge_type_strings[edge_type_idx]
                    graph.add_edge(u, v, type=edge_type_label)
                else:
                    print(
                        f"Warning: Edge ({u}->{v}) has invalid edge_type_idx {edge_type_idx} (max allowed: {len(aig_edge_type_strings) - 1}). Adding edge without 'type' attribute.")
                    graph.add_edge(u, v)  # Add edge without type if index is bad
            else:
                print(
                    f"Warning: Skipping edge ({u}->{v}) because node indices are out of range for actual_num_nodes={num_actual_nodes}.")

        return graph

    # --- Methods for property optimization (train_prop_opt, run_prop_opt, etc.) are unchanged ---
    # ... (original property optimization methods would be here) ...
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

