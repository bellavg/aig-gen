import os
import torch
import torch.nn as nn
import networkx as nx  # For DiGraph
import pickle  # For saving generated AIGs
from collections import OrderedDict  # For robust checkpoint loading

# Assuming GraphFlowModel is in .model submodule
# GraphDF uses GraphFlowModel as its internal model.
from .model import GraphFlowModel


# from .train_utils import adjust_learning_rate, DataIterator # Keep if other tasks are used

class Generator():
    r"""
    The method base class for graph generation.
    """

    def train_rand_gen(self, loader, *args, **kwargs):
        raise NotImplementedError("The function train_rand_gen is not implemented!")

    def run_rand_gen(self, *args, **kwargs):
        raise NotImplementedError("The function run_rand_gen is not implemented!")
    # Add other abstract methods if GraphDF implements other tasks (prop_opt, etc.)


class GraphDF(Generator):
    r"""
        The method class for GraphDF algorithm.
        Adapted for AIG generation.
    """

    def __init__(self):
        super(GraphDF, self).__init__()
        self.model = None  # Will be an instance of GraphFlowModel

    def get_model(self, task_conceptual_name, model_conf_dict, checkpoint_path=None):
        """
        Initializes or loads the GraphFlowModel for GraphDF.
        Args:
            task_conceptual_name (str): Conceptual name of the task, e.g., 'rand_gen_aig'.
            model_conf_dict (dict): Configuration dictionary for the model.
            checkpoint_path (str, optional): Path to a pre-trained checkpoint.
        """
        use_gpu_config = model_conf_dict.get('use_gpu', False)
        if use_gpu_config and not torch.cuda.is_available():
            print("Warning: CUDA requested in config but not available. Using CPU.")
            model_conf_dict['use_gpu'] = False  # Update conf
            target_device = torch.device("cpu")
        elif use_gpu_config:
            target_device = torch.device("cuda")
        else:
            target_device = torch.device("cpu")

        if task_conceptual_name == 'rand_gen' or task_conceptual_name == 'rand_gen_aig' or task_conceptual_name == 'rand_gen_train':  # Common task name for training/generation
            self.model = GraphFlowModel(model_conf_dict)
            print(f"GraphFlowModel instantiated for GraphDF. Target device: {target_device}")
        else:
            raise ValueError('Task {} is not supported in GraphDF!'.format(task_conceptual_name))

        if checkpoint_path is not None:
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                is_data_parallel = next(iter(state_dict)).startswith('module.')
                if is_data_parallel:
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    self.model.load_state_dict(new_state_dict)
                    print(f"Loaded DataParallel checkpoint for GraphDF from {checkpoint_path}.")
                else:
                    self.model.load_state_dict(state_dict)
                    print(f"Loaded checkpoint for GraphDF from {checkpoint_path}.")
            except FileNotFoundError:
                print(f"Error: Checkpoint file not found at {checkpoint_path}")
                raise
            except Exception as e:
                print(f"Error loading checkpoint for GraphDF: {e}")
                raise

        self.model.to(target_device)
        model_device = next(self.model.parameters()).device if list(self.model.parameters()) else target_device
        print(f"GraphDF's model ('{task_conceptual_name}') is now on device: {model_device}")

    def load_pretrain_model(self, path, model_conf_dict):
        """ Loads pre-trained weights into the existing self.model for GraphDF. """
        if self.model is None:
            # Initialize model if not already done, e.g. if run_rand_gen is called directly with a pretrained model
            print("Model not initialized in GraphDF. Calling get_model first.")
            self.get_model('rand_gen_aig_pretrain_load', model_conf_dict)  # Use a generic task name

        print(f"Loading pre-trained weights from {path} into GraphDF's model.")
        load_key = torch.load(path, map_location='cpu')

        is_data_parallel = next(iter(load_key)).startswith('module.')
        if is_data_parallel:
            new_state_dict = OrderedDict()
            for k, v in load_key.items():
                name = k[7:]
                new_state_dict[name] = v
            load_key = new_state_dict
            print("Adjusted pre-trained keys from DataParallel format for GraphDF.")

        self.model.load_state_dict(load_key, strict=False)
        print(f"Successfully loaded pre-trained weights into GraphDF's model from {path}.")

        use_gpu_config = model_conf_dict.get('use_gpu', False)
        target_device = torch.device("cuda" if use_gpu_config and torch.cuda.is_available() else "cpu")
        self.model.to(target_device)
        model_device = next(self.model.parameters()).device if list(self.model.parameters()) else target_device
        print(f"GraphDF's model with pre-trained weights is now on device: {model_device}")

    def train_rand_gen(self, loader, lr, wd, max_epochs, model_conf_dict, save_interval, save_dir):
        r"""
            Running training for random generation task with GraphDF.
            Uses discrete log probability (dis_log_prob).
        """
        self.get_model('rand_gen_train', model_conf_dict)
        current_device = next(self.model.parameters()).device
        self.model.train()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            print(f"Created save directory: {save_dir}")

        print(f"Starting GraphDF training on device: {current_device}")
        for epoch in range(1, max_epochs + 1):
            total_loss = 0
            processed_batches = 0
            for batch_idx, data_batch in enumerate(loader):
                optimizer.zero_grad()
                inp_node_features = data_batch.x.to(current_device)
                inp_adj_features = data_batch.adj.to(current_device)

                # GraphDF uses discrete log probability
                # The forward pass of GraphFlowModel is called to get 'out_z'
                # which are then interpreted by dis_log_prob
                out_z = self.model(inp_node_features, inp_adj_features)  # This calls GraphFlowModel.forward()
                loss = self.model.dis_log_prob(out_z)  # GraphFlowModel must have dis_log_prob

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                processed_batches += 1
                if batch_idx % 200 == 0:  # Log more frequently
                    print(
                        'Epoch {}/{} | Batch {}/{} | GraphDF Training Loss: {:.4f}'.format(epoch, max_epochs, batch_idx,
                                                                                           len(loader), loss.item()))

            avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
            print("Epoch {}/{} | GraphDF Average Training Loss: {:.4f}".format(epoch, max_epochs, avg_loss))

            if epoch % save_interval == 0:
                ckpt_path = os.path.join(save_dir, 'graphdf_rand_gen_ckpt_epoch_{}.pth'.format(epoch))
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Saved GraphDF checkpoint: {ckpt_path}")
        print("GraphDF training finished.")

    def _convert_raw_to_aig_digraph(self, node_features_tensor, adj_matrix_tensor, num_actual_nodes, aig_node_types):
        """
        Converts raw model output (node features and adjacency matrix for a single graph)
        to a NetworkX DiGraph for AIGs. (Identical to GraphAF's helper)
        """
        graph = nx.DiGraph()
        node_features = node_features_tensor.cpu().detach()
        adj_matrix = adj_matrix_tensor.cpu().detach()

        for i in range(num_actual_nodes):
            try:
                node_type_idx = torch.argmax(node_features[i]).item()
                if 0 <= node_type_idx < len(aig_node_types):
                    node_type_label = aig_node_types[node_type_idx]
                else:
                    node_type_label = "UNKNOWN"
                graph.add_node(i, type=node_type_label)
            except IndexError:
                print(f"Warning: Index error for node {i} features. Skipping node.")
                return None

        for i in range(num_actual_nodes):
            for j in range(num_actual_nodes):
                if adj_matrix[i, j] > 0.5:
                    graph.add_edge(i, j)
        return graph

    def run_rand_gen(self, model_conf_dict, checkpoint_path,
                     num_samples=100, num_min_nodes=5,
                     temperature=[0.6, 0.6],  # GraphDF often uses [temp_node, temp_edge]
                     aig_node_types=None,
                     output_pickle_path="GraphDF_generated_aigs.pkl"):
        r"""
        Running AIG graph generation for random generation task using GraphDF.

        Args:
            model_conf_dict (dict): Python dict for configuring model hyperparameters.
                                    Must include 'node_dim' and 'max_size'.
            checkpoint_path (str): Path to the saved model checkpoint file.
            num_samples (int, optional): Number of AIGs to generate. (default: 100)
            num_min_nodes (int, optional): Minimum number of actual nodes in generated AIGs. (default: 5)
            temperature (list, optional): List of two floats [temp_node, temp_edge] for discrete sampling.
                                          (default: [0.6, 0.6])
            aig_node_types (list, optional): List of AIG node type strings.
                                             Length must match model_conf_dict['node_dim'].
            output_pickle_path (str, optional): Full path to save the list of generated AIGs.

        :rtype:
            list: A list of generated AIGs, where each AIG is a networkx.DiGraph object.
        """
        node_dim_config = model_conf_dict.get('node_dim')
        max_size_config = model_conf_dict.get('max_size')

        if node_dim_config is None or max_size_config is None:
            raise ValueError("model_conf_dict must contain 'node_dim' and 'max_size'.")

        if aig_node_types is None:
            aig_node_types = [f"TYPE_{i}" for i in range(node_dim_config)]
            print(f"Warning: aig_node_types not provided. Using generic types: {aig_node_types}")

        if len(aig_node_types) != node_dim_config:
            raise ValueError(
                f"Length of aig_node_types ({len(aig_node_types)}) must match model_conf_dict['node_dim'] ({node_dim_config}).")

        if not (isinstance(temperature, list) and len(temperature) == 2):
            print(
                f"Warning: GraphDF temperature expected as [temp_node, temp_edge]. Received {temperature}. Using first element for both if possible, or defaulting.")
            if isinstance(temperature, (float, int)):
                temperature = [temperature, temperature]
            elif isinstance(temperature, list) and len(temperature) > 0:
                temperature = [temperature[0], temperature[0]]
            else:
                temperature = [0.6, 0.6]  # Default fallback

        self.get_model('rand_gen_aig', model_conf_dict, checkpoint_path)
        self.model.eval()

        generated_aig_graphs = []
        generated_count = 0
        attempts = 0
        max_attempts = num_samples * 10

        print(f"Attempting to generate {num_samples} AIGs with GraphDF (min nodes: {num_min_nodes})...")

        try:
            generation_device = next(self.model.parameters()).device
        except StopIteration:
            generation_device = torch.device(
                "cuda" if model_conf_dict.get('use_gpu', False) and torch.cuda.is_available() else "cpu")
        print(f"GraphDF generation will run on device: {generation_device}")

        while generated_count < num_samples and attempts < max_attempts:
            attempts += 1

            if not hasattr(self.model, 'generate_aig_discrete_raw_data'):
                raise NotImplementedError(
                    "The internal model (GraphFlowModel) does not have the required "
                    "'generate_aig_discrete_raw_data(self, max_nodes, temperature_node, temperature_edge, device)' method for GraphDF's AIG generation. "
                    "This method needs to be implemented in GraphFlowModel."
                )

            try:
                # Call the new discrete generation method
                raw_node_features, raw_adj_matrix, actual_nodes = \
                    self.model.generate_aig_discrete_raw_data(
                        max_nodes=max_size_config,
                        temperature_node=temperature[0],
                        temperature_edge=temperature[1],
                        device=generation_device
                    )
            except Exception as e:
                print(f"Error during self.model.generate_aig_discrete_raw_data (attempt {attempts}): {e}")
                if attempts % (max_attempts // 10 if max_attempts > 10 else 1) == 0:
                    print(f"GraphDF Generation progress: {generated_count}/{num_samples} after {attempts} attempts.")
                continue

            if actual_nodes >= num_min_nodes:
                aig_graph = self._convert_raw_to_aig_digraph(raw_node_features, raw_adj_matrix, actual_nodes,
                                                             aig_node_types)
                if aig_graph is not None:
                    generated_aig_graphs.append(aig_graph)
                    generated_count += 1
                    if generated_count % 10 == 0 or generated_count == num_samples:
                        print(f"GraphDF: Successfully generated {generated_count}/{num_samples} AIGs.")

            if attempts > 0 and attempts % (max_attempts // 20 if max_attempts > 20 else 1) == 0:
                print(f"GraphDF Generation progress: {generated_count}/{num_samples} after {attempts} attempts.")

        if generated_count < num_samples:
            print(
                f"Warning: GraphDF generated only {generated_count} AIGs after {max_attempts} attempts (target was {num_samples}).")

        try:
            output_dir = os.path.dirname(output_pickle_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory for GraphDF output: {output_dir}")

            with open(output_pickle_path, 'wb') as f:
                pickle.dump(generated_aig_graphs, f)
            print(f"Saved {len(generated_aig_graphs)} AIG DiGraphs from GraphDF to {output_pickle_path}")
        except Exception as e:
            print(f"Error saving AIGs from GraphDF to pickle file '{output_pickle_path}': {e}")

        return generated_aig_graphs

