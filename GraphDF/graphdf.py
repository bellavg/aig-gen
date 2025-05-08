import os
import torch
import torch.nn as nn
import networkx as nx  # For DiGraph
import pickle  # For saving generated AIGs
from collections import OrderedDict  # For robust checkpoint loading

# Assuming GraphFlowModel is in .model submodule
# GraphDF uses GraphFlowModel as its internal model.
# Make sure this GraphFlowModel is the one with the updated generate_aig_discrete_raw_data
from .model import GraphFlowModel


# Adjust this import path based on your actual project structure
from . import aig_config


AIG_NODE_TYPE_KEYS = aig_config.NODE_TYPE_KEYS
AIG_EDGE_TYPE_KEYS = aig_config.EDGE_TYPE_KEYS




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

        # Use a consistent task name for model instantiation
        model_task_key = 'rand_gen'  # Assuming GraphFlowModel handles this key
        if task_conceptual_name.startswith('rand_gen'):
            self.model = GraphFlowModel(model_conf_dict)
            print(
                f"GraphFlowModel instantiated for GraphDF task '{task_conceptual_name}'. Target device: {target_device}")
        else:
            raise ValueError('Task {} is not supported in GraphDF!'.format(task_conceptual_name))

        if checkpoint_path is not None:
            try:
                # Load checkpoint onto CPU first to inspect keys, then move model
                state_dict = torch.load(checkpoint_path, map_location='cpu')

                # Check if the checkpoint state_dict needs keys adjusted (e.g., remove 'module.')
                is_data_parallel_checkpoint = any(key.startswith('module.') for key in state_dict)

                if is_data_parallel_checkpoint:
                    print("Checkpoint appears to be from DataParallel model. Adjusting keys...")
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        if k.startswith('module.'):
                            name = k[7:]  # remove `module.`
                            new_state_dict[name] = v
                        else:
                            new_state_dict[k] = v  # Keep keys that don't start with module.
                    state_dict_to_load = new_state_dict
                else:
                    state_dict_to_load = state_dict

                # Load the state dict into the model
                # Note: If the current self.model is wrapped in DataParallel (done in GraphFlowModel.__init__ if multi-GPU),
                # load_state_dict might need adjustment or strict=False. However, GraphFlowModel handles its own DP wrapping.
                self.model.load_state_dict(state_dict_to_load, strict=True)  # Use strict=True first
                print(f"Loaded checkpoint for GraphDF from {checkpoint_path}.")

            except RuntimeError as e:
                # If strict loading fails, try non-strict (e.g., if base_log_probs shapes differ slightly)
                print(f"Strict checkpoint loading failed: {e}. Trying non-strict loading.")
                try:
                    self.model.load_state_dict(state_dict_to_load, strict=False)
                    print(f"Loaded checkpoint for GraphDF from {checkpoint_path} (non-strict).")
                except Exception as e_nonstrict:
                    print(f"Non-strict checkpoint loading also failed: {e_nonstrict}")
                    raise  # Re-raise the exception if non-strict also fails
            except FileNotFoundError:
                print(f"Error: Checkpoint file not found at {checkpoint_path}")
                raise
            except Exception as e:
                print(f"Error loading checkpoint for GraphDF: {e}")
                raise

        # Move the entire model to the target device
        self.model.to(target_device)
        # Verify device after moving
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:  # Handle case where model might have no parameters
            model_device = target_device
        print(f"GraphDF's model ('{task_conceptual_name}') is now on device: {model_device}")

    def load_pretrain_model(self, path, model_conf_dict):
        """ Loads pre-trained weights into the existing self.model for GraphDF. """
        if self.model is None:
            print("Model not initialized in GraphDF. Calling get_model first.")
            # Use a generic task name, assuming it corresponds to GraphFlowModel
            self.get_model('rand_gen', model_conf_dict)

        print(f"Loading pre-trained weights from {path} into GraphDF's model.")
        try:
            # Load to CPU first for inspection
            state_dict = torch.load(path, map_location='cpu')

            # Adjust keys if checkpoint is from DataParallel
            is_data_parallel_checkpoint = any(key.startswith('module.') for key in state_dict)
            if is_data_parallel_checkpoint:
                print("Adjusting pre-trained keys from DataParallel format for GraphDF.")
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        name = k[7:]
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
                state_dict_to_load = new_state_dict
            else:
                state_dict_to_load = state_dict

            # Load weights (non-strict allows loading partial models or different buffer shapes)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict_to_load, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys when loading pretrain weights: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys when loading pretrain weights: {unexpected_keys}")
            print(f"Successfully attempted loading pre-trained weights into GraphDF's model from {path}.")

        except Exception as e:
            print(f"Error in load_pretrain_model from {path}: {e}")
            # Decide whether to raise or just warn
            # raise

        # Ensure model is on the correct final device
        use_gpu_config = model_conf_dict.get('use_gpu', False)
        target_device = torch.device("cuda" if use_gpu_config and torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model.to(target_device)
            try:
                model_device = next(self.model.parameters()).device
            except StopIteration:
                model_device = target_device
            print(f"GraphDF's model with pre-trained weights is now on device: {model_device}")
        else:
            print("Error: Model is None after attempting pretrain load.")

    def train_rand_gen(self, loader, lr, wd, max_epochs, model_conf_dict, save_interval, save_dir):
        r"""
            Running training for random generation task with GraphDF.
            Uses discrete log probability (dis_log_prob).
        """
        # Instantiate or get the model, ensuring it's on the correct device
        self.get_model('rand_gen_train', model_conf_dict)

        # Verify model device
        try:
            current_device = next(self.model.parameters()).device
        except StopIteration:  # Handle model with no parameters
            current_device = torch.device(
                "cuda" if model_conf_dict.get('use_gpu', False) and torch.cuda.is_available() else "cpu")
            print(f"Warning: Model has no parameters. Assuming device {current_device}")

        self.model.train()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)  # Use exist_ok=True
            print(f"Created save directory: {save_dir}")

        print(f"Starting GraphDF training on device: {current_device}")
        for epoch in range(1, max_epochs + 1):
            total_loss = 0
            processed_batches = 0
            for batch_idx, data_batch in enumerate(loader):
                optimizer.zero_grad()
                # Move data to the model's device
                inp_node_features = data_batch.x.to(current_device)  # Expected one-hot
                inp_adj_features = data_batch.adj.to(current_device)  # Expected one-hot (bond_dim channels)

                # GraphDF uses discrete log probability
                # The forward pass of GraphFlowModel is called to get 'out_z'
                out_z = self.model(inp_node_features, inp_adj_features)  # Calls GraphFlowModel.forward()

                # Calculate loss using dis_log_prob
                loss = self.model.dis_log_prob(out_z)  # GraphFlowModel must have dis_log_prob

                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
                # Save the underlying model state if wrapped in DataParallel
                model_state_to_save = self.model.module.state_dict() if isinstance(self.model,
                                                                                   nn.DataParallel) else self.model.state_dict()
                torch.save(model_state_to_save, ckpt_path)
                print(f"Saved GraphDF checkpoint: {ckpt_path}")
        print("GraphDF training finished.")

    def run_rand_gen(self, model_conf_dict, checkpoint_path,
                     num_samples=100, num_min_nodes=5,
                     temperature=[0.6, 0.6],  # GraphDF uses [temp_node, temp_edge]
                     aig_node_type_strings=None,  # Renamed for clarity
                     output_pickle_path="GraphDF_generated_aigs.pkl"):
        r"""
        Running AIG graph generation for random generation task using GraphDF.

        Args:
            model_conf_dict (dict): Python dict for configuring model hyperparameters.
                                    Must include 'node_dim' (e.g., 4), 'bond_dim' (e.g., 3), and 'max_size'.
            checkpoint_path (str): Path to the saved model checkpoint file.
            num_samples (int, optional): Number of AIGs to generate. (default: 100)
            num_min_nodes (int, optional): Minimum number of actual nodes in generated AIGs. (default: 5)
            temperature (list, optional): List of two floats [temp_node, temp_edge] for discrete sampling.
                                          (default: [0.6, 0.6])
            aig_node_type_strings (list, optional): List of AIG node type strings.
                                             Length must match model_conf_dict['node_dim'].
                                             If None, uses defaults from AIG_NODE_TYPE_KEYS.
            output_pickle_path (str, optional): Full path to save the list of generated AIGs.

        :rtype:
            list: A list of generated AIGs, where each AIG is a networkx.DiGraph object.
        """
        node_dim_config = model_conf_dict.get('node_dim')
        max_size_config = model_conf_dict.get('max_size')
        bond_dim_config = model_conf_dict.get('bond_dim')  # Needed by GraphFlowModel

        if node_dim_config is None or max_size_config is None or bond_dim_config is None:
            raise ValueError("model_conf_dict must contain 'node_dim', 'max_size', and 'bond_dim'.")

        if aig_node_type_strings is None:
            aig_node_type_strings = AIG_NODE_TYPE_KEYS  # Use imported or default
            print(f"Using default AIG node type strings: {aig_node_type_strings}")

        # Define AIG edge type strings based on your config (these are for the final NetworkX graph)
        aig_edge_type_strings = AIG_EDGE_TYPE_KEYS  # Should be ['EDGE_REG', 'EDGE_INV']

        if len(aig_node_type_strings) != node_dim_config:
            raise ValueError(
                f"Length of aig_node_type_strings ({len(aig_node_type_strings)}) must match model_conf_dict['node_dim'] ({node_dim_config}).")
        if len(aig_edge_type_strings) != 2:  # Specific check for AIGs
            print(
                f"Warning: Expected 2 AIG edge type strings (REG, INV), got {len(aig_edge_type_strings)}: {aig_edge_type_strings}")

        if not (isinstance(temperature, list) and len(temperature) == 2):
            print(
                f"Warning: GraphDF temperature expected as [temp_node, temp_edge]. Received {temperature}. Using first element or default.")
            if isinstance(temperature, (float, int)):
                temperature = [temperature, temperature]
            elif isinstance(temperature, list) and len(temperature) > 0:
                temperature = [temperature[0], temperature[0]]  # Use first element for both
            else:
                temperature = [0.6, 0.6]  # Default fallback
        print(f"Using temperatures: Node={temperature[0]:.2f}, Edge={temperature[1]:.2f}")

        # Instantiate/load model, ensuring it's on the correct device
        self.get_model('rand_gen_aig', model_conf_dict, checkpoint_path)
        self.model.eval()

        generated_aig_graphs = []
        generated_count = 0
        attempts = 0
        max_attempts = num_samples * 20  # Allow more attempts for discrete generation

        print(f"Attempting to generate {num_samples} AIGs with GraphDF (min nodes: {num_min_nodes})...")

        # Determine device from the loaded model
        try:
            generation_device = next(self.model.parameters()).device
        except StopIteration:  # Model might have no parameters (unlikely for GraphFlow)
            generation_device = torch.device(
                "cuda" if model_conf_dict.get('use_gpu', False) and torch.cuda.is_available() else "cpu")
        print(f"GraphDF generation will run on device: {generation_device}")

        while generated_count < num_samples and attempts < max_attempts:
            attempts += 1
            if attempts % (max_attempts // 20 if max_attempts >= 20 else 1) == 0:
                print(f"Attempt {attempts}/{max_attempts}, Generated {generated_count}/{num_samples}")

            if not hasattr(self.model, 'generate_aig_discrete_raw_data'):
                raise NotImplementedError(
                    "The internal model (GraphFlowModel) does not have the required "
                    "'generate_aig_discrete_raw_data(self, max_nodes, temperature_node, temperature_edge, device)' method for GraphDF's AIG generation. "
                    "This method should return (raw_node_features_one_hot, typed_edges_list, actual_num_nodes)."
                )

            try:
                # Call the new discrete generation method from GraphFlowModel
                # It returns: one-hot node features, list of typed edges, actual node count
                raw_node_features_one_hot, typed_edges_list, actual_nodes = \
                    self.model.generate_aig_discrete_raw_data(
                        max_nodes=max_size_config,
                        temperature_node=temperature[0],
                        temperature_edge=temperature[1],
                        device=generation_device
                    )
            except Exception as e:
                print(f"Error during self.model.generate_aig_discrete_raw_data (attempt {attempts}): {e}")
                # Consider adding a small delay or more specific error handling if needed
                continue  # Skip to the next attempt

            # Check if minimum node count is met
            if actual_nodes >= num_min_nodes:
                # Convert the raw output to a NetworkX DiGraph
                aig_graph = self._convert_raw_to_aig_digraph(raw_node_features_one_hot,
                                                             typed_edges_list,
                                                             actual_nodes,
                                                             aig_node_type_strings,
                                                             aig_edge_type_strings)
                if aig_graph is not None:
                    generated_aig_graphs.append(aig_graph)
                    generated_count += 1
                    if generated_count % 10 == 0 or generated_count == num_samples:
                        print(f"GraphDF: Successfully generated {generated_count}/{num_samples} AIGs.")
            # else: # Optional: log if a graph was generated but too small
            # print(f"Attempt {attempts}: Generated graph with {actual_nodes} nodes (min required: {num_min_nodes}). Skipping.")

        if generated_count < num_samples:
            print(
                f"Warning: GraphDF generated only {generated_count} AIGs after {max_attempts} attempts (target was {num_samples}).")

        # Save the generated graphs
        try:
            output_dir = os.path.dirname(output_pickle_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)  # Use exist_ok=True
                print(f"Created directory for GraphDF output: {output_dir}")

            with open(output_pickle_path, 'wb') as f:
                pickle.dump(generated_aig_graphs, f)
            print(f"Saved {len(generated_aig_graphs)} AIG DiGraphs from GraphDF to {output_pickle_path}")
        except Exception as e:
            print(f"Error saving AIGs from GraphDF to pickle file '{output_pickle_path}': {e}")

        return generated_aig_graphs

    def _convert_raw_to_aig_digraph(self, node_features_one_hot, typed_edges_list,
                                    num_actual_nodes, aig_node_type_strings, aig_edge_type_strings):
        """
        Converts raw discrete model output (one-hot node features and typed edge list)
        to a NetworkX DiGraph for AIGs.

        Args:
            node_features_one_hot (torch.Tensor): Tensor of shape (max_nodes, node_dim), one-hot encoded.
            typed_edges_list (list): List of tuples (source_idx, target_idx, edge_type_idx),
                                     where edge_type_idx is 0 for 'EDGE_REG', 1 for 'EDGE_INV'.
            num_actual_nodes (int): The actual number of nodes in the graph (up to max_nodes).
            aig_node_type_strings (list): List of AIG node type strings (e.g., ['NODE_CONST0', ...]).
            aig_edge_type_strings (list): List of AIG edge type strings (e.g., ['EDGE_REG', 'EDGE_INV']).

        Returns:
            nx.DiGraph: A NetworkX directed graph representing the AIG, or None if conversion fails.
        """
        graph = nx.DiGraph()
        node_features = node_features_one_hot.cpu().detach()  # Ensure on CPU

        # Add nodes up to num_actual_nodes
        for i in range(num_actual_nodes):
            try:
                # Find the index of the '1' in the one-hot vector
                if node_features[i].sum() == 0:  # Handle case of all-zero vector (shouldn't happen with sampling)
                    print(f"Warning: Node {i} has all-zero feature vector. Setting type to 'UNKNOWN_NODE_TYPE'.")
                    node_type_label = "UNKNOWN_NODE_TYPE"
                else:
                    node_type_idx = torch.argmax(node_features[i]).item()
                    if 0 <= node_type_idx < len(aig_node_type_strings):
                        node_type_label = aig_node_type_strings[node_type_idx]
                    else:
                        print(
                            f"Warning: Node {i} has invalid type index {node_type_idx} from argmax (max allowed: {len(aig_node_type_strings) - 1}). Setting type to 'UNKNOWN_NODE_TYPE'.")
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
                # Check if the edge_type_idx is valid for the provided AIG edge types (REG/INV)
                if 0 <= edge_type_idx < len(aig_edge_type_strings):
                    edge_type_label = aig_edge_type_strings[edge_type_idx]
                    graph.add_edge(u, v, type=edge_type_label)
                else:
                    # This case should not happen if generate_aig_discrete_raw_data only returns indices 0 or 1
                    print(
                        f"Warning: Edge ({u}->{v}) has unexpected edge_type_idx {edge_type_idx} (expected 0 or 1). Adding edge without 'type' attribute.")
                    graph.add_edge(u, v)  # Add edge without type if index is bad
            else:
                print(
                    f"Warning: Skipping edge ({u}->{v}) because node indices are out of range for actual_num_nodes={num_actual_nodes}.")

        return graph

    # --- Other methods (e.g., for property optimization) would go here if implemented for GraphDF ---
    # ...
