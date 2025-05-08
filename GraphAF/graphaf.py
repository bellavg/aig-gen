import os
import torch
import torch.nn as nn
# from rdkit import Chem # Not needed for AIGs
from .model import GraphFlowModel  # Assuming .model points to the directory containing graphflow.py
from .train_utils import adjust_learning_rate, DataIterator  # Assuming train_utils is in the same directory
import pickle
import networkx as nx
import warnings  # For warnings

# --- AIG Configuration Import ---
# Attempt to import AIG configuration.
# Adjust the path '../data/aig_config' if your aig_config.py is located elsewhere relative to this file.
# For example, if aig_config.py is in the same directory as this graphaf.py, use:
# from . import aig_config as aig_config_module
# Or if it's in a G2PT.configs package:
# from G2PT.configs import aig as aig_config_module

AIG_NODE_TYPE_KEYS = None
AIG_EDGE_TYPE_KEYS = None

try:
    # Option 1: If aig_config.py is in a sibling 'data' directory (e.g., GraphAF/data/aig_config.py)
    from ..data import aig_config as aig_config_module  # If graphaf.py is in GraphAF/

    AIG_NODE_TYPE_KEYS = aig_config_module.NODE_TYPE_KEYS
    AIG_EDGE_TYPE_KEYS = aig_config_module.EDGE_TYPE_KEYS
    print("Successfully imported AIG config from ..data.aig_config")
except ImportError:
    try:
        # Option 2: If aig_config.py is in the same directory as this file (GraphAF/aig_config.py)
        from . import aig_config as aig_config_module_local

        AIG_NODE_TYPE_KEYS = aig_config_module_local.NODE_TYPE_KEYS
        AIG_EDGE_TYPE_KEYS = aig_config_module_local.EDGE_TYPE_KEYS
        print("Successfully imported AIG config from local .aig_config")
    except ImportError:
        warnings.warn(
            "Could not import AIG configuration (tried ..data.aig_config and .aig_config). "
            "Using default AIG type keys. Please ensure 'aig_config.py' is accessible."
        )

# Fallback default AIG type keys if import fails
if AIG_NODE_TYPE_KEYS is None:
    AIG_NODE_TYPE_KEYS = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO']
if AIG_EDGE_TYPE_KEYS is None:
    AIG_EDGE_TYPE_KEYS = ['EDGE_REG', 'EDGE_INV']

print(f"Using AIG_NODE_TYPE_KEYS: {AIG_NODE_TYPE_KEYS}")
print(f"Using AIG_EDGE_TYPE_KEYS: {AIG_EDGE_TYPE_KEYS}")


# --- End AIG Configuration Import ---


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
        The method class for GraphAF algorithm proposed in the paper `GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation <https://arxiv.org/abs/2001.09382>`_.
        This class provides interfaces for running random generation, property optimization, and constrained optimization with GraphAF.
        This version is adapted for And-Inverter Graph (AIG) generation.
    """

    def __init__(self):
        super(GraphAF, self).__init__()
        self.model = None  # Will be an instance of GraphFlowModel

    def get_model(self, task_conceptual_name, model_conf_dict, checkpoint_path=None):
        """
        Initializes or loads the GraphFlowModel for GraphAF.

        Args:
            task_conceptual_name (str): Conceptual name of the task, e.g., 'rand_gen' or 'rand_gen_aig'.
            model_conf_dict (dict): Configuration dictionary for the model.
                                    Must include 'node_dim', 'bond_dim', 'max_size'.
            checkpoint_path (str, optional): Path to a pre-trained checkpoint.
        """
        # Validate essential model configurations for AIGs
        if not all(k in model_conf_dict for k in ['node_dim', 'bond_dim', 'max_size']):
            raise ValueError("model_conf_dict must contain 'node_dim', 'bond_dim', and 'max_size'.")

        use_gpu_config = model_conf_dict.get('use_gpu', False)
        if use_gpu_config and not torch.cuda.is_available():
            warnings.warn("CUDA requested in config but not available. Using CPU.")
            model_conf_dict['use_gpu'] = False
            target_device = torch.device("cpu")
        elif use_gpu_config:
            target_device = torch.device("cuda")
        else:
            model_conf_dict['use_gpu'] = False  # Ensure it's explicitly set
            target_device = torch.device("cpu")

        # Instantiate the model (GraphFlowModel handles its own internal MaskedGraphAF)
        if task_conceptual_name.startswith('rand_gen'):  # Covers 'rand_gen', 'rand_gen_aig' etc.
            self.model = GraphFlowModel(model_conf_dict)
            print(
                f"GraphFlowModel instantiated for GraphAF task '{task_conceptual_name}'. Target device: {target_device}")
        else:
            # If you have different model classes for different tasks, handle them here
            raise ValueError('Task {} is not supported in GraphAF or model instantiation is not defined for it.'.format(
                task_conceptual_name))

        if checkpoint_path is not None:
            try:
                # Load checkpoint onto CPU first to inspect keys, then move model
                state_dict = torch.load(checkpoint_path, map_location='cpu')

                # Check if the checkpoint state_dict needs keys adjusted (e.g., remove 'module.')
                is_data_parallel_checkpoint = any(key.startswith('module.') for key in state_dict)

                current_model_is_dp = isinstance(self.model, nn.DataParallel)
                # If model is DP but state_dict is not, DP wrapper handles it.
                # If model is not DP but state_dict is, strip 'module.'
                if not current_model_is_dp and is_data_parallel_checkpoint:
                    print("Checkpoint is from DataParallel model, current model is not. Adjusting keys...")
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        if k.startswith('module.'):
                            name = k[7:]  # remove `module.`
                            new_state_dict[name] = v
                        else:
                            new_state_dict[k] = v  # Keep keys that don't start with module.
                    state_dict_to_load = new_state_dict
                elif current_model_is_dp and not is_data_parallel_checkpoint:
                    # This can happen if you train on single GPU then try to load on multi-GPU with DP wrapper.
                    # PyTorch's DataParallel usually expects keys to NOT have 'module.' in this case.
                    # Or, if the model's state_dict keys are already prefixed by its own DP wrapper.
                    # For simplicity, assume direct load works or model's DP wrapper handles it.
                    print("Warning: Current model is DataParallel, checkpoint is not. Attempting direct load.")
                    state_dict_to_load = state_dict
                else:  # Both DP or both not DP (or model handles it)
                    state_dict_to_load = state_dict

                self.model.load_state_dict(state_dict_to_load, strict=True)
                print(f"Loaded checkpoint for GraphAF from {checkpoint_path}.")

            except RuntimeError as e:
                warnings.warn(f"Strict checkpoint loading failed: {e}. Trying non-strict loading.")
                try:
                    self.model.load_state_dict(state_dict_to_load, strict=False)
                    print(f"Loaded checkpoint for GraphAF from {checkpoint_path} (non-strict).")
                except Exception as e_nonstrict:
                    print(f"Non-strict checkpoint loading also failed: {e_nonstrict}")
                    raise
            except FileNotFoundError:
                print(f"Error: Checkpoint file not found at {checkpoint_path}")
                raise
            except Exception as e:
                print(f"Error loading checkpoint for GraphAF: {e}")
                raise

        # Move the entire model to the target device
        self.model.to(target_device)
        # Verify device after moving
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:  # Handle case where model might have no parameters (e.g. freshly init, no layers)
            model_device = target_device  # Assume it's on target_device
        print(f"GraphAF's model ('{task_conceptual_name}') is now on device: {model_device}")

    def load_pretrain_model(self, path, model_conf_dict):
        """ Loads pre-trained weights into the existing self.model for GraphAF. """
        if self.model is None:
            print("Model not initialized in GraphAF. Calling get_model first.")
            # Use a generic task name, assuming it corresponds to GraphFlowModel
            self.get_model('rand_gen', model_conf_dict)  # model_conf_dict needed here

        print(f"Loading pre-trained weights from {path} into GraphAF's model.")
        try:
            # Load to CPU first for inspection
            state_dict = torch.load(path, map_location='cpu')

            from collections import OrderedDict
            is_data_parallel_checkpoint = any(key.startswith('module.') for key in state_dict)

            if is_data_parallel_checkpoint and not isinstance(self.model, nn.DataParallel):
                print("Adjusting pre-trained keys from DataParallel format for non-DP GraphAF model.")
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        name = k[7:]
                    else:
                        name = k
                    new_state_dict[name] = v
                state_dict_to_load = new_state_dict
            elif not is_data_parallel_checkpoint and isinstance(self.model, nn.DataParallel):
                print("Pre-trained keys are not DataParallel, but GraphAF model is. Wrapping keys with 'module.'.")
                # This case is less common for loading into a DP model; usually, the DP model itself handles it.
                # However, if strict loading fails, this might be a reason. For now, assume direct load works or DP handles it.
                # Or, one might need to load into self.model.module.
                state_dict_to_load = state_dict  # Try direct load first
            else:
                state_dict_to_load = state_dict

            # Load weights (non-strict allows loading partial models or different buffer shapes)
            # If self.model is DP, load into self.model.module
            target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            missing_keys, unexpected_keys = target_model.load_state_dict(state_dict_to_load, strict=False)

            if missing_keys:
                warnings.warn(f"Warning: Missing keys when loading pretrain weights: {missing_keys}")
            if unexpected_keys:
                warnings.warn(f"Warning: Unexpected keys when loading pretrain weights: {unexpected_keys}")
            print(f"Successfully attempted loading pre-trained weights into GraphAF's model from {path}.")

        except Exception as e:
            print(f"Error in load_pretrain_model from {path}: {e}")
            # raise # Decide whether to raise or just warn

        # Ensure model is on the correct final device (already handled by get_model, but good for safety)
        use_gpu_config = model_conf_dict.get('use_gpu', False)
        final_device = torch.device("cuda" if use_gpu_config and torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model.to(final_device)
            try:
                model_device = next(self.model.parameters()).device
            except StopIteration:
                model_device = final_device
            print(f"GraphAF's model with pre-trained weights is now on device: {model_device}")
        else:
            print("Error: Model is None after attempting pretrain load.")

    def train_rand_gen(self, loader, lr, wd, max_epochs, model_conf_dict, save_interval, save_dir):
        r"""
            Running training for random generation task (e.g., AIGs).

            Args:
                loader: Data loader for training samples (e.g., AIGs).
                lr (float): Learning rate.
                wd (float): Weight decay.
                max_epochs (int): Maximum training epochs.
                model_conf_dict (dict): Configuration for model hyperparameters.
                                        Must include 'node_dim', 'bond_dim', 'max_size'.
                save_interval (int): Frequency to save model checkpoints.
                save_dir (str): Directory to save checkpoints.
        """
        # Instantiate or get the model, ensuring it's on the correct device
        self.get_model('rand_gen_train', model_conf_dict)  # Task name can be specific

        # Determine device from the model itself (it's already moved in get_model)
        try:
            current_device = next(self.model.parameters()).device
        except StopIteration:  # Model might have no parameters
            current_device = torch.device(
                "cuda" if model_conf_dict.get('use_gpu', False) and torch.cuda.is_available() else "cpu")
            warnings.warn(f"Model has no parameters. Assuming device {current_device}")

        self.model.train()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created save directory: {save_dir}")

        print(f"Starting GraphAF training on device: {current_device}")
        for epoch in range(1, max_epochs + 1):
            total_loss = 0
            processed_batches = 0
            for batch_idx, data_batch in enumerate(loader):
                optimizer.zero_grad()
                # Move data to the model's device
                # For AIGs, data_batch.x should be one-hot node features
                # data_batch.adj should be one-hot adjacency features (bond_dim channels)
                inp_node_features = data_batch.x.to(current_device)
                inp_adj_features = data_batch.adj.to(current_device)

                # GraphAF uses continuous flow, so forward pass returns latent z and logdet
                out_z, out_logdet = self.model(inp_node_features, inp_adj_features)
                loss = self.model.log_prob(out_z, out_logdet)  # log_prob is for continuous flows

                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                processed_batches += 1
                if batch_idx % 200 == 0:  # Log more frequently
                    print(
                        'Epoch {}/{} | Batch {}/{} | GraphAF Training Loss: {:.4f}'.format(epoch, max_epochs, batch_idx,
                                                                                           len(loader), loss.item()))

            avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
            print("Epoch {}/{} | GraphAF Average Training Loss: {:.4f}".format(epoch, max_epochs, avg_loss))

            if epoch % save_interval == 0:
                ckpt_path = os.path.join(save_dir, 'graphaf_rand_gen_ckpt_epoch_{}.pth'.format(epoch))
                model_state_to_save = self.model.module.state_dict() if isinstance(self.model,
                                                                                   nn.DataParallel) else self.model.state_dict()
                torch.save(model_state_to_save, ckpt_path)
                print(f"Saved GraphAF checkpoint: {ckpt_path}")
        print("GraphAF training finished.")

    def run_rand_gen(self, model_conf_dict, checkpoint_path,
                     num_samples=100, num_min_nodes=5,
                     temperature=0.75,  # Single temperature for GraphAF's prior
                     aig_node_type_strings_override=None,  # Allow overriding default/config node types
                     output_pickle_path="GraphAF_generated_aigs.pkl"):
        r"""
        Running AIG graph generation for random generation task using GraphAF.

        Args:
            model_conf_dict (dict): Python dict for configuring model hyperparameters.
                                    Must include 'node_dim', 'bond_dim', and 'max_size'.
            checkpoint_path (str): Path to the saved model checkpoint file.
            num_samples (int, optional): Number of AIGs to generate.
            num_min_nodes (int, optional): Minimum number of actual nodes in generated AIGs.
            temperature (float, optional): Temperature for sampling from the prior distribution.
            aig_node_type_strings_override (list, optional): Override default/config AIG node type strings.
                                                             Length must match model_conf_dict['node_dim'].
            output_pickle_path (str, optional): Full path to save the list of generated AIGs.

        :rtype:
            list: A list of generated AIGs, where each AIG is a networkx.DiGraph object.
        """
        node_dim_config = model_conf_dict.get('node_dim')
        max_size_config = model_conf_dict.get('max_size')
        bond_dim_config = model_conf_dict.get('bond_dim')

        if not all([node_dim_config, max_size_config, bond_dim_config]):
            raise ValueError("model_conf_dict must contain 'node_dim', 'max_size', and 'bond_dim'.")

        current_aig_node_types = aig_node_type_strings_override if aig_node_type_strings_override else AIG_NODE_TYPE_KEYS
        current_aig_edge_types = AIG_EDGE_TYPE_KEYS  # Typically fixed for AIGs (REG, INV)

        if len(current_aig_node_types) != node_dim_config:
            raise ValueError(
                f"Length of AIG node type strings ({len(current_aig_node_types)}) must match model_conf_dict['node_dim'] ({node_dim_config}).")
        if len(current_aig_edge_types) != 2:  # Specific check for AIGs: REG, INV
            warnings.warn(
                f"Expected 2 AIG edge type strings (REG, INV), got {len(current_aig_edge_types)}: {current_aig_edge_types}")

        # Instantiate/load model, ensuring it's on the correct device
        self.get_model('rand_gen_aig', model_conf_dict, checkpoint_path)
        self.model.eval()

        generated_aig_graphs = []
        generated_count = 0
        attempts = 0
        # Allow more attempts, as continuous generation might produce small graphs more often
        max_attempts = num_samples * 30 if num_samples > 10 else num_samples * 50

        print(
            f"Attempting to generate {num_samples} AIGs with GraphAF (min nodes: {num_min_nodes}, max_size: {max_size_config}, temp: {temperature})...")

        try:
            generation_device = next(self.model.parameters()).device
        except StopIteration:
            generation_device = torch.device(
                "cuda" if model_conf_dict.get('use_gpu', False) and torch.cuda.is_available() else "cpu")
        print(f"GraphAF generation will run on device: {generation_device}")

        while generated_count < num_samples and attempts < max_attempts:
            attempts += 1
            if attempts % (max(1, max_attempts // 20)) == 0:  # Log progress
                print(f"Attempt {attempts}/{max_attempts}, Generated {generated_count}/{num_samples}")

            if not hasattr(self.model, 'generate_aig_raw_data'):
                raise NotImplementedError(
                    "The internal model (GraphFlowModel) does not have the required "
                    "'generate_aig_raw_data(self, max_nodes, temperature, device)' method. "
                    "This method should return (raw_node_features_tensor, typed_edges_list, actual_num_nodes)."
                )

            try:
                # This method in GraphFlowModel is responsible for the actual autoregressive generation loop,
                # calling MaskedGraphAF.reverse(), and applying AIG-specific constraints/logic.
                raw_node_features, typed_edges_list, actual_nodes = \
                    self.model.generate_aig_raw_data(
                        max_nodes=max_size_config,
                        temperature=temperature,  # Single temperature for GraphAF
                        device=generation_device
                    )
            except Exception as e:
                print(f"Error during self.model.generate_aig_raw_data (attempt {attempts}): {e}")
                import traceback
                traceback.print_exc()
                continue

            if actual_nodes >= num_min_nodes:
                aig_graph = self._convert_raw_to_aig_digraph(raw_node_features,
                                                             typed_edges_list,
                                                             actual_nodes,
                                                             current_aig_node_types,
                                                             current_aig_edge_types)
                if aig_graph is not None:
                    # Optional: Add further validation specific to AIGs here if needed
                    # e.g., check for graph connectivity, specific structural properties.
                    generated_aig_graphs.append(aig_graph)
                    generated_count += 1
                    if generated_count % 10 == 0 or generated_count == num_samples:
                        print(f"GraphAF: Successfully generated {generated_count}/{num_samples} AIGs.")
            # else:
            #     print(f"Attempt {attempts}: Generated graph with {actual_nodes} nodes (min required: {num_min_nodes}). Skipping.")

        if generated_count < num_samples:
            warnings.warn(
                f"GraphAF generated only {generated_count} AIGs after {max_attempts} attempts (target was {num_samples}). "
                "Consider adjusting temperature, num_min_nodes, or increasing max_attempts."
            )

        try:
            output_dir = os.path.dirname(output_pickle_path)
            if output_dir and not os.path.exists(output_dir):  # Ensure output_dir is not empty string
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created directory for GraphAF output: {output_dir}")

            with open(output_pickle_path, 'wb') as f:
                pickle.dump(generated_aig_graphs, f)
            print(f"Saved {len(generated_aig_graphs)} AIG DiGraphs from GraphAF to {output_pickle_path}")
        except Exception as e:
            print(f"Error saving AIGs from GraphAF to pickle file '{output_pickle_path}': {e}")

        return generated_aig_graphs

    def _convert_raw_to_aig_digraph(self, node_features_tensor, typed_edges_list,
                                    num_actual_nodes, aig_node_type_strings, aig_edge_type_strings):
        """
        Converts raw model output (node features scores and a list of typed edges)
        to a NetworkX DiGraph for AIGs.

        Args:
            node_features_tensor (torch.Tensor): Tensor of shape (max_nodes, node_dim)
                                                 representing node type scores/probabilities.
            typed_edges_list (list): List of tuples (source_idx, target_idx, aig_edge_type_idx),
                                     where aig_edge_type_idx is 0 for 'EDGE_REG', 1 for 'EDGE_INV'.
            num_actual_nodes (int): The actual number of nodes in the graph.
            aig_node_type_strings (list): List of AIG node type strings (e.g., ['NODE_CONST0', ...]).
            aig_edge_type_strings (list): List of AIG edge type strings (e.g., ['EDGE_REG', 'EDGE_INV']).

        Returns:
            nx.DiGraph: A NetworkX directed graph representing the AIG, or None if conversion fails.
        """
        graph = nx.DiGraph()
        # Ensure node_features_tensor is on CPU for numpy/python operations if it's not already
        node_features = node_features_tensor.cpu().detach()

        # Add nodes up to num_actual_nodes
        for i in range(num_actual_nodes):
            try:
                # Determine node type by taking argmax of the feature vector for node i
                # These features are continuous scores from the flow model's reverse pass.
                if node_features[i].sum() == 0 and torch.all(node_features[i] == 0):
                    # This might happen if a node was intended to be padding or generation stopped early.
                    warnings.warn(f"Node {i} has all-zero feature vector. Setting type to 'UNKNOWN_NODE_TYPE'.")
                    node_type_label = "UNKNOWN_NODE_TYPE"
                else:
                    node_type_idx = torch.argmax(node_features[i]).item()
                    if 0 <= node_type_idx < len(aig_node_type_strings):
                        node_type_label = aig_node_type_strings[node_type_idx]
                    else:
                        warnings.warn(
                            f"Node {i} has invalid type index {node_type_idx} from argmax "
                            f"(max allowed: {len(aig_node_type_strings) - 1}). Setting type to 'UNKNOWN_NODE_TYPE'."
                        )
                        node_type_label = "UNKNOWN_NODE_TYPE"
                graph.add_node(i, type=node_type_label)
            except IndexError:
                print(f"Error: Index out of bounds when accessing node_features for node {i}. "
                      f"num_actual_nodes={num_actual_nodes}, node_features.shape={node_features.shape}. Skipping graph.")
                return None  # Critical error, cannot proceed with this graph
            except Exception as e:
                print(f"Error processing node {i}: {e}. Skipping graph.")
                return None  # Critical error

        # Add typed directed edges from the typed_edges_list
        # typed_edges_list is assumed to be [(source, target, aig_edge_type_idx)]
        # where aig_edge_type_idx is 0 for REG, 1 for INV.
        for u, v, aig_edge_type_idx in typed_edges_list:
            if not (0 <= u < num_actual_nodes and 0 <= v < num_actual_nodes):
                warnings.warn(f"Skipping edge ({u}->{v}) because node indices are out of range "
                              f"for actual_num_nodes={num_actual_nodes}.")
                continue

            if 0 <= aig_edge_type_idx < len(aig_edge_type_strings):
                edge_type_label = aig_edge_type_strings[aig_edge_type_idx]
                graph.add_edge(u, v, type=edge_type_label)
            else:
                # This should ideally be prevented by the generation logic in GraphFlowModel
                warnings.warn(
                    f"Edge ({u}->{v}) has unexpected aig_edge_type_idx {aig_edge_type_idx} "
                    f"(expected 0 for '{AIG_EDGE_TYPE_KEYS[0]}' or 1 for '{AIG_EDGE_TYPE_KEYS[1]}'). "
                    "Adding edge without 'type' attribute or skipping."
                )
                graph.add_edge(u, v)  # Or decide to skip: continue

        return graph

    # --- Methods for property optimization (train_prop_opt, run_prop_opt, etc.) ---
    # These methods are less directly impacted by AIG-specific changes at this GraphAF wrapper level,
    # as the core generation/modification logic is within self.model (GraphFlowModel).
    # If AIGs have specific properties to optimize, the reward functions and
    # generation/modification strategies within GraphFlowModel would need to be adapted.
    # For now, the structure of these methods can remain similar.
    # ... (original property optimization methods would be here) ...
    # def train_prop_optim(self, ...): ...
    # def run_prop_optim(self, ...): ...
    # def train_cons_optim(self, ...): ...
    # def run_cons_optim_one_mol(self, ...): ...
    # def run_cons_optim(self, ...): ...

