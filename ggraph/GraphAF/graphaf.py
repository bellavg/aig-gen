import os
import torch
import warnings  # Added for wandb import warning

from generator import Generator
from .model import GraphFlowModel  # Assuming .model imports GraphFlowModel
from aig_config import display_graph_details
# wandb import
try:
    import wandb
except ImportError:
    wandb = None
    # This warning will be more relevant if wandb_active is True during training
    # warnings.warn("wandb not installed. Wandb logging will be disabled if attempted. Run 'pip install wandb'")


class GraphAF(Generator):
    r"""
        The method class for GraphAF algorithm proposed in the paper `GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation <https://arxiv.org/abs/2001.09382>`_. This class provides interfaces for running random generation, property
        optimization, and constrained optimization with GraphAF. Please refer to the `example codes <https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph/GraphAF>`_ for usage examples.
    """

    def __init__(self):
        super(GraphAF, self).__init__()
        self.model = None

    def get_model(self, task, model_conf_dict, checkpoint_path=None):
        # Ensure 'use_gpu' key exists, default to False if not
        use_gpu_config = model_conf_dict.get('use_gpu', False)
        if use_gpu_config and not torch.cuda.is_available():
            warnings.warn("CUDA is not available, 'use_gpu' in model_conf_dict is set to False.")
            model_conf_dict['use_gpu'] = False  # Modify a copy or ensure this is intended

        if task == 'rand_gen':
            self.model = GraphFlowModel(model_conf_dict)
        else:
            # Original code had 'GraphDF', assuming it should be 'GraphAF' here
            raise ValueError('Task {} is not supported in GraphAF!'.format(task))

        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                warnings.warn(
                    f"Checkpoint path {checkpoint_path} does not exist. Model will be initialized from scratch.")
            else:
                try:
                    # Determine device for loading checkpoint
                    device_to_load_on = torch.device(
                        'cuda' if model_conf_dict.get('use_gpu') and torch.cuda.is_available() else 'cpu')
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=device_to_load_on))
                    print(f"Loaded model checkpoint from {checkpoint_path} onto {device_to_load_on}")
                except Exception as e:
                    warnings.warn(
                        f"Could not load checkpoint from {checkpoint_path}: {e}. Model will be initialized from scratch.")

    def load_pretrain_model(self, path):
        # Determine device based on current model's configuration if available
        device = torch.device('cuda' if self.model and self.model.dp and torch.cuda.is_available() else 'cpu')
        try:
            load_key = torch.load(path, map_location=device)
            is_data_parallel = isinstance(self.model, torch.nn.DataParallel)

            model_state_dict = self.model.module.state_dict() if is_data_parallel else self.model.state_dict()

            new_state_dict = {}
            for k, v in load_key.items():
                name = k[7:] if k.startswith('module.') and not is_data_parallel else k
                if name in model_state_dict:
                    new_state_dict[name] = v
                elif k in model_state_dict:
                    new_state_dict[k] = v

            if not new_state_dict:
                warnings.warn(f"No matching keys found between checkpoint {path} and model state_dict.")
                return

            if is_data_parallel:
                self.model.module.load_state_dict(new_state_dict, strict=False)
            else:
                self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Successfully loaded pre-trained model from {path}")

        except Exception as e:
            warnings.warn(f"Error loading pre-trained model from {path}: {e}")

    def train_rand_gen(self, loader, lr, wd, max_epochs, model_conf_dict,
                       save_interval, save_dir, wandb_active=True):  # Added wandb_active
        r"""
            Running training for random generation task.
            ... (rest of docstring)
            Args:
                ...
                wandb_active (bool, optional): Flag to enable/disable wandb logging.
        """

        self.get_model('rand_gen', model_conf_dict)
        self.model.train()

        model_device = next(self.model.parameters()).device

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if wandb_active and wandb is None:
            warnings.warn("wandb_active is True, but wandb library is not installed. Logging will be skipped.")
            wandb_active = False

        print(f"Starting GraphAF training for {max_epochs} epochs on device {model_device}...")
        for epoch in range(1, max_epochs + 1):
            total_loss = 0
            num_batches = 0
            for batch_idx, data_batch in enumerate(loader):
                optimizer.zero_grad()

                inp_node_features = data_batch.x.to(model_device)
                inp_adj_features = data_batch.adj.to(model_device)

                # Forward pass for GraphAF
                out_z, out_logdet = self.model(inp_node_features, inp_adj_features)
                loss = self.model.log_prob(out_z, out_logdet)  # GraphAF specific loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if batch_idx % 100 == 0:
                    print('Epoch {}/{} | Training iteration {}/{} | loss {:.4f}'.format(epoch, max_epochs, batch_idx,
                                                                                        len(loader), loss.item()))

            avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
            print("Epoch {}/{} | Average training loss: {:.4f}".format(epoch, max_epochs, avg_loss))

            # --- Wandb Logging (Per Epoch) ---
            if wandb_active and wandb.run:
                try:
                    wandb.log({"epoch": epoch, "avg_loss": avg_loss, "learning_rate": lr})
                except Exception as e:
                    warnings.warn(f"Failed to log to wandb for epoch {epoch}: {e}")
            # --- End Wandb Logging ---

            if epoch % save_interval == 0:
                # Original GraphAF saved with 'rand_gen_ckpt_', GraphDF with 'GraphDF_ckpt_'
                # Using a consistent naming for GraphAF here.
                ckpt_path = os.path.join(save_dir, 'GraphAF_rand_gen_ckpt_{}.pth'.format(epoch))
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        print("GraphAF training finished.")

    def run_rand_gen(self, model_conf_dict, checkpoint_path, n_mols=1000, num_min_node=5, num_max_node=64,
                     temperature=0.75):
        r"""
            Running graph generation for random generation task.
            ... (rest of docstring) ...
        """

        self.get_model('rand_gen', model_conf_dict, checkpoint_path)
        self.model.eval()

        model_device = next(self.model.parameters()).device
        print(f"GraphAF generation will run on device: {model_device}")

        all_mols, pure_valids = [], []  # In AIG context, 'mols' will be NetworkX graphs
        cnt_generated = 0

        print(f"Attempting to generate {n_mols} AIGs with GraphAF...")
        with torch.no_grad():
            while cnt_generated < n_mols:
                try:
                    # The generate method in GraphAF's GraphFlowModel needs to be adapted for AIGs
                    # and handle temperature correctly.
                    # It also needs to return an AIG (e.g., NetworkX graph)
                    aig_graph, no_resample_flag, num_actual_nodes = self.model.generate(
                        min_atoms=num_min_node,  # min_atoms in generate corresponds to num_min_node
                        max_atoms=num_max_node,  # max_atoms in generate corresponds to num_max_node
                        temperature=temperature
                    )

                    if aig_graph is not None and (num_actual_nodes >= num_min_node):
                        cnt_generated += 1
                        all_mols.append(aig_graph)
                        pure_valids.append(no_resample_flag)
                        if cnt_generated == 1:
                            display_graph_details(aig_graph)
                        if cnt_generated % 100 == 0:  # Changed from 10 to 100 for less verbose output
                            print('Generated {}/{} AIGs'.format(cnt_generated, n_mols))
                    elif aig_graph is None:
                        warnings.warn("GraphAF generation returned None for an AIG. Skipping.")
                except Exception as e:
                    warnings.warn(f"Error during GraphAF generation of an AIG: {e}. Skipping this attempt.")
                    # Add a counter or break if too many errors occur to prevent infinite loops.

        if cnt_generated < n_mols:
            warnings.warn(
                f"Desired {n_mols} AIGs, but only {cnt_generated} were successfully generated and met criteria by GraphAF.")

        return all_mols, pure_valids
