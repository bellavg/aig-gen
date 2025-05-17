import os
import torch
import warnings  # Added for wandb import warning

from generator import Generator
from .model import GraphFlowModel  # Assuming .model imports GraphFlowModel

# wandb import
try:
    import wandb
except ImportError:
    wandb = None
    # This warning will be more relevant if wandb_active is True during training
    # warnings.warn("wandb not installed. Wandb logging will be disabled if attempted. Run 'pip install wandb'")


class GraphDF(Generator):
    r"""
        The method class for GraphDF algorithm proposed in the paper `GraphDF: A Discrete Flow Model for Molecular Graph Generation <https://arxiv.org/abs/2102.01189>`_. This class provides interfaces for running random generation, property
        optimization, and constrained optimization with GraphDF algorithm. Please refer to the `example codes <https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph/GraphDF>`_ for usage examples.
    """

    def __init__(self):
        super(GraphDF, self).__init__()
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
            raise ValueError('Task {} is not supported in GraphDF!'.format(task))

        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                warnings.warn(
                    f"Checkpoint path {checkpoint_path} does not exist. Model will be initialized from scratch.")
            else:
                try:
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(
                        'cuda' if model_conf_dict.get('use_gpu') else 'cpu')))
                    print(f"Loaded model checkpoint from {checkpoint_path}")
                except Exception as e:
                    warnings.warn(
                        f"Could not load checkpoint from {checkpoint_path}: {e}. Model will be initialized from scratch.")

    def load_pretrain_model(self, path):
        # Determine device based on current model's configuration if available
        device = torch.device('cuda' if self.model and self.model.dp and torch.cuda.is_available() else 'cpu')
        try:
            load_key = torch.load(path, map_location=device)
            # If model is DataParallel, keys might be prefixed with 'module.'
            is_data_parallel = isinstance(self.model, torch.nn.DataParallel)

            model_state_dict = self.model.module.state_dict() if is_data_parallel else self.model.state_dict()

            # Create a new state_dict for loading, stripping 'module.' if necessary from load_key
            new_state_dict = {}
            for k, v in load_key.items():
                name = k[7:] if k.startswith('module.') and not is_data_parallel else k  # remove `module.`
                if name in model_state_dict:
                    new_state_dict[name] = v
                elif k in model_state_dict:  # If keys in checkpoint already match non-DP model
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
                wandb_active (bool, optional): Flag to enable/disable wandb logging during this training run.
        """

        self.get_model('rand_gen', model_conf_dict)
        self.model.train()

        # Determine device for optimizer based on model's device
        # Assuming self.model is on the correct device after get_model
        model_device = next(self.model.parameters()).device

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)  # Added exist_ok=True

        # Check wandb availability if wandb_active is True
        if wandb_active and wandb is None:
            warnings.warn(
                "wandb_active is True, but wandb library is not installed. Logging will be skipped for this training.")
            wandb_active = False  # Disable if not installed

        print(f"Starting training for {max_epochs} epochs...")
        for epoch in range(1, max_epochs + 1):
            total_loss = 0
            num_batches = 0
            for batch_idx, data_batch in enumerate(loader):
                optimizer.zero_grad()

                # Move data to the same device as the model
                inp_node_features = data_batch.x.to(model_device) # Size B, N, Node_dim = 5

                inp_adj_features = data_batch.adj.to(model_device) #Size, B, edge_dim, N, N edg_dim = 3

                # Forward pass
                out_z = self.model(inp_node_features, inp_adj_features)
                loss = self.model.dis_log_prob(out_z)  # GraphDF specific loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()  # Use .item() to get Python number
                num_batches += 1

                if batch_idx % 100 == 0:
                    print('Epoch {}/{} | Training iteration {}/{} | loss {:.4f}'.format(epoch, max_epochs, batch_idx,
                                                                                        len(loader), loss.item()))

            avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
            print("Epoch {}/{} | Average training loss: {:.4f}".format(epoch, max_epochs, avg_loss))

            # --- Wandb Logging (Per Epoch) ---
            if wandb_active and wandb.run:  # Check if wandb.run is active
                try:
                    wandb.log({"epoch": epoch, "avg_loss": avg_loss, "learning_rate": lr})
                except Exception as e:
                    warnings.warn(f"Failed to log to wandb for epoch {epoch}: {e}")
            # --- End Wandb Logging ---

            if epoch % save_interval == 0:
                ckpt_path = os.path.join(save_dir, 'GraphDF_ckpt_{}.pth'.format(epoch))
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        print("Training finished.")

    def run_rand_gen(self, model_conf_dict, checkpoint_path, n_mols=1000, num_min_node=5, num_max_node=64,
                     temperature=[0.3, 0.3]):
        r"""
            Running graph generation for random generation task.
            ... (rest of docstring) ...
        """

        self.get_model('rand_gen', model_conf_dict, checkpoint_path)
        self.model.eval()

        # Determine device for generation based on model's device
        model_device = next(self.model.parameters()).device
        print(f"Generation will run on device: {model_device}")

        all_graphs, pure_valids = [], []
        cnt_mol = 0

        print(f"Attempting to generate {n_mols} graphs...")
        with torch.no_grad():  # Ensure no gradients are computed during generation
            while cnt_mol < n_mols:
                # The generate method of GraphFlowModel needs to handle the device internally
                # or accept device as an argument if it creates new tensors.
                # For now, assuming it uses self.dp and self.node_base_log_probs.device correctly.
                try:
                    mol, no_resample, num_atoms = self.model.generate(
                        min_atoms=num_min_node,
                        max_atoms=num_max_node,
                        temperature=temperature
                    )
                    if mol is not None and (num_atoms >= num_min_node):  # Check if mol is not None
                        cnt_mol += 1
                        all_graphs.append(mol)
                        pure_valids.append(no_resample)
                        if cnt_mol % 100 == 0:
                            print('Generated {}/{} Graphs'.format(cnt_mol, n_mols))
                    elif mol is None:
                        warnings.warn("Generation returned None for a molecule. Skipping.")
                    # Optionally, add a small break or counter to prevent infinite loops if generation consistently fails
                except Exception as e:
                    warnings.warn(f"Error during generation of a molecule: {e}. Skipping this attempt.")
                    # Optionally, break or add a counter for max generation errors
                    # For now, we'll let it continue trying up to n_mols successful generations.

        if cnt_mol < n_mols:
            warnings.warn(f"Desired {n_mols} graphs, but only {cnt_mol} were successfully generated and met criteria.")

        # TODO: Convert to directed graphs if needed by downstream tasks.
        # The current generation process in GraphFlowModel (commented out) produces nx.DiGraph.
        # If the `mol` objects returned are already nx.DiGraph, this comment might be outdated.
        # If they are RDKit Mol objects, conversion to NetworkX might be needed later.

        return all_graphs, pure_valids
