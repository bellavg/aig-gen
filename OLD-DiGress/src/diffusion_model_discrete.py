import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
import networkx as nx
import numpy as np
import pathlib
import logging
import traceback
from collections import defaultdict, Counter
from tqdm import tqdm

# Assuming models, diffusion, metrics, utils are in the same src directory or PYTHONPATH
from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
# Corrected import: Use SumExceptBatchMetric instead of SumExceptBatchKL for KL accumulation
from metrics.abstract_metrics import SumExceptBatchMetric, NLL # Removed SumExceptBatchKL
import utils

# --- Try to import AIG config for mappings ---
import aig_config as aig_cfg

# --- Fallback logic remains the same ---
NODE_INDEX_TO_ENCODING = {
    i: np.array(aig_cfg.NODE_TYPE_ENCODING[key], dtype=np.float32)
    for i, key in enumerate(aig_cfg.NODE_TYPE_KEYS)
}
# Corrected EDGE_INDEX_TO_ENCODING to match AIG config (2 types + NoEdge)
EDGE_INDEX_TO_ENCODING = {
    0: np.array([1., 0., 0.], dtype=np.float32) # NoEdge encoding (needs 3 dims if output is 3)
}
# Ensure the output dimension matches the number of edge types + NoEdge
num_edge_output_classes = len(aig_cfg.EDGE_TYPE_KEYS) + 1 # +1 for NoEdge
for i, key in enumerate(aig_cfg.EDGE_TYPE_KEYS):
     # Create one-hot encoding based on the final output dimension
     one_hot = np.zeros(num_edge_output_classes, dtype=np.float32)
     one_hot[i + 1] = 1.0 # Index 0 is NoEdge, indices 1, 2 are REG, INV
     EDGE_INDEX_TO_ENCODING[i + 1] = one_hot

# Assuming evaluate_aigs.py is in the same directory
from evaluate_aigs import (
    calculate_structural_aig_metrics,
    count_pi_po_paths,
    calculate_uniqueness,
    calculate_novelty,
    load_training_graphs_from_pkl,
    NODE_PI, NODE_PO, NODE_CONST0
)


eval_logger = logging.getLogger("internal_aig_evaluation")
eval_logger.setLevel(logging.INFO)


class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, extra_features, domain_features, sampling_metrics=None, visualization_tools=None):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        # Feature dimensions
        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E'] # Should be 3 (NoEdge, REG, INV)
        self.ydim_output = output_dims['y'] # Should be 0 for AIG
        self.node_dist = nodes_dist

        # Check dimensions match AIG config
        if self.Xdim_output != len(NODE_INDEX_TO_ENCODING):
             print(f"Warning: Model output dim X ({self.Xdim_output}) != AIG config node types ({len(NODE_INDEX_TO_ENCODING)})")
        if self.Edim_output != len(EDGE_INDEX_TO_ENCODING):
             print(f"Warning: Model output dim E ({self.Edim_output}) != AIG config edge types ({len(EDGE_INDEX_TO_ENCODING)})")

        self.dataset_info = dataset_infos

        # Loss and Metrics
        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)
        self.val_nll = NLL()
        # Changed KL metrics to SumExceptBatchMetric to accumulate pre-computed KL sums
        self.val_X_kl = SumExceptBatchMetric()
        self.val_E_kl = SumExceptBatchMetric()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.test_nll = NLL()
        # Changed KL metrics to SumExceptBatchMetric to accumulate pre-computed KL sums
        self.test_X_kl = SumExceptBatchMetric()
        self.test_E_kl = SumExceptBatchMetric()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics # Keep None for AIG
        self.visualization_tools = visualization_tools # Keep None for AIG

        self.extra_features = extra_features
        self.domain_features = domain_features

        # Model Definition (Unchanged)
        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        # Noise Schedule (Unchanged)
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        # Transition Model & Limit Distribution (Unchanged logic)
        if cfg.model.transition == 'uniform':
            # Ensure output dimensions match AIG types
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output if self.Xdim_output > 0 else torch.zeros(0)
            e_limit = torch.ones(self.Edim_output) / self.Edim_output if self.Edim_output > 0 else torch.zeros(0)
            y_limit = torch.ones(self.ydim_output) / self.ydim_output if self.ydim_output > 0 else torch.zeros(0)
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        elif cfg.model.transition == 'marginal':
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types) if torch.sum(node_types) > 0 else torch.ones(self.Xdim_output) / self.Xdim_output
            edge_types = self.dataset_info.edge_types.float()
            # Ensure marginals match output dimension
            if len(edge_types) != self.Edim_output:
                 print(f"Warning: Dataset edge types ({len(edge_types)}) != Model output dim E ({self.Edim_output}). Using uniform for edges.")
                 e_marginals = torch.ones(self.Edim_output) / self.Edim_output if self.Edim_output > 0 else torch.zeros(0)
            else:
                 e_marginals = edge_types / torch.sum(edge_types) if torch.sum(edge_types) > 0 else torch.ones(self.Edim_output) / self.Edim_output

            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output if self.ydim_output > 0 else torch.zeros(0))
        else:
            raise ValueError(f"Unknown transition type {cfg.model.transition}")

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics', 'visualization_tools'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps # Not used, but kept
        self.best_val_nll = 1e8
        self.val_counter = 0

    def convert_to_nx(self, node_indices, edge_indices):
        """ Converts the sampled discrete indices back to a NetworkX graph for AIGs. """
        G = nx.DiGraph()
        n_nodes = node_indices.shape[0]

        # Add nodes with their types
        for i in range(n_nodes):
            node_type_idx = node_indices[i].item()
            if node_type_idx == -1: continue # Skip masked nodes
            # Use the globally defined mapping
            node_one_hot = NODE_INDEX_TO_ENCODING.get(node_type_idx)
            if node_one_hot is None:
                # print(f"Warning: Unknown node type index {node_type_idx} encountered during NX conversion.")
                continue # Skip nodes with unknown types
            G.add_node(i, type=node_one_hot) # Store one-hot encoding as 'type' attribute

        # Add edges with their types
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j: continue # No self-loops
                edge_type_idx = edge_indices[i, j].item()
                # Index 0 represents NoEdge in our AIG setup
                if edge_type_idx > 0: # Only add edge if type is not NoEdge
                    # Use the globally defined mapping
                    edge_one_hot = EDGE_INDEX_TO_ENCODING.get(edge_type_idx)
                    if edge_one_hot is None:
                        # print(f"Warning: Unknown edge type index {edge_type_idx} encountered during NX conversion.")
                        continue # Skip edges with unknown types

                    # Ensure nodes exist before adding edge (important if skipping unknown node types)
                    if G.has_node(i) and G.has_node(j):
                        G.add_edge(i, j, type=edge_one_hot) # Store one-hot encoding as 'type' attribute
        return G

    # --- Training, Validation, Test Steps (Mostly Unchanged Logic) ---
    def training_step(self, data, i):
        if not hasattr(data, 'edge_index') or data.edge_index.numel() == 0:
            # Skip batches with no edges, as they can cause issues with dense conversion
            # Or handle them differently if necessary for your dataset
            # self.print("Warning: Skipping training batch with no edges.")
            return {'loss': torch.tensor(0.0, device=self.device, requires_grad=True)} # Return zero loss
        try:
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask) # Apply mask to zero out features of padded nodes
            X, E = dense_data.X, dense_data.E
            # Ensure data.y exists and is shaped correctly, or provide a placeholder
            y = data.y if hasattr(data, 'y') and data.y is not None else torch.zeros((X.size(0), 0), device=self.device)

            noisy_data = self.apply_noise(X, E, y, node_mask)
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)
            loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                                   true_X=X, true_E=E, true_y=y,
                                   log=i % self.log_every_steps == 0)

            # Optional: Log specific AIG training metrics if implemented
            if self.train_metrics and hasattr(self.train_metrics, 'log_epoch_metrics'): # Check if it's not None and has the method
                 self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                                    log=i % self.log_every_steps == 0)

            return {'loss': loss}
        except Exception as e:
             # Log the error and return a zero loss to avoid crashing
             print(f"Error in training_step {i}: {e}")
             # Consider logging the traceback for detailed debugging
             # import traceback
             # traceback.print_exc()
             return {'loss': torch.tensor(0.0, device=self.device, requires_grad=True)}

    def validation_step(self, data, i):
        if not hasattr(data, 'edge_index') or data.edge_index.numel() == 0:
            # self.print("Warning: Skipping validation batch with no edges.")
            return {'loss': torch.tensor(float('inf'), device=self.device)} # Return inf loss? Or skip?
        try:
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask)
            X, E = dense_data.X, dense_data.E
            y = data.y if hasattr(data, 'y') and data.y is not None else torch.zeros((X.size(0), 0), device=self.device)

            noisy_data = self.apply_noise(X, E, y, node_mask)
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)
            nll = self.compute_val_loss(pred, noisy_data, X, E, y, node_mask, test=False)
            return {'loss': nll}
        except Exception as e:
             print(f"Error in validation_step {i}: {e}")
             traceback.print_exc() # Keep traceback printing
             return {'loss': torch.tensor(float('inf'), device=self.device)} # Return Inf loss on error

    def test_step(self, data, i):
        if not hasattr(data, 'edge_index') or data.edge_index.numel() == 0:
            # self.print("Warning: Skipping test batch with no edges.")
            return {'loss': torch.tensor(float('inf'), device=self.device)}
        try:
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask)
            X, E = dense_data.X, dense_data.E
            y = data.y if hasattr(data, 'y') and data.y is not None else torch.zeros((X.size(0), 0), device=self.device)

            noisy_data = self.apply_noise(X, E, y, node_mask)
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)
            nll = self.compute_val_loss(pred, noisy_data, X, E, y, node_mask, test=True)
            return {'loss': nll}
        except Exception as e:
             print(f"Error in test_step {i}: {e}")
             traceback.print_exc() # Also added here for consistency during testing
             return {'loss': torch.tensor(float('inf'), device=self.device)} # Return Inf loss on error

    # --- Epoch Start/End Callbacks (Mostly Unchanged) ---
    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        if self.train_metrics and hasattr(self.train_metrics, 'reset'): # Check if train_metrics exists and has reset
             self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics() # Logs batch CE values
        log_freq = getattr(self.cfg.general, 'log_every_n_epochs_train_end', 1) # How often to print summary

        if self.current_epoch % log_freq == 0:
             # Calculate combined loss from logged components
             x_ce = to_log.get('train_epoch/x_CE', 0)
             e_ce = to_log.get('train_epoch/E_CE', 0)
             y_ce = to_log.get('train_epoch/y_CE', 0) # Should be 0 or -1 for AIG
             # Ensure components are tensors before summing
             x_ce_tensor = torch.tensor(x_ce) if not isinstance(x_ce, torch.Tensor) else x_ce
             e_ce_tensor = torch.tensor(e_ce) if not isinstance(e_ce, torch.Tensor) else e_ce
             y_ce_tensor = torch.tensor(y_ce) if not isinstance(y_ce, torch.Tensor) else y_ce

             # Handle potential -1 values from failed computes
             x_ce_val = x_ce_tensor.item() if x_ce_tensor.numel() == 1 and x_ce_tensor.item() != -1 else 0.0
             e_ce_val = e_ce_tensor.item() if e_ce_tensor.numel() == 1 and e_ce_tensor.item() != -1 else 0.0
             y_ce_val = y_ce_tensor.item() if y_ce_tensor.numel() == 1 and y_ce_tensor.item() != -1 else 0.0

             # Calculate the sum of valid components
             loss_val = x_ce_val + e_ce_val + y_ce_val

             # --- ADDED DEBUG PRINT ---
             self.print(f"Epoch {self.current_epoch}: DEBUG - X_CE={x_ce_val:.3f}, E_CE={e_ce_val:.3f}, Y_CE={y_ce_val:.3f}, Calculated Sum={loss_val:.3f}")
             # --- END DEBUG PRINT ---

             # Original print statement (now uses the same calculated loss_val)
             self.print(f"Epoch {self.current_epoch}: Train Loss {loss_val:.3f} (X: {x_ce_val:.3f}, E: {e_ce_val:.3f}) ({time.time() - self.start_epoch_time:.1f}s)")

             # Log AIG-specific metrics if available
             if self.train_metrics and hasattr(self.train_metrics, 'log_epoch_metrics'):
                 try:
                     # Assuming train_metrics for AIG doesn't return separate atom/bond metrics
                     epoch_metrics = self.train_metrics.log_epoch_metrics()
                     if epoch_metrics: # Check if anything was returned
                         self.print(f"  Train Metrics: {epoch_metrics}")
                 except Exception as e:
                      print(f"Error logging train metrics: {e}")

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()

    def on_validation_epoch_end(self) -> None:
        # Compute NLL and KL metrics (Standard DiGress NLL calculation)
        # Compute NLL and KL metrics
        try:
            # Attempt to compute metrics directly. If validation_step failed consistently,
            # compute() might raise an error if called before update(), which is caught below.
            val_nll_value = self.val_nll.compute()
            # Use the corrected metric type (SumExceptBatchMetric)
            val_x_kl_value = self.val_X_kl.compute() * self.T # Now just averages the sums
            val_e_kl_value = self.val_E_kl.compute() * self.T # Now just averages the sums
            val_x_logp_value = self.val_X_logp.compute()
            val_e_logp_value = self.val_E_logp.compute()

        # Catch potential errors during .compute() if metrics weren't updated
        # Also catches the specific error if compute is called before update
        except (RuntimeError, Exception) as e:
            print(f"Error computing validation NLL/KL metrics (possibly due to failed steps): {e}")
            # If NLL compute failed, likely means no steps succeeded
            if isinstance(self.val_nll.compute(), torch.Tensor) and torch.isnan(
                    self.val_nll.compute()).any():  # Check if compute returns NaN
                val_nll_value = torch.tensor(float('nan'))
            else:  # Otherwise set to Inf or handle based on specific error if needed
                val_nll_value = torch.tensor(float('inf'))

            val_x_kl_value, val_e_kl_value, val_x_logp_value, val_e_logp_value = 0.0, 0.0, 0.0, 0.0
            # You might still see the UserWarning about compute before update, but this handles the crash.

            # Use the corrected metric type (SumExceptBatchMetric)
            val_x_kl_value = self.val_X_kl.compute() * self.T if self.val_X_kl.total > 0 else 0.0
            val_e_kl_value = self.val_E_kl.compute() * self.T if self.val_E_kl.total > 0 else 0.0
            val_x_logp_value = self.val_X_logp.compute() if self.val_X_logp.total > 0 else 0.0
            val_e_logp_value = self.val_E_logp.compute() if self.val_E_logp.total > 0 else 0.0

        except Exception as e:
            print(f"Error computing validation NLL/KL metrics: {e}")
            val_nll_value = torch.tensor(float('inf')) # Set to infinity on error
            val_x_kl_value, val_e_kl_value, val_x_logp_value, val_e_logp_value = 0.0, 0.0, 0.0, 0.0

        # Log NLL and KL metrics
        if wandb.run:
            wandb.log({"val/epoch_NLL": val_nll_value,
                       "val/X_kl": val_x_kl_value,
                       "val/E_kl": val_e_kl_value,
                       "val/X_logp": val_x_logp_value,
                       "val/E_logp": val_e_logp_value}, commit=False) # Commit later with validity

        self.print(f"Epoch {self.current_epoch}: Val NLL {val_nll_value:.4f}")
        self.log("val/epoch_NLL", val_nll_value, sync_dist=True) # Log for checkpointing

        # Update best NLL
        if val_nll_value < self.best_val_nll:
            self.best_val_nll = val_nll_value
        self.print(f'Val NLL: {val_nll_value:.4f} \t Best val NLL: {self.best_val_nll:.4f}')

        # --- Perform Validity Check during Validation (AIG Specific) ---
        self.val_counter += 1
        # Use separate config flags for validation sampling
        val_samples_to_generate = getattr(self.cfg.general, 'val_samples_to_generate', 64) # Default 64
        sample_every_val = getattr(self.cfg.general, 'sample_every_val', 1) # Default 1

        # Check if it's time to sample and if generation count is > 0
        if self.val_counter % sample_every_val == 0 and val_samples_to_generate > 0:
            self.print("Sampling graphs for validation validity check...")
            start_time_val_sampling = time.time()
            samples_left_to_generate = val_samples_to_generate
            generated_samples_raw = [] # Store raw tensor outputs
            ident = 0
            batch_size_sample = self.cfg.train.batch_size # Use train batch size for sampling

            # Sampling loop
            while samples_left_to_generate > 0:
                to_generate = min(samples_left_to_generate, batch_size_sample)
                try:
                    # Call sample_batch - returns list of [node_indices, edge_indices] tensors
                    batch_samples_raw = self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                          save_final=0, keep_chain=0, number_chain_steps=0) # No saving needed

                    # *** CRITICAL CHECK: Ensure tensors are moved to CPU *before* conversion ***
                    processed_batch = []
                    for node_idx_tensor, edge_idx_tensor in batch_samples_raw:
                        # Move tensors to CPU here if they aren't already
                        processed_batch.append([node_idx_tensor.cpu(), edge_idx_tensor.cpu()])

                    generated_samples_raw.extend(processed_batch) # Add CPU tensors to list

                except Exception as e:
                     # Log error during sampling
                     print(f"\nError during validation sampling batch starting at ID {ident}: {e}")
                     # import traceback
                     # traceback.print_exc() # Uncomment for detailed traceback
                     # Decide whether to break or continue sampling
                     # break # Option: Stop sampling on first error
                ident += to_generate
                samples_left_to_generate -= to_generate

            # --- Convert and Check Validity ---
            generated_nx_graphs = []
            if generated_samples_raw: # Proceed only if some raw samples were generated
                for i, sample_data in enumerate(generated_samples_raw):
                    try:
                        # sample_data[0] is node_indices (CPU tensor)
                        # sample_data[1] is edge_indices (CPU tensor)
                        nx_graph = self.convert_to_nx(sample_data[0], sample_data[1])
                        if nx_graph is not None:
                            generated_nx_graphs.append(nx_graph)
                    except Exception as e:
                        # Log conversion errors but continue
                        # print(f"Warning: Error converting raw sample {i} to NetworkX: {e}")
                        pass # Ignore conversion errors for validation check

            # Calculate validity
            num_generated = len(generated_nx_graphs)
            num_valid = 0
            if num_generated > 0:
                for graph in generated_nx_graphs: # Iterate through successfully converted graphs
                     try:
                         # Only check structural validity
                         struct_metrics = calculate_structural_aig_metrics(graph)
                         if struct_metrics.get('is_structurally_valid', 0.0) > 0.5:
                             num_valid += 1
                     except Exception as e:
                          # Log validity check errors but continue
                          # print(f"Warning: Error during validity check for graph: {e}")
                          pass # Ignore errors in validity check for robustness

                validity_fraction = num_valid / num_generated
                self.print(f"Validation Validity Check: {num_valid}/{num_generated} ({validity_fraction*100:.2f}%)")
                if wandb.run:
                    wandb.log({"val/validity": validity_fraction}, commit=True) # Commit validity with NLL
            else:
                 self.print("No graphs generated or converted successfully for validation validity check.")
                 if wandb.run:
                     wandb.log({"val/validity": 0.0}, commit=True) # Log 0 validity

            self.print(f'Validation sampling & validity check done. Took {time.time() - start_time_val_sampling:.2f} seconds.')
        elif wandb.run:
             # If not sampling, commit the NLL/KL metrics logged earlier
             wandb.log({}, commit=True)


    # --- Test Epoch End (AIG Specific Evaluation) ---
    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and evaluate using imported AIG functions. """
        self.print("\n--- Test Evaluation ---")
        # Compute and log NLL metrics
        try:
            test_nll_value = self.test_nll.compute()
            if self.test_nll.total == 0: test_nll_value = torch.tensor(float('inf')) # Handle no updates
            # Use the corrected metric type (SumExceptBatchMetric)
            test_x_kl_value = self.test_X_kl.compute() * self.T if self.test_X_kl.total > 0 else 0.0
            test_e_kl_value = self.test_E_kl.compute() * self.T if self.test_E_kl.total > 0 else 0.0
            test_x_logp_value = self.test_X_logp.compute() if self.test_X_logp.total > 0 else 0.0
            test_e_logp_value = self.test_E_logp.compute() if self.test_E_logp.total > 0 else 0.0
        except Exception as e:
             print(f"Error computing test NLL metrics: {e}")
             test_nll_value = torch.tensor(float('inf'))
             test_x_kl_value, test_e_kl_value, test_x_logp_value, test_e_logp_value = 0.0, 0.0, 0.0, 0.0

        log_dict = {"test/epoch_NLL": test_nll_value, "test/X_kl": test_x_kl_value, "test/E_kl": test_e_kl_value,
                    "test/X_logp": test_x_logp_value, "test/E_logp": test_e_logp_value}
        if wandb.run:
            wandb.log(log_dict, commit=False) # Commit later with VUN

        self.print(f"Test NLL: {test_nll_value:.4f} -- Test KL (X E): {test_x_kl_value:.2f}, {test_e_kl_value:.2f}")

        # --- Generate final samples ---
        self.print("Generating final samples for evaluation...")
        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        generated_samples_raw = [] # Store raw tensor outputs
        ident = 0
        batch_size_sample = self.cfg.train.batch_size # Use train batch size

        start_time_test_sampling = time.time()
        while samples_left_to_generate > 0:
            current_batch_num = ident // batch_size_sample + 1
            self.print(f'\rGenerating batch {current_batch_num}...', end='', flush=True)
            to_generate = min(samples_left_to_generate, batch_size_sample)
            try:
                # Call sample_batch - returns list of [node_indices, edge_indices] tensors
                batch_samples_raw = self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                      save_final=0, keep_chain=0, number_chain_steps=0)

                # *** CRITICAL CHECK: Ensure tensors are moved to CPU *before* conversion ***
                processed_batch = []
                for node_idx_tensor, edge_idx_tensor in batch_samples_raw:
                     processed_batch.append([node_idx_tensor.cpu(), edge_idx_tensor.cpu()])
                generated_samples_raw.extend(processed_batch) # Add CPU tensors to list

            except Exception as e:
                 print(f"\nError during test sampling batch starting at ID {ident}: {e}")
                 # import traceback; traceback.print_exc() # Uncomment for detailed traceback
                 # break # Option: Stop sampling on error
            ident += to_generate
            samples_left_to_generate -= to_generate
        self.print(f"\nGenerated {len(generated_samples_raw)} raw samples in {time.time() - start_time_test_sampling:.2f}s.")

        # --- Convert raw samples to NetworkX graphs ---
        self.print("Converting generated samples to NetworkX graphs...")
        generated_nx_graphs = []
        for i, sample_data in enumerate(tqdm(generated_samples_raw, desc="Converting to NX")):
            try:
                # sample_data[0] is node_indices (CPU tensor)
                # sample_data[1] is edge_indices (CPU tensor)
                nx_graph = self.convert_to_nx(sample_data[0], sample_data[1])
                if nx_graph is not None:
                    generated_nx_graphs.append(nx_graph)
                # else: print(f"Warning: Failed to convert sample {i} to NetworkX graph.") # Less verbose
            except Exception as e:
                # print(f"Error converting sample {i} to NetworkX: {e}") # Less verbose
                pass
        self.print(f"Successfully converted {len(generated_nx_graphs)} samples to NetworkX.")

        if not generated_nx_graphs:
            self.print("No valid NetworkX graphs generated. Skipping evaluation.")
            if wandb.run: wandb.log({}, commit=True) # Commit NLL metrics even if eval fails
            return

        # --- Load Training Data for Novelty ---
        # (Keep the existing logic for loading training graphs)
        train_graphs = None
        try:
            # Construct path relative to the *current working directory* where main.py is run
            # Assumes 'data/aig/raw' exists relative to CWD
            train_data_dir_for_eval = os.path.join("data", "aig", "raw") # Example relative path
            # Or use absolute path based on config if needed:
            # train_data_dir_for_eval = os.path.join(self.cfg.dataset.datadir, "raw")

            train_data_prefix_for_eval = "real_aigs_part_" # Assuming this prefix
            num_train_files_for_eval = 4 # Load first 4 parts for novelty check

            if os.path.isdir(train_data_dir_for_eval):
                 self.print(f"Loading training graphs from: {train_data_dir_for_eval}")
                 train_graphs = load_training_graphs_from_pkl(train_data_dir_for_eval, train_data_prefix_for_eval, num_train_files_for_eval)
                 if train_graphs is None: self.print("Failed to load training graphs.")
                 elif not train_graphs: self.print("Loaded 0 training graphs.")
                 else: self.print(f"Loaded {len(train_graphs)} training graphs.")
            else:
                 self.print(f"Training data directory not found: {train_data_dir_for_eval}")
        except Exception as e:
            self.print(f"Error loading training graphs: {e}")


        # --- Perform Direct Evaluation (Full V.U.N.) ---
        # (Keep the existing evaluation and reporting logic)
        self.print("Starting internal evaluation...")
        num_total = len(generated_nx_graphs)
        valid_graphs = []
        aggregate_metrics = defaultdict(list)
        aggregate_path_metrics = defaultdict(list)
        failed_constraints_summary = Counter()

        for i, graph in enumerate(tqdm(generated_nx_graphs, desc="Evaluating Validity")):
            struct_metrics = calculate_structural_aig_metrics(graph)
            for key, value in struct_metrics.items():
                 if isinstance(value, (int, float, bool)): aggregate_metrics[key].append(float(value))
            if struct_metrics.get('is_structurally_valid', 0.0) > 0.5:
                valid_graphs.append(graph)
                try:
                     path_metrics = count_pi_po_paths(graph)
                     if path_metrics.get('error') is None:
                        for key, value in path_metrics.items():
                            if isinstance(value, (int, float)): aggregate_path_metrics[key].append(value)
                     else: eval_logger.warning(f"Skipping path metrics for valid graph {i} due to error: {path_metrics['error']}")
                except Exception as e: eval_logger.error(f"Error calculating path metrics for valid graph {i}: {e}")
            else:
                for reason in struct_metrics.get('constraints_failed', ["Unknown Failure"]):
                    failed_constraints_summary[reason] += 1

        num_valid_structurally = len(valid_graphs)
        uniqueness_score, num_unique = calculate_uniqueness(valid_graphs)

        novelty_score, num_novel = (-1.0, -1)
        if train_graphs is not None:
            novelty_score, num_novel = calculate_novelty(valid_graphs, train_graphs)
        else:
            self.print("Skipping novelty calculation as training graphs were not loaded.")

        # --- Print Evaluation Summary ---
        validity_fraction = (num_valid_structurally / num_total) if num_total > 0 else 0.0
        validity_percentage = validity_fraction * 100
        print("\n--- Internal AIG V.U.N. Evaluation Summary ---")
        print(f"Total Graphs Generated          : {num_total}")
        print(f"Structurally Valid AIGs (V)     : {num_valid_structurally} ({validity_percentage:.2f}%)")
        if num_valid_structurally > 0:
             print(f"Unique Valid AIGs             : {num_unique}")
             print(f"Uniqueness (U) among valid    : {uniqueness_score:.4f} ({uniqueness_score*100:.2f}%)")
             if train_graphs is not None:
                 print(f"Novel Valid AIGs vs Train Set : {num_novel}")
                 print(f"Novelty (N) among valid       : {novelty_score:.4f} ({novelty_score*100:.2f}%)")
             else:
                 print(f"Novelty (N) among valid       : Not calculated")
        else:
             print(f"Uniqueness (U) among valid    : N/A")
             print(f"Novelty (N) among valid       : N/A")

        # (Keep the detailed metric printing as before)
        print("\n--- Average Structural Metrics (All Generated Graphs) ---")
        for key, values in sorted(aggregate_metrics.items()):
            if key == 'is_structurally_valid': continue
            if not values: continue
            avg_value = np.mean(values)
            if key == 'is_dag': print(f"  - Avg {key:<27}: {avg_value*100:.2f}%")
            else: print(f"  - Avg {key:<27}: {avg_value:.3f}")

        print("\n--- Constraint Violation Summary (Across Invalid Graphs) ---")
        num_invalid_graphs = num_total - num_valid_structurally
        if num_invalid_graphs == 0: print("  No structural violations detected.")
        else:
            sorted_reasons = sorted(failed_constraints_summary.items(), key=lambda item: item[1], reverse=True)
            print(f"  (Violations summarized across {num_invalid_graphs} invalid graphs)")
            for reason, count in sorted_reasons:
                reason_percentage_of_invalid = (count / num_invalid_graphs) * 100 if num_invalid_graphs > 0 else 0
                print(f"  - {reason:<45}: {count:<6} graphs ({reason_percentage_of_invalid:.1f}% of invalid)")

        print("\n--- Average Path Connectivity Metrics (Valid Graphs Only) ---")
        num_graphs_for_path_metrics = len(aggregate_path_metrics.get('num_pi', []))
        if num_graphs_for_path_metrics == 0: print("  No structurally valid graphs to calculate path metrics for.")
        else:
            print(f"  (Based on {num_graphs_for_path_metrics} structurally valid graphs)")
            for key, values in sorted(aggregate_path_metrics.items()):
                 if key == 'error' or not values: continue
                 avg_value = np.mean(values)
                 print(f"  - Avg {key:<27}: {avg_value:.3f}")
        print("------------------------------------")

        # --- Log key evaluation metrics to Wandb if available ---
        if wandb.run:
            eval_log = {
                "test/validity": validity_fraction,
                "test/uniqueness_valid": uniqueness_score if num_valid_structurally > 0 else 0.0,
                "test/novelty_valid": novelty_score if train_graphs is not None and num_valid_structurally > 0 else -1.0,
                "test/num_valid": num_valid_structurally,
                "test/num_unique": num_unique,
                "test/num_novel": num_novel if train_graphs is not None else -1
            }
            # Add average metrics if desired
            for key, values in aggregate_metrics.items():
                 if values: eval_log[f"test/avg_{key}"] = np.mean(values)
            for key, values in aggregate_path_metrics.items():
                 if values: eval_log[f"test/avg_valid_{key}"] = np.mean(values)

            wandb.log(eval_log, commit=True) # Commit evaluation metrics along with NLL

        self.print("Test epoch finished.")


    # --- Core Diffusion Logic (Unchanged from previous AIG version) ---
    def kl_prior(self, X, E, node_mask):
        # This function calculates KL divergence between q(z_T|x_0) and p(z_T)
        # p(z_T) is the limit distribution (uniform or marginal)
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones # Integer timesteps T
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts) # Get alpha_bar_T
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device) # Get Q_T_bar

        # Calculate q(z_T|x_0) = x_0 @ Q_T_bar
        probX = X @ Qtb.X
        probE = E @ Qtb.E.unsqueeze(1) # Add dimension for broadcasting

        bs, n, _ = probX.shape
        # Get the limit distribution p(z_T)
        limit_X = self.limit_dist.X.to(X.device)[None, None, :].expand(bs, n, -1)
        limit_E = self.limit_dist.E.to(E.device)[None, None, None, :].expand(bs, n, n, -1)

        # Mask distributions before KL calculation
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(
            true_X=limit_X.clone(), true_E=limit_E.clone(), pred_X=probX, pred_E=probE, node_mask=node_mask
        )

        # Calculate KL divergence: sum over features, sum over nodes/edges, mean over batch
        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        # Sum over all dimensions except batch
        return diffusion_utils.sum_except_batch(kl_distance_X) + diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        # Computes the KL divergence loss for intermediate timesteps t > 0
        # KL( q(z_{t-1}|z_t, x_0) || p_theta(z_{t-1}|z_t) )

        # Get predicted probabilities for x_0
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1) # y is likely unused for AIG

        # Get transition matrices
        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Calculate the "true" posterior q(z_{t-1}|z_t, x_0)
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'], y_t=noisy_data['y_t'],
                                                            Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        # Calculate the model's predicted posterior p_theta(z_{t-1}|z_t)
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'], y_t=noisy_data['y_t'],
                                                            Qt=Qt, Qsb=Qsb, Qtb=Qtb)

        # Reshape edge probabilities and mask
        bs, n, d = X.shape
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))
        prob_true_X, prob_true_E, prob_pred_X, prob_pred_E = diffusion_utils.mask_distributions(
            true_X=prob_true.X, true_E=prob_true.E, pred_X=prob_pred.X, pred_E=prob_pred.E, node_mask=node_mask
        )

        # Calculate KL divergence
        # Note: F.kl_div expects log-probabilities for the input
        kl_x = F.kl_div(input=prob_pred_X.log(), target=prob_true_X, reduction='none')
        kl_e = F.kl_div(input=prob_pred_E.log(), target=prob_true_E, reduction='none')

        # Sum KL divergence over all dimensions except batch
        kl_x_batch_sum = diffusion_utils.sum_except_batch(kl_x)
        kl_e_batch_sum = diffusion_utils.sum_except_batch(kl_e)

        # Update validation/test metrics using the corrected metric type (SumExceptBatchMetric)
        metric_x_kl = self.test_X_kl if test else self.val_X_kl
        metric_e_kl = self.test_E_kl if test else self.val_E_kl
        metric_x_kl.update(kl_x_batch_sum) # Now correctly updates SumExceptBatchMetric
        metric_e_kl.update(kl_e_batch_sum) # Now correctly updates SumExceptBatchMetric

        # Return the sum for the current batch (used in NLL calculation)
        return kl_x_batch_sum + kl_e_batch_sum


    def reconstruction_logp(self, t, X, E, node_mask):
        # Computes -log p(x_0|z_0)
        t_zeros = torch.zeros_like(t) # Time step 0
        beta_0 = self.noise_schedule(t_zeros) # Beta for step 0
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device) # Q_1 matrix (t=1)

        # Sample z_0 ~ q(z_0 | x_0) = x_0 @ Q_1
        probX0 = X @ Q0.X
        probE0 = E @ Q0.E.unsqueeze(1)
        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)
        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y # y is likely empty for AIG

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Get model prediction for p(x_0|z_0)
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)} # Pass t=0
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Calculate probabilities p(x_0|z_0)
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1) # y is likely unused

        # Mask probabilities before calculating logp
        # Ensure masked values don't affect logp calculation by setting to uniform prob doesn't work well
        # Instead, we calculate logp only on unmasked elements later
        probX0 = probX0 + 1e-9 # Add epsilon to prevent log(0)
        probE0 = probE0 + 1e-9

        # Return the probabilities, masking will be handled during loss calculation
        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)


    def apply_noise(self, X, E, y, node_mask):
        # Sample timestep t
        lowest_t = 0 if self.training else 1 # Don't sample t=0 during validation/testing NLL calculation
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()
        s_int = t_int - 1 # Previous timestep

        # Normalize timesteps
        t_float = t_int / self.T
        s_float = s_int / self.T

        # Get noise schedule parameters
        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        # Get transition matrix Q_t_bar
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)

        # Calculate probabilities q(z_t | x_0) = x_0 @ Q_t_bar
        probX = X @ Qtb.X
        probE = E @ Qtb.E.unsqueeze(1)

        # Sample z_t ~ Cat(q(z_t | x_0))
        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        # Convert sampled indices to one-hot
        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output).float()
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output).float()

        # Apply mask
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        # Return dictionary with noisy data and schedule parameters
        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        # Computes the NLL loss for validation/testing
        t = noisy_data['t'] # Normalized time
        bs, n, _ = X.shape

        # 1. Calculate -log p(N) - Prior probability of graph size
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N) # Get log prob from distribution

        # 2. Calculate KL(q(z_T|x_0) || p(z_T)) - KL divergence at the last timestep
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Calculate L_t = KL(q(z_{t-1}|z_t, x_0) || p_theta(z_{t-1}|z_t)) for t > 0
        # Sum over all intermediate timesteps (represented by the single sampled t in this batch)
        loss_all_t_batch = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Calculate L_0 = -log p_theta(x_0|z_0) - Reconstruction loss
        metric_x_logp = self.test_X_logp if test else self.val_X_logp
        metric_e_logp = self.test_E_logp if test else self.val_E_logp
        # Reset metrics before calculating L0 for this batch
        metric_x_logp.reset()
        metric_e_logp.reset()

        prob0 = self.reconstruction_logp(t, X, E, node_mask) # Get p(x_0|z_0)

        # Calculate negative log likelihood -log p(x_0|z_0) only for original nodes/edges
        # Select the probability of the true class using the original one-hot vectors X, E
        log_prob_X = (X * prob0.X.log()).sum(dim=-1) # Sum over classes
        log_prob_E = (E * prob0.E.log()).sum(dim=-1) # Sum over classes

        # Apply mask: sum only over non-masked nodes/edges
        neg_log_px = - (log_prob_X * node_mask).sum(dim=1) # Sum over nodes
        # Sum over unique edges (upper triangle) applying mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        upper_triangle_mask = torch.triu(torch.ones_like(E[..., 0]), diagonal=1).bool()
        neg_log_pe = - (log_prob_E * edge_mask * upper_triangle_mask).sum(dim=(1, 2)) # Sum over edges

        # Update metrics with the negative log probabilities for the batch
        metric_x_logp.update(neg_log_px)
        metric_e_logp.update(neg_log_pe)

        # L0 for the batch is the sum of negative log probabilities
        loss_term_0_batch = neg_log_px + neg_log_pe

        # Combine terms for NLL for the batch
        # NLL = -log p(x) = -log p(N) + KL_prior + sum_{t=1}^T KL_t - log p(x_0|z_0)
        nlls_batch = - log_pN + kl_prior + loss_all_t_batch + loss_term_0_batch

        # Update overall NLL metric
        nll_metric = self.test_nll if test else self.val_nll
        nll_metric.update(nlls_batch) # Update with batch NLL values
        nll = nll_metric.compute() # Compute average NLL across updates

        return nll


    def forward(self, noisy_data, extra_data, node_mask):
        # Concatenate noisy data with extra features and pass to the model
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        # Ensure y has a batch dimension even if empty
        y_t = noisy_data['y_t'].view(X.size(0), -1) if noisy_data['y_t'].numel() > 0 else torch.zeros(X.size(0), 0, device=X.device)
        extra_y = extra_data.y.view(X.size(0), -1) if extra_data.y.numel() > 0 else torch.zeros(X.size(0), 0, device=X.device)
        y = torch.hstack((y_t, extra_y)).float()

        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):
        """ Samples a batch of graphs using the reverse diffusion process. """
        # Determine graph sizes and mask
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif isinstance(num_nodes, int):
            n_nodes = torch.full((batch_size,), num_nodes, device=self.device, dtype=torch.int)
        else: # Should be a tensor
            n_nodes = num_nodes.to(self.device)
        n_max = torch.max(n_nodes).item()
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample initial noise z_T from the limit distribution
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist.to(self.device),
                                                             node_mask=node_mask)
        X, E = z_T.X, z_T.E
        # Initialize y as empty tensor with correct batch size
        y = torch.zeros((batch_size, 0), device=self.device, dtype=self.model_dtype)

        # Reverse diffusion loop
        for s_int in reversed(range(0, self.T)):
            s_array = torch.full((batch_size, 1), s_int, device=self.device, dtype=self.model_dtype)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Call sample_p_zs_given_zt, which now only returns the one-hot state
            sampled_s_one_hot = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)

            # Update the state for the next iteration
            X, E, y = sampled_s_one_hot.X, sampled_s_one_hot.E, sampled_s_one_hot.y
            # Ensure symmetry (although sample_p_zs_given_zt should handle it)
            E = 0.5 * (E + E.transpose(1, 2))

        # After the loop, X and E contain the final one-hot z_0
        # Collapse to get discrete indices
        sampled_0_discrete = utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse=True)
        X_final, E_final = sampled_0_discrete.X, sampled_0_discrete.E # These are integer indices

        # Convert final indices to list format for evaluation
        molecule_list = [] # Keep original name for consistency
        for i in range(batch_size):
            n = n_nodes[i].item()
            # Extract the relevant part of the tensors and move to CPU
            atom_types = X_final[i, :n].cpu()
            edge_types = E_final[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        return molecule_list # Return list of [node_indices, edge_indices]

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """ Samples p(z_{t-1} | z_t). Returns the *one-hot* encoding of z_{t-1}. """
        bs, n, _ = X_t.shape
        # Get noise schedule parameters and transition matrices
        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Prepare input for the network
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)

        # Get model prediction p_theta(x_0 | z_t)
        pred = self.forward(noisy_data, extra_data, node_mask)
        pred_X = F.softmax(pred.X, dim=-1)
        pred_E = F.softmax(pred.E, dim=-1)

        # Calculate posterior probabilities q(z_{t-1} | z_t, x_0) for all possible x_0
        # This uses the formula derived from Bayes' theorem (Equation 1 in paper)
        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(
            X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
        ) # Shape: bs, n, d0, d_{t-1}
        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(
            X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
        ) # Shape: bs, n*n, d0, d_{t-1}

        # Calculate model posterior p_theta(z_{t-1} | z_t) by marginalizing over x_0
        # p_theta(z_{t-1}|z_t) = sum_{x_0} p_theta(x_0|z_t) q(z_{t-1}|z_t, x_0)
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X # bs, n, d0, d_{t-1}
        unnormalized_prob_X = weighted_X.sum(dim=2) # Sum over d0 -> bs, n, d_{t-1}

        # Reshape E for broadcasting before weighting
        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1])) # bs, n*n, d0
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E # bs, n*n, d0, d_{t-1}
        unnormalized_prob_E = weighted_E.sum(dim=2) # Sum over d0 -> bs, n*n, d_{t-1}

        # Normalize probabilities ensuring no division by zero
        unnormalized_prob_X = unnormalized_prob_X + 1e-7
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)

        unnormalized_prob_E = unnormalized_prob_E + 1e-7
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, -1) # Reshape back to bs, n, n, d_{t-1}

        # Sample discrete indices z_{t-1} from the calculated probabilities
        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        # Convert sampled indices to one-hot encoding
        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        # Symmetrize edge indices before one-hot encoding
        E_s_idx = sampled_s.E
        E_s_idx = torch.triu(E_s_idx, diagonal=1) # Operate on upper triangle
        E_s_idx = E_s_idx + E_s_idx.transpose(1, 2) # Make symmetric
        E_s = F.one_hot(E_s_idx, num_classes=self.Edim_output).float()

        # Create PlaceHolder for the one-hot state z_{t-1}
        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros_like(y_t)) # y remains empty

        # Apply mask and ensure correct dtype/device
        return out_one_hot.mask(node_mask).type_as(y_t)


    def compute_extra_data(self, noisy_data):
        """ Computes extra features based on noisy data. Returns a PlaceHolder. """
        # If extra features are enabled, compute them
        if self.extra_features and hasattr(self.extra_features, '__call__'): # Check if callable
             extra_features = self.extra_features(noisy_data)
        else: # Otherwise, create empty placeholders
             extra_features = utils.PlaceHolder(
                 X=torch.zeros((*noisy_data['X_t'].shape[:-1], 0)).type_as(noisy_data['X_t']),
                 E=torch.zeros((*noisy_data['E_t'].shape[:-1], 0)).type_as(noisy_data['E_t']),
                 y=torch.zeros((noisy_data['X_t'].shape[0], 0)).type_as(noisy_data['X_t']) # Use X_t for type/device
             )

        # Same logic for domain-specific features (likely none for AIG)
        if self.domain_features and hasattr(self.domain_features, '__call__'):
             domain_features = self.domain_features(noisy_data)
        else:
             domain_features = utils.PlaceHolder(
                 X=torch.zeros((*noisy_data['X_t'].shape[:-1], 0)).type_as(noisy_data['X_t']),
                 E=torch.zeros((*noisy_data['E_t'].shape[:-1], 0)).type_as(noisy_data['E_t']),
                 y=torch.zeros((noisy_data['X_t'].shape[0], 0)).type_as(noisy_data['X_t'])
             )

        # Concatenate features
        extra_X = torch.cat((extra_features.X, domain_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, domain_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, domain_features.y), dim=-1)

        # Add time embedding 't' to global features 'y'
        t = noisy_data['t'] # Normalized time
        # Ensure t has the correct batch dimension if extra_y is empty
        if extra_y.numel() == 0:
             extra_y = t.view(extra_y.size(0), -1) # Reshape t to (bs, 1)
        else:
             extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def configure_optimizers(self):
        # Standard AdamW optimizer
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        # Print model dimensions on start
        if self.trainer.datamodule:
             try:
                 self.train_iterations = len(self.trainer.datamodule.train_dataloader())
             except Exception as e:
                  print(f"Warning: Could not determine train_iterations: {e}")
                  self.train_iterations = 1 # Default
        else:
             self.train_iterations = 1
        self.print("Input feature dims (X, E, y):", self.Xdim, self.Edim, self.ydim)
        self.print("Output feature dims (X, E, y):", self.Xdim_output, self.Edim_output, self.ydim_output)

    def on_test_epoch_start(self) -> None:
        # Reset test metrics
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()

