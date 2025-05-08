import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
import pickle
import networkx as nx
# import subprocess # Removed subprocess
import numpy as np
import pathlib # Added for path manipulation
import logging # Added for evaluation logging consistency
from collections import defaultdict, Counter # Added for evaluation
from tqdm import tqdm # Added for evaluation progress

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
import utils

# --- Try to import AIG config for mappings ---
try:
    import aig_config as aig_cfg
    NODE_INDEX_TO_ENCODING = {
        i: np.array(aig_cfg.NODE_TYPE_ENCODING[key], dtype=np.float32)
        for i, key in enumerate(aig_cfg.NODE_TYPE_KEYS)
    }
    EDGE_INDEX_TO_ENCODING = {
        0: np.array([1., 0., 0.], dtype=np.float32) # NoEdge encoding
    }
    for i, key in enumerate(aig_cfg.EDGE_TYPE_KEYS):
         EDGE_INDEX_TO_ENCODING[i + 1] = np.array(aig_cfg.EDGE_LABEL_ENCODING[key], dtype=np.float32)
    print("Successfully loaded AIG config for conversion.")
except ImportError:
    print("WARNING: Could not import G2PT.configs.aig. Using fallback mappings.")
    NODE_INDEX_TO_ENCODING = {i: np.array([1.0 if j == i else 0.0 for j in range(4)]) for i in range(4)}
    EDGE_INDEX_TO_ENCODING = {i: np.array([1.0 if j == i else 0.0 for j in range(3)]) for i in range(3)}
except AttributeError:
     print("WARNING: G2PT.configs.aig missing attributes. Using fallback mappings.")
     NODE_INDEX_TO_ENCODING = {i: np.array([1.0 if j == i else 0.0 for j in range(4)]) for i in range(4)}
     EDGE_INDEX_TO_ENCODING = {i: np.array([1.0 if j == i else 0.0 for j in range(3)]) for i in range(3)}

# --- Import evaluation functions ---
try:
    from evaluate_aigs import (
        calculate_structural_aig_metrics,
        count_pi_po_paths,
        calculate_uniqueness,
        calculate_novelty,
        load_training_graphs_from_pkl,
        NODE_PI, NODE_PO, NODE_CONST0
    )
    print("Successfully imported evaluation functions from evaluate_aigs.")
except ImportError as e:
    print(f"ERROR: Could not import functions from evaluate_aigs.py: {e}")
    print("Evaluation in on_test_epoch_end will be skipped.")
    def calculate_structural_aig_metrics(*args, **kwargs): return {'is_structurally_valid': 0.0, 'constraints_failed': ['Import Failed']}
    def count_pi_po_paths(*args, **kwargs): return {}
    def calculate_uniqueness(*args, **kwargs): return 0.0, 0
    def calculate_novelty(*args, **kwargs): return 0.0, 0
    def load_training_graphs_from_pkl(*args, **kwargs): return None
    NODE_PI, NODE_PO, NODE_CONST0 = "NODE_PI", "NODE_PO", "NODE_CONST0"

eval_logger = logging.getLogger("internal_aig_evaluation")
eval_logger.setLevel(logging.INFO)


class DiscreteDenoisingDiffusion(pl.LightningModule):
    # __init__ remains the same as the previous version
    def __init__(self, cfg, dataset_infos, train_metrics, extra_features, domain_features, sampling_metrics=None, visualization_tools=None): # Added defaults
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        # Store sampling_metrics and visualization_tools even if None
        self.sampling_metrics = sampling_metrics
        self.visualization_tools = visualization_tools

        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
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
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    def convert_to_nx(self, node_indices, edge_indices):
        """ Converts the sampled discrete indices back to a NetworkX graph. """
        G = nx.DiGraph()
        n_nodes = node_indices.shape[0]
        for i in range(n_nodes):
            node_type_idx = node_indices[i].item()
            if node_type_idx == -1: continue
            node_one_hot = NODE_INDEX_TO_ENCODING.get(node_type_idx)
            if node_one_hot is None: continue
            G.add_node(i, type=node_one_hot)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j: continue
                edge_type_idx = edge_indices[i, j].item()
                if edge_type_idx > 0:
                    edge_one_hot = EDGE_INDEX_TO_ENCODING.get(edge_type_idx)
                    if edge_one_hot is None: continue
                    if G.has_node(i) and G.has_node(j): G.add_edge(i, j, type=edge_one_hot)
        return G

    def training_step(self, data, i):
        if not hasattr(data, 'edge_index') or data.edge_index.numel() == 0:
            return {'loss': torch.tensor(0.0, device=self.device, requires_grad=True)}
        try:
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask)
            X, E = dense_data.X, dense_data.E
            noisy_data = self.apply_noise(X, E, data.y, node_mask)
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)
            loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                                   true_X=X, true_E=E, true_y=data.y,
                                   log=i % self.log_every_steps == 0)
            if self.train_metrics:
                self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                                   log=i % self.log_every_steps == 0)
            return {'loss': loss}
        except Exception as e:
             print(f"Error in training_step: {e}")
             return {'loss': torch.tensor(0.0, device=self.device, requires_grad=True)}

    def validation_step(self, data, i):
        try:
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask)
            noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)
            nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False)
            return {'loss': nll}
        except Exception as e:
             print(f"Error in validation_step: {e}")
             return {'loss': torch.tensor(float('inf'), device=self.device)}

    def test_step(self, data, i):
        try:
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask)
            noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)
            nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)
            return {'loss': nll}
        except Exception as e:
             print(f"Error in test_step: {e}")
             return {'loss': torch.tensor(float('inf'), device=self.device)}

    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        if self.train_metrics: self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        log_freq = getattr(self.cfg.general, 'log_every_n_epochs_train_end', 1)
        if self.current_epoch % log_freq == 0:
             x_ce = to_log.get('train_epoch/x_CE', 0)
             e_ce = to_log.get('train_epoch/E_CE', 0)
             y_ce = to_log.get('train_epoch/y_CE', 0)
             loss_val = x_ce + e_ce + y_ce
             self.print(f"Epoch {self.current_epoch}: Train Loss {loss_val:.3f} ({time.time() - self.start_epoch_time:.1f}s)")
             if self.train_metrics and hasattr(self.train_metrics, 'log_epoch_metrics'):
                 try:
                     epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
                     if epoch_at_metrics or epoch_bond_metrics:
                         self.print(f"  Metrics: {epoch_at_metrics} -- {epoch_bond_metrics}")
                 except Exception as e:
                      print(f"Error logging train metrics: {e}")

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()

    def on_validation_epoch_end(self) -> None:
        # Compute NLL and KL metrics
        try:
            val_nll_value = self.val_nll.compute()
            val_x_kl_value = self.val_X_kl.compute() * self.T
            val_e_kl_value = self.val_E_kl.compute() * self.T
            val_x_logp_value = self.val_X_logp.compute()
            val_e_logp_value = self.val_E_logp.compute()
        except Exception as e:
            print(f"Error computing validation NLL/KL metrics: {e}")
            val_nll_value = torch.tensor(float('inf'))
            val_x_kl_value, val_e_kl_value, val_x_logp_value, val_e_logp_value = 0.0, 0.0, 0.0, 0.0

        # Log NLL and KL metrics
        if wandb.run:
            wandb.log({"val/epoch_NLL": val_nll_value,
                       "val/X_kl": val_x_kl_value,
                       "val/E_kl": val_e_kl_value,
                       "val/X_logp": val_x_logp_value,
                       "val/E_logp": val_e_logp_value}, commit=False) # Commit later with validity

        self.print(f"Epoch {self.current_epoch}: Val NLL {val_nll_value :.2f}")
        self.log("val/epoch_NLL", val_nll_value, sync_dist=True)

        if val_nll_value < self.best_val_nll:
            self.best_val_nll = val_nll_value
        self.print(f'Val NLL: {val_nll_value:.4f} \t Best val NLL: {self.best_val_nll:.4f}')

        # --- Perform Validity Check during Validation ---
        self.val_counter += 1
        # Use separate config flags for validation sampling
        val_samples_to_generate = getattr(self.cfg.general, 'val_samples_to_generate', 64) # Default 64
        sample_every_val = getattr(self.cfg.general, 'sample_every_val', 1) # Default 1

        if self.val_counter % sample_every_val == 0 and val_samples_to_generate > 0:
            self.print("Sampling graphs for validation validity check...")
            start = time.time()
            samples_left_to_generate = val_samples_to_generate
            generated_samples_raw = []
            ident = 0
            batch_size_sample = self.cfg.train.batch_size # Or a specific val batch size

            while samples_left_to_generate > 0:
                # self.print(f'\rGenerating val batch {ident // batch_size_sample + 1}...', end='', flush=True)
                to_generate = min(samples_left_to_generate, batch_size_sample)
                try:
                    batch_samples_raw = self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                          save_final=0, keep_chain=0, number_chain_steps=0)
                    generated_samples_raw.extend(batch_samples_raw)
                except Exception as e:
                     print(f"\nError during validation sampling batch starting at ID {ident}: {e}")
                ident += to_generate
                samples_left_to_generate -= to_generate
            # self.print(f"\nGenerated {len(generated_samples_raw)} raw samples for validation.")

            # --- Convert and Check Validity ---
            generated_nx_graphs = []
            for i, sample_data in enumerate(generated_samples_raw):
                try:
                    nx_graph = self.convert_to_nx(sample_data[0], sample_data[1])
                    if nx_graph is not None: generated_nx_graphs.append(nx_graph)
                except Exception as e: pass # Ignore conversion errors for validation check

            num_generated = len(generated_nx_graphs)
            num_valid = 0
            if num_generated > 0:
                for graph in generated_nx_graphs:
                     try:
                         # Only check validity, not other metrics
                         struct_metrics = calculate_structural_aig_metrics(graph)
                         if struct_metrics.get('is_structurally_valid', 0.0) > 0.5:
                             num_valid += 1
                     except Exception as e:
                          # print(f"Error during validity check: {e}") # Less verbose
                          pass # Ignore errors in validity check for robustness

                validity_fraction = num_valid / num_generated
                self.print(f"Validation Validity Check: {num_valid}/{num_generated} ({validity_fraction*100:.2f}%)")
                if wandb.run:
                    wandb.log({"val/validity": validity_fraction}, commit=True) # Commit validity with NLL
            else:
                 self.print("No graphs generated for validation validity check.")
                 if wandb.run:
                     wandb.log({"val/validity": 0.0}, commit=True) # Log 0 validity


            self.print(f'Validation sampling & validity check done. Took {time.time() - start:.2f} seconds.')


    # --- Test Epoch End (Unchanged - performs full V.U.N.) ---
    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and evaluate using imported functions. """
        self.print("\n--- Test Evaluation ---")
        try:
            test_nll_value = self.test_nll.compute()
            test_x_kl_value = self.test_X_kl.compute() * self.T
            test_e_kl_value = self.test_E_kl.compute() * self.T
            test_x_logp_value = self.test_X_logp.compute()
            test_e_logp_value = self.test_E_logp.compute()
        except Exception as e:
             print(f"Error computing test NLL metrics: {e}")
             test_nll_value = torch.tensor(float('inf'))
             test_x_kl_value, test_e_kl_value, test_x_logp_value, test_e_logp_value = 0.0, 0.0, 0.0, 0.0

        log_dict = {"test/epoch_NLL": test_nll_value, "test/X_kl": test_x_kl_value, "test/E_kl": test_e_kl_value,
                    "test/X_logp": test_x_logp_value, "test/E_logp": test_e_logp_value}
        if wandb.run:
            wandb.log(log_dict, commit=False) # Commit later with VUN

        self.print(f"Test NLL: {test_nll_value:.2f} -- Test KL (X E): {test_x_kl_value:.2f}, {test_e_kl_value:.2f}")

        # --- Generate final samples ---
        self.print("Generating final samples for evaluation...")
        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        generated_samples_raw = []
        ident = 0
        batch_size_sample = self.cfg.train.batch_size # Or a specific sampling batch size config
        while samples_left_to_generate > 0:
            current_batch_num = ident // batch_size_sample + 1
            self.print(f'\rGenerating batch {current_batch_num}...', end='', flush=True)
            to_generate = min(samples_left_to_generate, batch_size_sample)
            try:
                batch_samples_raw = self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                      save_final=0, keep_chain=0, number_chain_steps=0)
                generated_samples_raw.extend(batch_samples_raw)
            except Exception as e:
                 print(f"\nError during sampling batch starting at ID {ident}: {e}")
            ident += to_generate
            samples_left_to_generate -= to_generate
        self.print(f"\nGenerated {len(generated_samples_raw)} raw samples.")

        # --- Convert raw samples to NetworkX graphs ---
        self.print("Converting generated samples to NetworkX graphs...")
        generated_nx_graphs = []
        for i, sample_data in enumerate(tqdm(generated_samples_raw, desc="Converting to NX")):
            try:
                nx_graph = self.convert_to_nx(sample_data[0], sample_data[1])
                if nx_graph is not None: generated_nx_graphs.append(nx_graph)
                else: print(f"Warning: Failed to convert sample {i} to NetworkX graph.")
            except Exception as e: print(f"Error converting sample {i} to NetworkX: {e}")
        self.print(f"Successfully converted {len(generated_nx_graphs)} samples to NetworkX.")

        if not generated_nx_graphs:
            self.print("No valid NetworkX graphs generated. Skipping evaluation.")
            return

        # --- Load Training Data for Novelty ---
        train_graphs = None
        try:
            train_data_dir_for_eval = os.path.join(self.cfg.dataset.datadir, "raw")
            train_data_prefix_for_eval = "real_aigs_part_"
            num_train_files_for_eval = 4
            project_base_path = pathlib.Path(os.getcwd()).parents[2] # Adjust if needed
            abs_train_data_dir = os.path.join(project_base_path, train_data_dir_for_eval)

            if os.path.exists(abs_train_data_dir):
                 self.print(f"Loading training graphs from: {abs_train_data_dir}")
                 train_graphs = load_training_graphs_from_pkl(abs_train_data_dir, train_data_prefix_for_eval, num_train_files_for_eval)
                 if train_graphs is None: self.print("Failed to load training graphs.")
                 elif not train_graphs: self.print("Loaded 0 training graphs.")
                 else: self.print(f"Loaded {len(train_graphs)} training graphs.")
            else:
                 self.print(f"Training data directory not found: {abs_train_data_dir}")
        except Exception as e:
            self.print(f"Error loading training graphs: {e}")

        # --- Perform Direct Evaluation (Full V.U.N.) ---
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
            wandb.log(eval_log, commit=True) # Commit evaluation metrics

        self.print("Test epoch finished.")


    # --- Core Diffusion Logic (Unchanged) ---
    # ... (kl_prior, compute_Lt, reconstruction_logp, apply_noise, compute_val_loss, forward, sample_batch, sample_p_zs_given_zt, compute_extra_data) ...
    # --- Keep these methods exactly as they were in the previous version ---
    def kl_prior(self, X, E, node_mask):
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        probX = X @ Qtb.X
        probE = E @ Qtb.E.unsqueeze(1)
        bs, n, _ = probX.shape
        limit_X = self.limit_dist.X.to(X.device)[None, None, :].expand(bs, n, -1)
        limit_E = self.limit_dist.E.to(E.device)[None, None, None, :].expand(bs, n, n, -1)
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(
            true_X=limit_X.clone(), true_E=limit_E.clone(), pred_X=probX, pred_E=probE, node_mask=node_mask
        )
        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')
        return diffusion_utils.sum_except_batch(kl_distance_X) + diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)
        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        bs, n, d = X.shape
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))
        prob_true_X, prob_true_E, prob_pred_X, prob_pred_E = diffusion_utils.mask_distributions(
            true_X=prob_true.X, true_E=prob_true.E, pred_X=prob_pred.X, pred_E=prob_pred.E, node_mask=node_mask
        )
        # KL divergence expects log probabilities as input
        kl_x = F.kl_div(input=prob_pred_X.log(), target=prob_true_X, reduction='none')
        kl_e = F.kl_div(input=prob_pred_E.log(), target=prob_true_E, reduction='none')

        kl_x_batch_sum = diffusion_utils.sum_except_batch(kl_x)
        kl_e_batch_sum = diffusion_utils.sum_except_batch(kl_e)
        # Update metrics with batch sums
        metric_x_kl = self.test_X_kl if test else self.val_X_kl
        metric_e_kl = self.test_E_kl if test else self.val_E_kl
        metric_x_kl.update(kl_x_batch_sum)
        metric_e_kl.update(kl_e_batch_sum)
        # Return the sum for the current batch (used in NLL calculation)
        return kl_x_batch_sum + kl_e_batch_sum


    def reconstruction_logp(self, t, X, E, node_mask):
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)
        probX0 = X @ Q0.X
        probE0 = E @ Q0.E.unsqueeze(1)
        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)
        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)
        # Ensure masked values don't affect logp calculation by setting to uniform
        probX0[~node_mask] = 1.0 / self.Xdim_output if self.Xdim_output > 0 else 1.0
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = 1.0 / self.Edim_output if self.Edim_output > 0 else 1.0
        diag_mask = torch.eye(probE0.size(1), device=probE0.device, dtype=torch.bool).unsqueeze(0)
        probE0[diag_mask] = 1.0 / self.Edim_output if self.Edim_output > 0 else 1.0
        # Add epsilon to prevent log(0)
        probX0 = probX0 + 1e-7
        probE0 = probE0 + 1e-7
        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()
        s_int = t_int - 1
        t_float = t_int / self.T
        s_float = s_int / self.T
        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)
        probX = X @ Qtb.X
        probE = E @ Qtb.E.unsqueeze(1)
        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)
        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output).float()
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output).float()
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)
        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        t = noisy_data['t']
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)
        kl_prior = self.kl_prior(X, E, node_mask)

        # Compute Lt sum for the batch
        loss_all_t_batch = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # Compute L0 sum for the batch
        metric_x_logp = self.test_X_logp if test else self.val_X_logp
        metric_e_logp = self.test_E_logp if test else self.val_E_logp
        metric_x_logp.reset()
        metric_e_logp.reset()
        prob0 = self.reconstruction_logp(t, X, E, node_mask)
        # Calculate neg log prob for the batch
        neg_log_px = -diffusion_utils.sum_except_batch(X * prob0.X.log())
        neg_log_pe = -diffusion_utils.sum_except_batch(E * prob0.E.log())
        metric_x_logp.update(neg_log_px)
        metric_e_logp.update(neg_log_pe)
        # L0 for the batch is the sum of neg log probs
        loss_term_0_batch = neg_log_px + neg_log_pe

        # NLL for the batch
        nlls_batch = - log_pN + kl_prior + loss_all_t_batch + loss_term_0_batch

        # Update NLL metric object
        nll_metric = self.test_nll if test else self.val_nll
        nll_metric.update(nlls_batch)
        nll = nll_metric.compute() # Get the mean NLL for logging

        # Log batch-level details if needed (potentially very verbose)
        # if wandb.run and not self.training:
        #      wandb.log({"val_test/kl_prior_batch": kl_prior.mean(), # Mean over batch
        #                 "val_test/Lt_batch": loss_all_t_batch.mean(), # Mean over batch
        #                 "val_test/log_pn_batch": log_pN.mean(), # Mean over batch
        #                 "val_test/L0_batch": loss_term_0_batch.mean(), # Mean over batch
        #                 'val_test/batch_nll': nll.item()}, commit=False) # Log final NLL for the batch
        return nll


    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist.to(self.device), node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        sampled_s = utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse=True)
        X_final, E_final = sampled_s.X, sampled_s.E

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X_final[i, :n].cpu()
            edge_types = E_final[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        return molecule_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        pred_X = F.softmax(pred.X, dim=-1)
        pred_E = F.softmax(pred.E, dim=-1)
        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X)
        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E)
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
        unnormalized_prob_X = weighted_X.sum(dim=2)
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)
        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, -1) # Reshape back to original edge shape
        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)
        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        # Symmetrize edge indices before one-hot encoding
        E_s_idx = sampled_s.E
        E_s_idx = torch.triu(E_s_idx, diagonal=1) # Sample upper triangle
        E_s_idx = E_s_idx + E_s_idx.transpose(1, 2) # Symmetrize indices
        E_s = F.one_hot(E_s_idx, num_classes=self.Edim_output).float() # One-hot the symmetrized indices

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros_like(y_t)) # Use zeros_like for y shape
        out_discrete = utils.PlaceHolder(X=sampled_s.X, E=E_s_idx, y=torch.zeros_like(y_t)) # Return symmetrized indices
        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=False).type_as(y_t) # Return both

    def compute_extra_data(self, noisy_data):
        # Simplified: Ensure PlaceHolder is created even if features are None
        if self.extra_features:
             extra_features = self.extra_features(noisy_data)
        else:
             extra_features = utils.PlaceHolder(X=torch.zeros((*noisy_data['X_t'].shape[:-1], 0)).type_as(noisy_data['X_t']), E=torch.zeros((*noisy_data['E_t'].shape[:-1], 0)).type_as(noisy_data['E_t']), y=torch.zeros((noisy_data['y_t'].shape[0], 0)).type_as(noisy_data['y_t']))

        if self.domain_features:
             extra_molecular_features = self.domain_features(noisy_data)
        else:
             extra_molecular_features = utils.PlaceHolder(X=torch.zeros((*noisy_data['X_t'].shape[:-1], 0)).type_as(noisy_data['X_t']), E=torch.zeros((*noisy_data['E_t'].shape[:-1], 0)).type_as(noisy_data['E_t']), y=torch.zeros((noisy_data['y_t'].shape[0], 0)).type_as(noisy_data['y_t']))

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)
        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)
        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        if self.trainer.datamodule: # Check if datamodule exists
             try:
                 self.train_iterations = len(self.trainer.datamodule.train_dataloader())
             except Exception as e:
                  print(f"Warning: Could not determine train_iterations: {e}")
                  self.train_iterations = 1 # Set a default
        else:
             self.train_iterations = 1 # Default if no datamodule
        self.print("Input feature dims (X, E, y):", self.Xdim, self.Edim, self.ydim)
        self.print("Output feature dims (X, E, y):", self.Xdim_output, self.Edim_output, self.ydim_output)

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
