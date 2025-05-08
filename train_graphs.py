import os
import json
import argparse
import torch
import torch.nn as nn # Added for nn.Module access if needed
from torch_geometric.loader import DenseDataLoader
import warnings
import os.path as osp
import numpy as np # Added for np.inf
import pickle # Added for saving generated graphs if needed during validation
import networkx as nx # Added for converting generated graphs
from tqdm import tqdm # Added for progress bars

# --- Assume ggraph and your AIG dataset loader are importable ---
try:
    # Import both the base loader and the wrapper
    from data.aig_dataset import AIGDatasetLoader, AugmentedAIGDataset
    from GraphDF import GraphDF # Assuming these are in the current dir or PYTHONPATH
    from GraphAF import GraphAF
    # Ensure this is the correct import path for your GraphEBM
    from GraphEBM import GraphEBM
    # Import the evaluation function (adjust path if needed)
    # Ensure evaluate_aigs.py is in the python path or same directory
    from evaluate_aigs import calculate_structural_aig_metrics
    # Import config for node/edge types needed for conversion
    # Ensure aig_config.py is accessible
    try:
        import aig_config
    except ImportError:
        # Try relative import if direct fails (common in package structures)
        # This assumes train_graphs.py is in GraphDF/ and aig_config.py is also in GraphDF/
        # Adjust relative path if needed (e.g., from . import aig_config if in same dir)
        import aig_config # If they are in the same directory relative import is not needed
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'ggraph' (including dig.ggraph) is installed, your AIGDatasetLoader/AugmentedAIGDataset classes are accessible, "
          "GraphDF/GraphAF/GraphEBM models are correctly placed, evaluate_aigs.py and aig_config.py are importable.")
    exit(1)
# --- End Imports ---

# --- Constants ---
AIG_NODE_TYPE_KEYS = getattr(aig_config, 'NODE_TYPE_KEYS', ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"])
AIG_EDGE_TYPE_KEYS = getattr(aig_config, 'EDGE_TYPE_KEYS', ["EDGE_REG", "EDGE_INV"])
# --- End Constants ---


# --- Base Configuration Dictionary (Keep as before) ---
base_conf = {
    "data_name": "aig",
    "model": {
        "max_size": 64,
        "node_dim": 4,
        "bond_dim": 3,
        "use_gpu": True,
        "edge_unroll": 12,
        "num_flow_layer": 12,
        "num_rgcn_layer": 3,
        "nhid": 128,
        "nout": 128,
        "deq_coeff": 0.9,
        "st_type": "exp",
        "use_df": False
    },
    "model_ebm": {
        "hidden": 64,
        "depth": 2,
        "swish_act": True,
        "add_self": False,
        "dropout": 0.0,
        "n_power_iterations": 1
    },
    "lr": 0.001,
    "weight_decay": 0,
    "batch_size": 128, # Default, overridden by args
    "max_epochs": 50,  # Default, overridden by args
    "save_interval": 3, # Default, overridden by args
    "grad_clip_value": None,
    "train_ebm": {
        "c": 0.0,
        "ld_step": 150,
        "ld_noise": 0.005,
        "ld_step_size": 30,
        "clamp_lgd_grad": True,
        "alpha": 1.0
    }
}
# --- End Base Configuration ---

# --- Helper function to convert raw generated data to NetworkX ---
# (Copied from GraphDF/graphdf.py - ensure consistency)
def _convert_raw_to_aig_digraph(node_features_one_hot, typed_edges_list,
                                num_actual_nodes, aig_node_type_strings, aig_edge_type_strings):
    """ Converts raw discrete model output to a NetworkX DiGraph for AIGs. """
    graph = nx.DiGraph()
    # Ensure tensor is on CPU for numpy conversion or direct processing
    node_features = node_features_one_hot.cpu().detach()
    for i in range(num_actual_nodes):
        try:
            # Check if the row sum is zero (should not happen with valid one-hot)
            if node_features[i].sum() == 0:
                node_type_label = "UNKNOWN_NODE_TYPE"
                warnings.warn(f"Node {i} has all-zero feature vector during conversion.")
            else:
                # Get the index of the max value (which is 1 in one-hot)
                node_type_idx = torch.argmax(node_features[i]).item()
                # Map index to string label using the provided list
                node_type_label = aig_node_type_strings[node_type_idx] if 0 <= node_type_idx < len(aig_node_type_strings) else "UNKNOWN_NODE_TYPE"
            # Add node with its string type label
            graph.add_node(i, type=node_type_label)
        except Exception as e:
            warnings.warn(f"Error processing node {i} during conversion: {e}")
            return None # Indicate failure if node processing fails

    # Add edges using the provided list of (u, v, type_idx)
    for u, v, edge_type_idx in typed_edges_list:
        # Ensure nodes exist within the actual graph size
        if u < num_actual_nodes and v < num_actual_nodes:
            # Map edge type index (0 or 1) to string label
            if 0 <= edge_type_idx < len(aig_edge_type_strings):
                edge_type_label = aig_edge_type_strings[edge_type_idx]
                graph.add_edge(u, v, type=edge_type_label)
            else:
                # Handle unexpected edge type indices
                warnings.warn(f"Edge ({u}->{v}) has unexpected edge_type_idx {edge_type_idx}. Adding untyped.")
                graph.add_edge(u, v) # Add edge without type attribute
        # else: # This warning can be very noisy during generation if termination happens early
             # warnings.warn(f"Skipping edge ({u}->{v}) due to node index out of range ({num_actual_nodes}).")
    return graph
# --- End Helper ---


def run_validation(model, val_loader, device, model_type, ebm_conf=None):
    """ Calculates loss on the validation set. """
    model.eval() # Set model to evaluation mode
    total_val_loss = 0
    num_batches = 0
    if val_loader is None:
        print("  Validation loader not available. Skipping validation loss calculation.")
        return float('inf') # Return infinity if no validation loader

    print("  Running validation loss calculation...")
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc="  Validation", leave=False)
        for data_batch in pbar_val:
            inp_node_features = data_batch.x.to(device)
            inp_adj_features = data_batch.adj.to(device)

            # Forward pass and loss calculation
            loss = 0
            try:
                if model_type == 'GraphDF' or model_type == 'GraphAF':
                    # Ensure the model has the dis_log_prob method
                    if hasattr(model, 'dis_log_prob'):
                        out_z = model(inp_node_features, inp_adj_features)
                        loss = model.dis_log_prob(out_z)
                    else:
                        warnings.warn(f"{model_type} model missing 'dis_log_prob' method. Cannot calculate validation loss.")
                        return float('inf')
                elif model_type == 'GraphEBM':
                    # Ensure the EBM model itself (not the runner) has calculate_loss
                    if hasattr(model, 'calculate_loss') and ebm_conf:
                         loss = model.calculate_loss(inp_node_features, inp_adj_features, ebm_conf)
                    else:
                         warnings.warn("GraphEBM model missing 'calculate_loss' method or ebm_conf not provided. Skipping validation loss.")
                         return float('inf')
                else:
                     warnings.warn(f"Validation loss calculation not defined for model type {model_type}.")
                     return float('inf')

                if not torch.isnan(loss) and not torch.isinf(loss):
                     total_val_loss += loss.item()
                     num_batches += 1
                else:
                     warnings.warn(f"NaN or Inf validation loss detected in batch {num_batches}. Skipping batch.")
                pbar_val.set_postfix({'Loss': loss.item() if not torch.isnan(loss) and not torch.isinf(loss) else 'NaN/Inf'})
            except Exception as val_e:
                 warnings.warn(f"Error during validation batch {num_batches}: {val_e}. Skipping batch.")
                 continue # Skip to next batch

    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
    return avg_val_loss

def run_generation_and_eval(runner, gen_args, num_graphs_to_gen=50):
    """ Generates a small batch of graphs and evaluates their structural validity. """
    print(f"  Generating {num_graphs_to_gen} graphs for validation check...")
    if runner.model is None:
        warnings.warn("Runner's model is None. Cannot perform generation check.")
        return 0.0

    runner.model.eval() # Ensure model is in eval mode

    # We need direct access to the model's generation method.
    if not hasattr(runner.model, 'generate_aig_discrete_raw_data'):
        warnings.warn("Model does not have 'generate_aig_discrete_raw_data'. Cannot perform generation check.")
        return 0.0 # Return 0 validity

    generated_nxg = []
    # Ensure necessary keys are present in gen_args
    model_conf = gen_args.get('model_conf_dict', {})
    max_nodes = model_conf.get('max_size', 64) # Default if missing
    temperature = gen_args.get('temperature', [0.6, 0.6]) # Default if missing
    temp_node, temp_edge = temperature[0], temperature[1]
    min_nodes_gen = gen_args.get('num_min_nodes', 0)

    try:
        device = next(runner.model.parameters()).device # Get device from model
    except StopIteration:
        device = torch.device("cuda" if model_conf.get('use_gpu') and torch.cuda.is_available() else "cpu")
        warnings.warn(f"Model has no parameters. Assuming device {device} for generation check.")

    graphs_attempted = 0
    # Increase max attempts relative to requested graphs
    max_gen_attempts = num_graphs_to_gen * 10 + 10 # Try harder

    pbar_gen = tqdm(total=num_graphs_to_gen, desc="  Generating for Eval", leave=False)
    while len(generated_nxg) < num_graphs_to_gen and graphs_attempted < max_gen_attempts:
        graphs_attempted += 1
        try:
            raw_nodes, raw_edges, num_actual = runner.model.generate_aig_discrete_raw_data(
                max_nodes=max_nodes,
                temperature_node=temp_node,
                temperature_edge=temp_edge,
                device=device
                # disconnection_patience uses default from the method
            )
            # Convert raw output to NetworkX
            nxg = _convert_raw_to_aig_digraph(
                raw_nodes, raw_edges, num_actual,
                AIG_NODE_TYPE_KEYS, AIG_EDGE_TYPE_KEYS
            )
            # Check min nodes *after* conversion
            if nxg is not None and nxg.number_of_nodes() >= min_nodes_gen:
                 generated_nxg.append(nxg)
                 pbar_gen.update(1) # Update progress bar when a graph is successfully generated and meets criteria

        except Exception as e:
            warnings.warn(f"Error during validation generation attempt {graphs_attempted}: {e}")
            # Continue trying to generate the requested number
    pbar_gen.close()

    # --- Evaluate Validity ---
    num_generated = len(generated_nxg)
    num_valid = 0
    if num_generated > 0:
        print(f"  Evaluating validity of {num_generated} generated graphs...")
        for g in tqdm(generated_nxg, desc="  Evaluating Validity", leave=False):
            try:
                metrics = calculate_structural_aig_metrics(g)
                if metrics.get('is_structurally_valid', False):
                    num_valid += 1
            except Exception as eval_e:
                warnings.warn(f"Error evaluating generated graph during validation: {eval_e}")

        validity_percent = (num_valid / num_generated) * 100 if num_generated > 0 else 0.0
        print(f"  Validation Generation: {num_valid}/{num_generated} ({validity_percent:.1f}%) structurally valid.")
        return validity_percent / 100.0 # Return fraction
    else:
        print(f"  Validation Generation: No valid graphs generated meeting criteria ({min_nodes_gen} nodes) after {graphs_attempted} attempts.")
        return 0.0


def main(args):
    conf = base_conf.copy()
    # Deep copy nested dictionaries
    conf['model'] = base_conf['model'].copy()
    conf['model_ebm'] = base_conf['model_ebm'].copy()
    conf['train_ebm'] = base_conf['train_ebm'].copy()

    # Update conf from args (General, GAF/DF, EBM - same as before)
    conf['lr'] = args.lr
    conf['weight_decay'] = args.weight_decay
    conf['batch_size'] = args.batch_size
    conf['max_epochs'] = args.max_epochs
    conf['save_interval'] = args.save_interval
    conf['grad_clip_value'] = args.grad_clip_value
    conf['model']['edge_unroll'] = args.edge_unroll
    conf['model']['num_flow_layer'] = args.num_flow_layer
    conf['model']['num_rgcn_layer'] = args.num_rgcn_layer
    conf['model']['nhid'] = args.gaf_nhid
    conf['model']['nout'] = args.gaf_nout
    conf['model']['deq_coeff'] = args.deq_coeff
    conf['model']['st_type'] = args.st_type
    if args.model_type == 'GraphDF': conf['model']['use_df'] = True
    conf['model_ebm']['hidden'] = args.ebm_hidden
    conf['model_ebm']['depth'] = args.ebm_depth
    conf['model_ebm']['swish_act'] = args.ebm_swish_act
    conf['model_ebm']['add_self'] = args.ebm_add_self
    conf['model_ebm']['dropout'] = args.ebm_dropout
    conf['model_ebm']['n_power_iterations'] = args.ebm_n_power_iterations
    conf['train_ebm']['c'] = args.ebm_c
    conf['train_ebm']['ld_step'] = args.ebm_ld_step
    conf['train_ebm']['ld_noise'] = args.ebm_ld_noise
    conf['train_ebm']['ld_step_size'] = args.ebm_ld_step_size
    conf['train_ebm']['alpha'] = args.ebm_alpha
    conf['train_ebm']['clamp_lgd_grad'] = args.ebm_clamp_lgd_grad

    # Determine device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    else:
        device = torch.device('cpu')
        print("Using CPU.")
    conf['model']['use_gpu'] = (device.type == 'cuda')

    # --- Dataset Loading (Train and Validation) ---
    print("Instantiating AIGDatasetLoader for Training...")
    try:
        base_train_dataset = AIGDatasetLoader(root=args.data_root, name=conf.get('data_name', 'aig'), dataset_type="train")
        train_dataset = AugmentedAIGDataset(base_dataset=base_train_dataset, num_augmentations=args.num_augmentations)
        print(f"Total training samples including augmentations: {len(train_dataset)}")
        if len(train_dataset) == 0: raise ValueError("Training dataset is empty.")
    except Exception as e: print(f"Error loading training dataset: {e}"); exit(1)

    print("Instantiating AIGDatasetLoader for Validation...")
    try:
        # Load validation data without augmentation wrapper
        val_dataset = AIGDatasetLoader(root=args.data_root, name=conf.get('data_name', 'aig'), dataset_type="val") # Use "val" split
        print(f"Validation dataset loaded with {len(val_dataset)} graphs.")
        if len(val_dataset) == 0: warnings.warn("Validation dataset is empty.")
    except FileNotFoundError:
        print(f"Validation data file not found at expected location (e.g., {osp.join(args.data_root, conf.get('data_name', 'aig'), 'processed', 'val', 'data.pt')}). Validation will be skipped.")
        val_dataset = None
    except Exception as e:
        print(f"Error loading validation dataset: {e}")
        val_dataset = None # Set to None if loading fails

    # DataLoaders
    # Use drop_last=True for training to ensure consistent batch sizes, especially if using BatchNorm
    train_loader = DenseDataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=True, drop_last=True)
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_batch_size = conf['batch_size'] # Use same batch size for validation
        # No drop_last for validation, process all samples
        val_loader = DenseDataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        print(f"Created Validation DataLoader with batch size {val_batch_size}.")
    else:
        print("Validation DataLoader not created (validation dataset missing, empty, or failed to load).")


    # --- Model Instantiation ---
    print(f"Instantiating model runner: {args.model_type}")
    runner = None
    # (Model instantiation logic remains the same as before)
    if args.model_type == 'GraphDF': runner = GraphDF()
    elif args.model_type == 'GraphAF': runner = GraphAF()
    elif args.model_type == 'GraphEBM':
        try:
            runner = GraphEBM(n_atom=conf['model']['max_size'], n_atom_type=conf['model']['node_dim'], n_edge_type=conf['model']['bond_dim'], hidden=conf['model_ebm']['hidden'], depth=conf['model_ebm']['depth'], swish_act=conf['model_ebm']['swish_act'], add_self=conf['model_ebm']['add_self'], dropout=conf['model_ebm']['dropout'], n_power_iterations=conf['model_ebm']['n_power_iterations'], device=device)
        except Exception as e: print(f"Error instantiating GraphEBM: {e}"); exit(1)
    else: print(f"Error: Unknown model type '{args.model_type}'."); exit(1)
    if runner is None: print(f"Failed to instantiate model runner for {args.model_type}"); exit(1)

    # --- Initialize the actual model within the runner ---
    # Pass model_conf_dict which now contains updated hyperparams from args
    runner.get_model(f'rand_gen_{args.model_type}', conf['model'])
    if runner.model is None: print(f"Runner failed to initialize internal model for {args.model_type}"); exit(1)
    runner.model.to(device) # Move model to device

    # --- Determine Save Directory ---
    # (Save directory logic remains the same)
    if args.save_dir: save_dir = args.save_dir
    else:
        default_save_dir_base = "outputs" # Changed base dir name
        model_specific_path = f"{args.model_type}/rand_gen_{conf.get('data_name', 'aig')}_ckpts"
        save_dir = osp.join(default_save_dir_base, model_specific_path)
    conf['save_dir'] = save_dir # Store final path in conf
    os.makedirs(conf['save_dir'], exist_ok=True)
    print(f"Model checkpoints will be saved in: {conf['save_dir']}")

    # --- Training Setup ---
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, runner.model.parameters()), lr=conf['lr'], weight_decay=conf['weight_decay'])
    # --- LR Scheduler ---
    scheduler = None
    if val_loader is not None and args.use_lr_scheduler:
         # Reduce LR when validation loss plateaus
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_scheduler_patience, verbose=True)
         print(f"Enabled ReduceLROnPlateau LR scheduler (patience={args.lr_scheduler_patience}).")
    elif args.use_lr_scheduler:
         warnings.warn("LR scheduler requested but no validation loader available. Scheduler disabled.")
    # --- End LR Scheduler ---

    best_val_loss = np.inf
    best_epoch = -1
    train_losses = []
    val_losses = []
    gen_validities = [] # Store validity fraction from generation checks

    print(f"--- Starting Training ---")
    # --- Main Training Loop ---
    for epoch in range(1, conf['max_epochs'] + 1):
        runner.model.train() # Set model to training mode
        epoch_train_loss = 0
        processed_batches = 0
        # Use tqdm for training loop progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{conf['max_epochs']} (Train)", leave=False)

        # --- Training Epoch ---
        for data_batch in pbar:
            optimizer.zero_grad()
            inp_node_features = data_batch.x.to(device)
            inp_adj_features = data_batch.adj.to(device)
            loss = 0
            try:
                # Forward pass and loss calculation (specific to model type)
                if args.model_type == 'GraphDF' or args.model_type == 'GraphAF':
                    if hasattr(runner.model, 'dis_log_prob'):
                        out_z = runner.model(inp_node_features, inp_adj_features)
                        loss = runner.model.dis_log_prob(out_z)
                    else: raise NotImplementedError(f"{args.model_type} model missing 'dis_log_prob'")
                elif args.model_type == 'GraphEBM':
                     if hasattr(runner, 'calculate_loss'): loss = runner.calculate_loss(inp_node_features, inp_adj_features, conf['train_ebm'])
                     elif hasattr(runner.model, 'calculate_loss'): loss = runner.model.calculate_loss(inp_node_features, inp_adj_features, conf['train_ebm'])
                     else: raise NotImplementedError("EBM loss calculation method not found.")
                else: raise NotImplementedError(f"Loss calculation not defined for model type {args.model_type}.")

                # Check for invalid loss before backward pass
                if torch.isnan(loss) or torch.isinf(loss):
                     warnings.warn(f"NaN or Inf loss detected during training epoch {epoch}, batch {processed_batches}. Skipping update.")
                     continue # Skip optimizer step if loss is invalid

                loss.backward()
                # Apply gradient clipping if specified
                if conf['grad_clip_value'] is not None and conf['grad_clip_value'] > 0:
                    torch.nn.utils.clip_grad_norm_(runner.model.parameters(), conf['grad_clip_value'])
                optimizer.step()

                epoch_train_loss += loss.item()
                processed_batches += 1
                # Update progress bar postfix with current batch loss
                pbar.set_postfix({'Loss': loss.item()})

            except Exception as train_e:
                 warnings.warn(f"Error during training batch {processed_batches} (epoch {epoch}): {train_e}. Skipping batch.")
                 optimizer.zero_grad() # Clear potentially bad gradients
                 continue # Skip to next batch
        pbar.close() # Close training progress bar for the epoch

        avg_train_loss = epoch_train_loss / processed_batches if processed_batches > 0 else float('nan')
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch}/{conf['max_epochs']} | Avg Train Loss: {avg_train_loss:.4f}")

        # --- Validation & Checkpointing Step (Run every save_interval epochs) ---
        if epoch % conf['save_interval'] == 0:
            avg_val_loss = run_validation(runner.model, val_loader, device, args.model_type, conf.get('train_ebm'))
            val_losses.append(avg_val_loss) # Store validation loss
            print(f"Epoch {epoch}/{conf['max_epochs']} | Avg Valid Loss: {avg_val_loss:.4f}")

            current_validity = -1.0 # Default if not run
            if args.validate_generation:
                 gen_args_val = {
                     "model_conf_dict": conf['model'],
                     "temperature": args.temperature_df if args.model_type == 'GraphDF' else [0.7, 0.7], # Pass appropriate temp
                     "num_min_nodes": args.gen_min_nodes # Use the new arg
                 }
                 current_validity = run_generation_and_eval(runner, gen_args_val, num_graphs_to_gen=args.num_val_gen_samples)
                 gen_validities.append(current_validity) # Store validity
                 print(f"Epoch {epoch}/{conf['max_epochs']} | Generation Validity Check: {current_validity*100:.1f}%")

            # Save checkpoint (always save at interval)
            ckpt_path = osp.join(conf['save_dir'], f'{args.model_type.lower()}_ckpt_epoch_{epoch}.pth')
            model_state_to_save = runner.model.module.state_dict() if isinstance(runner.model, nn.DataParallel) else runner.model.state_dict()
            # Include optimizer state and epoch info in checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss, # Store validation loss in checkpoint
                'train_loss': avg_train_loss,
                'gen_validity': current_validity # Store validity if calculated
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            # Save best model based on validation loss (only if validation was possible)
            if avg_val_loss != float('inf') and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_ckpt_path = osp.join(conf['save_dir'], f'{args.model_type.lower()}_ckpt_best_val.pth')
                torch.save(model_state_to_save, best_ckpt_path) # Save only model state for best
                print(f"*** New best validation loss ({best_val_loss:.4f}). Saved best model to {best_ckpt_path} ***")

            # --- LR Scheduler Step ---
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss) # Step based on validation loss
            elif scheduler:
                scheduler.step() # Step based on epoch for other schedulers
            # --- End LR Scheduler Step ---

    print("--- Training Finished ---")
    if best_epoch != -1:
        print(f"Best validation loss ({best_val_loss:.4f}) achieved at epoch {best_epoch}.")
    else:
        print("Validation was not performed or no improvement detected.")

    # --- Save Training History ---
    # Ensure lists have consistent lengths if validation/gen checks didn't run every epoch
    num_epochs_run = len(train_losses)
    num_val_epochs = len(val_losses)
    num_gen_epochs = len(gen_validities)
    # Pad validation/generation history if needed (e.g., fill with None or NaN)
    padded_val_losses = ([None] * (num_epochs_run - num_val_epochs)) + val_losses if num_val_epochs < num_epochs_run else val_losses
    padded_gen_validity = ([None] * (num_epochs_run - num_gen_epochs)) + gen_validities if num_gen_epochs < num_epochs_run else gen_validities

    history = {
        'train_loss': train_losses,
        'val_loss': padded_val_losses, # Use padded list
        'gen_validity': padded_gen_validity, # Use padded list
        'best_val_epoch': best_epoch,
        'best_val_loss': best_val_loss if best_epoch != -1 else None
    }
    history_path = osp.join(conf['save_dir'], f'{args.model_type.lower()}_training_history.json')
    try:
        with open(history_path, 'w') as f:
            # Handle potential NaN values for JSON serialization
            json.dump(history, f, indent=4, default=lambda x: None if np.isnan(x) else x)
        print(f"Saved training history to {history_path}")
    except Exception as e:
        print(f"Error saving training history: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train graph generation models on pre-processed AIG dataset.")

    # --- General Arguments ---
    parser.add_argument('--model_type', type=str, default='GraphDF', choices=['GraphDF', 'GraphAF', 'GraphEBM'], help='Model to train.')
    parser.add_argument('--data_root', default="./data/", help="Root directory for data (e.g., contains 'aig/processed/train/data.pt').")
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device for training.')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save model checkpoints. Overrides internal construction.')
    parser.add_argument('--num_augmentations', type=int, default=5, help='Number of randomized topological sort augmentations per graph during training.')

    # --- General Training Hyperparameters ---
    parser.add_argument('--lr', type=float, default=base_conf['lr'])
    parser.add_argument('--weight_decay', type=float, default=base_conf['weight_decay'])
    parser.add_argument('--batch_size', type=int, default=base_conf['batch_size'])
    parser.add_argument('--max_epochs', type=int, default=base_conf['max_epochs'])
    parser.add_argument('--save_interval', type=int, default=base_conf['save_interval'], help="Save checkpoints and run validation every N epochs.")
    parser.add_argument('--grad_clip_value', type=float, default=base_conf['grad_clip_value'], help='Value for gradient norm clipping. None or 0 to disable.')
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Enable ReduceLROnPlateau learning rate scheduler based on validation loss.')
    parser.add_argument('--lr_scheduler_patience', type=int, default=3, help='Patience for ReduceLROnPlateau scheduler.') # Added patience arg

    # --- Validation & Generation Check Arguments ---
    parser.add_argument('--validate_generation', action='store_true', help='If set, generate and evaluate a few graphs during validation steps.')
    parser.add_argument('--num_val_gen_samples', type=int, default=50, help='Number of graphs to generate for validity check during validation.')
    parser.add_argument('--gen_min_nodes', type=int, default=5, help='Minimum nodes for graphs generated during validation check.') # Added min nodes for gen check
    # Add temperature args needed for validation generation (might reuse training args or have separate ones)
    parser.add_argument('--temperature_df', type=float, nargs=2, default=[0.6, 0.6], help='Temperature [node, edge] (for GraphDF generation check).')
    # Add other model-specific generation args if needed for validation check (e.g., --temperature_af)

    # --- GraphAF/GraphDF Specific Model Hyperparameters ---
    parser.add_argument('--edge_unroll', type=int, default=base_conf['model']['edge_unroll'])
    parser.add_argument('--num_flow_layer', type=int, default=base_conf['model']['num_flow_layer'])
    parser.add_argument('--num_rgcn_layer', type=int, default=base_conf['model']['num_rgcn_layer'])
    parser.add_argument('--gaf_nhid', type=int, default=base_conf['model']['nhid'], help="Hidden dim for GAF/GDF's RGCN/ST-nets.")
    parser.add_argument('--gaf_nout', type=int, default=base_conf['model']['nout'], help="Output dim for GAF/GDF's RGCN (embedding size).")
    parser.add_argument('--deq_coeff', type=float, default=base_conf['model']['deq_coeff'], help="Dequantization coefficient if used.")
    parser.add_argument('--st_type', type=str, default=base_conf['model']['st_type'], choices=['exp', 'sigmoid', 'softplus'], help="Type of ST network for GAF/GDF.")

    # --- GraphEBM Model Hyperparameters ---
    parser.add_argument('--ebm_hidden', type=int, default=base_conf['model_ebm']['hidden'])
    parser.add_argument('--ebm_depth', type=int, default=base_conf['model_ebm']['depth'])
    parser.add_argument('--ebm_swish_act', action=argparse.BooleanOptionalAction, default=base_conf['model_ebm']['swish_act'])
    parser.add_argument('--ebm_add_self', action=argparse.BooleanOptionalAction, default=base_conf['model_ebm']['add_self'])
    parser.add_argument('--ebm_dropout', type=float, default=base_conf['model_ebm']['dropout'])
    parser.add_argument('--ebm_n_power_iterations', type=int, default=base_conf['model_ebm']['n_power_iterations'])

    # --- GraphEBM Training Hyperparameters ---
    parser.add_argument('--ebm_c', type=float, default=base_conf['train_ebm']['c'])
    parser.add_argument('--ebm_ld_step', type=int, default=base_conf['train_ebm']['ld_step'])
    parser.add_argument('--ebm_ld_noise', type=float, default=base_conf['train_ebm']['ld_noise'])
    parser.add_argument('--ebm_ld_step_size', type=float, default=base_conf['train_ebm']['ld_step_size'])
    parser.add_argument('--ebm_alpha', type=float, default=base_conf['train_ebm']['alpha'])
    parser.add_argument('--ebm_clamp_lgd_grad', action=argparse.BooleanOptionalAction, default=base_conf['train_ebm']['clamp_lgd_grad'])

    args = parser.parse_args()
    main(args)