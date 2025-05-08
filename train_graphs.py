import os
import json
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DenseDataLoader
import warnings
import os.path as osp
import numpy as np
import pickle
import networkx as nx
from tqdm import tqdm
import sys

# --- Dataset and Model Imports ---
try:
    # *** Update this import path if needed ***
    from data.aig_dataset import AIGProcessedAugmentedDataset

    from GraphDF import GraphDF
    from GraphAF import GraphAF
    from GraphEBM import GraphEBM
    # Ensure evaluate_aigs is importable
    try:
        from evaluate_aigs import calculate_structural_aig_metrics
    except ImportError:
        print("Warning: evaluate_aigs.py not found or cannot be imported. Generation validation will be skipped.")
        calculate_structural_aig_metrics = None # Define as None to handle gracefully

    try:
        import data.aig_config as aig_config
    except ImportError:
        import G2PT.configs.aig as aig_config
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure dataset class, models, evaluate_aigs.py, and aig_config.py are accessible.")
    exit(1)
# --- End Imports ---

# --- Constants ---
if aig_config:
    AIG_NODE_TYPE_KEYS = getattr(aig_config, 'NODE_TYPE_KEYS', ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"])
    AIG_EDGE_TYPE_KEYS = getattr(aig_config, 'EDGE_TYPE_KEYS', ["EDGE_REG", "EDGE_INV"])
else:
    AIG_NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
    AIG_EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]
# --- End Constants ---


# --- Base Configuration Dictionary (Updated Defaults) ---
base_conf = {
    "data_name": "aig_augmented_ds",
    "model": {
        "max_size": 64,
        "node_dim": 4,
        "bond_dim": 3,
        "use_gpu": True,
        "edge_unroll": 15,     # *** Suggestion: Update based on analysis ***
        "num_flow_layer": 12,  # Tunable
        "num_rgcn_layer": 3,   # Tunable
        "nhid": 128,           # Tunable
        "nout": 128,           # Tunable
        "deq_coeff": 0.9,
        "st_type": "exp",
        "use_df": False
    },
    "model_ebm": { # Keep EBM params separate
        "hidden": 64, "depth": 2, "swish_act": True, "add_self": False,
        "dropout": 0.0, "n_power_iterations": 1
    },
    "lr": 0.0005,              # *** Suggestion: Lower default LR ***
    "weight_decay": 1e-5,      # *** Suggestion: Add small weight decay ***
    "batch_size": 64,          # *** Suggestion: Try smaller batch size ***
    "max_epochs": 50,          # Keep or adjust based on convergence
    "save_interval": 5,        # *** Suggestion: Save/Validate more often? ***
    "grad_clip_value": 1.0,    # *** Suggestion: Enable gradient clipping ***
    "train_ebm": { # Keep EBM params separate
        "c": 0.0, "ld_step": 150, "ld_noise": 0.005, "ld_step_size": 30,
        "clamp_lgd_grad": True, "alpha": 1.0
    }
}
# --- End Base Configuration ---

# --- Helper function to convert raw generated data to NetworkX ---
def _convert_raw_to_aig_digraph(node_features_one_hot, typed_edges_list,
                                num_actual_nodes, aig_node_type_strings, aig_edge_type_strings):
    """ Converts raw discrete model output to a NetworkX DiGraph for AIGs. """
    # (Keep this function exactly as before)
    graph = nx.DiGraph()
    node_features = node_features_one_hot.cpu().detach()
    for i in range(num_actual_nodes):
        try:
            if node_features[i].sum() == 0:
                node_type_label = "UNKNOWN_NODE_TYPE"
                warnings.warn(f"Node {i} has all-zero feature vector during conversion.")
            else:
                node_type_idx = torch.argmax(node_features[i]).item()
                node_type_label = aig_node_type_strings[node_type_idx] if 0 <= node_type_idx < len(aig_node_type_strings) else "UNKNOWN_NODE_TYPE"
            graph.add_node(i, type=node_type_label)
        except Exception as e:
            warnings.warn(f"Error processing node {i} during conversion: {e}")
            return None

    for u, v, edge_type_idx in typed_edges_list:
        if u < num_actual_nodes and v < num_actual_nodes:
            if 0 <= edge_type_idx < len(aig_edge_type_strings):
                edge_type_label = aig_edge_type_strings[edge_type_idx]
                graph.add_edge(u, v, type=edge_type_label)
            else:
                warnings.warn(f"Edge ({u}->{v}) has unexpected edge_type_idx {edge_type_idx}. Adding untyped.")
                graph.add_edge(u, v)
    return graph
# --- End Helper ---


def run_validation(model, val_loader, device, model_type, ebm_conf=None):
    """ Calculates loss on the validation set. """
    # (Keep this function exactly as before)
    model.eval()
    total_val_loss = 0
    num_batches = 0
    if val_loader is None: return float('inf')
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc="  Validation", leave=False)
        for data_batch in pbar_val:
            inp_node_features = data_batch.x.to(device)
            inp_adj_features = data_batch.adj.to(device)
            loss = 0
            try:
                if model_type == 'GraphDF' or model_type == 'GraphAF':
                    if hasattr(model, 'dis_log_prob'):
                        out_z = model(inp_node_features, inp_adj_features)
                        loss = model.dis_log_prob(out_z)
                    else: return float('inf') # Cannot calculate
                elif model_type == 'GraphEBM':
                    if hasattr(model, 'calculate_loss') and ebm_conf:
                         loss = model.calculate_loss(inp_node_features, inp_adj_features, ebm_conf)
                    else: return float('inf') # Cannot calculate
                else: return float('inf') # Unknown model

                if not torch.isnan(loss) and not torch.isinf(loss):
                     total_val_loss += loss.item()
                     num_batches += 1
                else: warnings.warn(f"NaN/Inf validation loss")
                pbar_val.set_postfix({'Loss': loss.item() if not torch.isnan(loss) and not torch.isinf(loss) else 'NaN/Inf'})
            except Exception as val_e:
                 warnings.warn(f"Error during validation batch: {val_e}")
                 continue
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
    return avg_val_loss

def run_generation_and_eval(runner, gen_args, num_graphs_to_gen=50):
    """ Generates graphs and evaluates structural validity. """
    # (Keep this function exactly as before, but ensure calculate_structural_aig_metrics is available)
    if calculate_structural_aig_metrics is None:
        print("Skipping generation validation as evaluate_aigs could not be imported.")
        return -1.0 # Indicate skip/error

    print(f"  Generating {num_graphs_to_gen} graphs for validation check...")
    if runner.model is None: return 0.0
    runner.model.eval()
    if not hasattr(runner.model, 'generate_aig_discrete_raw_data'): return 0.0

    generated_nxg = []
    model_conf = gen_args.get('model_conf_dict', {})
    max_nodes = model_conf.get('max_size', 64)
    temperature = gen_args.get('temperature', [0.6, 0.6])
    temp_node, temp_edge = temperature[0], temperature[1]
    min_nodes_gen = gen_args.get('num_min_nodes', 0)

    try: device = next(runner.model.parameters()).device
    except StopIteration: device = torch.device("cuda" if model_conf.get('use_gpu') and torch.cuda.is_available() else "cpu")

    graphs_attempted, max_gen_attempts = 0, num_graphs_to_gen * 10 + 10
    pbar_gen = tqdm(total=num_graphs_to_gen, desc="  Generating for Eval", leave=False)
    while len(generated_nxg) < num_graphs_to_gen and graphs_attempted < max_gen_attempts:
        graphs_attempted += 1
        try:
            raw_nodes, raw_edges, num_actual = runner.model.generate_aig_discrete_raw_data(
                max_nodes=max_nodes, temperature_node=temp_node, temperature_edge=temp_edge, device=device)
            nxg = _convert_raw_to_aig_digraph(raw_nodes, raw_edges, num_actual, AIG_NODE_TYPE_KEYS, AIG_EDGE_TYPE_KEYS)
            if nxg is not None and nxg.number_of_nodes() >= min_nodes_gen:
                 generated_nxg.append(nxg); pbar_gen.update(1)
        except Exception as e: warnings.warn(f"Gen attempt {graphs_attempted} error: {e}")
    pbar_gen.close()

    num_generated = len(generated_nxg)
    num_valid = 0
    if num_generated > 0:
        print(f"  Evaluating validity of {num_generated} generated graphs...")
        for g in tqdm(generated_nxg, desc="  Evaluating Validity", leave=False):
            try:
                metrics = calculate_structural_aig_metrics(g)
                if metrics.get('is_structurally_valid', False): num_valid += 1
            except Exception as eval_e: warnings.warn(f"Eval error: {eval_e}")
        validity_percent = (num_valid / num_generated) * 100
        print(f"  Validation Generation: {num_valid}/{num_generated} ({validity_percent:.1f}%) structurally valid.")
        return validity_percent / 100.0
    else:
        print(f"  Validation Generation: No valid graphs generated meeting criteria ({min_nodes_gen} nodes) after {graphs_attempted} attempts.")
        return 0.0


def main(args):
    conf = base_conf.copy()
    conf['model'] = base_conf['model'].copy()
    conf['model_ebm'] = base_conf['model_ebm'].copy()
    conf['train_ebm'] = base_conf['train_ebm'].copy()

    # Update conf from args - Use getattr to safely access args, falling back to conf default
    conf['lr'] = getattr(args, 'lr', conf['lr'])
    conf['weight_decay'] = getattr(args, 'weight_decay', conf['weight_decay'])
    conf['batch_size'] = getattr(args, 'batch_size', conf['batch_size'])
    conf['max_epochs'] = getattr(args, 'max_epochs', conf['max_epochs'])
    conf['save_interval'] = getattr(args, 'save_interval', conf['save_interval'])
    conf['grad_clip_value'] = getattr(args, 'grad_clip_value', conf['grad_clip_value'])
    conf['model']['edge_unroll'] = getattr(args, 'edge_unroll', conf['model']['edge_unroll'])
    conf['model']['num_flow_layer'] = getattr(args, 'num_flow_layer', conf['model']['num_flow_layer'])
    conf['model']['num_rgcn_layer'] = getattr(args, 'num_rgcn_layer', conf['model']['num_rgcn_layer'])
    conf['model']['nhid'] = getattr(args, 'gaf_nhid', conf['model']['nhid'])
    conf['model']['nout'] = getattr(args, 'gaf_nout', conf['model']['nout'])
    conf['model']['deq_coeff'] = getattr(args, 'deq_coeff', conf['model']['deq_coeff'])
    conf['model']['st_type'] = getattr(args, 'st_type', conf['model']['st_type'])
    if args.model_type == 'GraphDF': conf['model']['use_df'] = True
    conf['model_ebm']['hidden'] = getattr(args, 'ebm_hidden', conf['model_ebm']['hidden'])
    conf['model_ebm']['depth'] = getattr(args, 'ebm_depth', conf['model_ebm']['depth'])
    conf['model_ebm']['swish_act'] = getattr(args, 'ebm_swish_act', conf['model_ebm']['swish_act'])
    conf['model_ebm']['add_self'] = getattr(args, 'ebm_add_self', conf['model_ebm']['add_self'])
    conf['model_ebm']['dropout'] = getattr(args, 'ebm_dropout', conf['model_ebm']['dropout'])
    conf['model_ebm']['n_power_iterations'] = getattr(args, 'ebm_n_power_iterations', conf['model_ebm']['n_power_iterations'])
    conf['train_ebm']['c'] = getattr(args, 'ebm_c', conf['train_ebm']['c'])
    conf['train_ebm']['ld_step'] = getattr(args, 'ebm_ld_step', conf['train_ebm']['ld_step'])
    conf['train_ebm']['ld_noise'] = getattr(args, 'ebm_ld_noise', conf['train_ebm']['ld_noise'])
    conf['train_ebm']['ld_step_size'] = getattr(args, 'ebm_ld_step_size', conf['train_ebm']['ld_step_size'])
    conf['train_ebm']['alpha'] = getattr(args, 'ebm_alpha', conf['train_ebm']['alpha'])
    conf['train_ebm']['clamp_lgd_grad'] = getattr(args, 'ebm_clamp_lgd_grad', conf['train_ebm']['clamp_lgd_grad'])
    conf['data_name'] = args.dataset_name # Use dataset name from args

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

    # --- Dataset Loading ---
    print(f"Loading processed augmented dataset from root: {args.data_root}, name: {args.dataset_name}")
    try:
        print("Instantiating AIGProcessedAugmentedDataset for Training...")
        train_dataset = AIGProcessedAugmentedDataset(root=args.data_root, dataset_name=args.dataset_name, split="train")
        print(f"Total training samples loaded: {len(train_dataset)}")
        if len(train_dataset) == 0: raise ValueError("Training dataset is empty after loading.")
    except FileNotFoundError:
         print(f"Error: Processed training file not found at expected location under {osp.join(args.data_root, args.dataset_name, 'processed')}")
         print("Please ensure you ran the processing script first.")
         exit(1)
    except Exception as e: print(f"Error loading training dataset: {e}"); exit(1)

    val_dataset = None
    try:
        print("Instantiating AIGProcessedAugmentedDataset for Validation...")
        val_dataset = AIGProcessedAugmentedDataset(root=args.data_root, dataset_name=args.dataset_name, split="val")
        print(f"Validation dataset loaded with {len(val_dataset)} graphs.")
        if len(val_dataset) == 0: warnings.warn("Validation dataset is empty.")
    except FileNotFoundError:
        print(f"Validation data file not found under {osp.join(args.data_root, args.dataset_name, 'processed')}. Validation will be skipped.")
    except Exception as e: print(f"Error loading validation dataset: {e}")
    val_dataset = val_dataset if val_dataset and len(val_dataset) > 0 else None # Ensure it's None if empty

    # DataLoaders
    train_loader = DenseDataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=True, drop_last=True)
    val_loader = DenseDataLoader(val_dataset, batch_size=conf['batch_size'], shuffle=False) if val_dataset else None
    print(f"Created Training DataLoader with batch size {conf['batch_size']}.")
    if val_loader: print(f"Created Validation DataLoader with batch size {conf['batch_size']}.")
    else: print("Validation DataLoader not created.")


    # --- Model Instantiation ---
    print(f"Instantiating model runner: {args.model_type}")
    runner = None
    if args.model_type == 'GraphDF': runner = GraphDF()
    elif args.model_type == 'GraphAF': runner = GraphAF()
    elif args.model_type == 'GraphEBM':
        try:
            runner = GraphEBM(n_atom=conf['model']['max_size'], n_atom_type=conf['model']['node_dim'],
                              n_edge_type=conf['model']['bond_dim'], **conf['model_ebm'], device=device) # Pass EBM conf
        except Exception as e: print(f"Error instantiating GraphEBM: {e}"); exit(1)
    else: print(f"Error: Unknown model type '{args.model_type}'."); exit(1)
    if runner is None: print(f"Failed to instantiate model runner for {args.model_type}"); exit(1)

    # Initialize the actual model within the runner
    runner.get_model(f'rand_gen_{args.model_type}', conf['model'])
    if runner.model is None: print(f"Runner failed to initialize internal model for {args.model_type}"); exit(1)
    runner.model.to(device)

    # --- Determine Save Directory ---
    if args.save_dir: save_dir = args.save_dir
    else:
        default_save_dir_base = "outputs"
        model_specific_path = f"{args.model_type}/rand_gen_{conf.get('data_name', 'aig')}_ckpts"
        save_dir = osp.join(default_save_dir_base, model_specific_path)
    conf['save_dir'] = save_dir
    os.makedirs(conf['save_dir'], exist_ok=True)
    print(f"Model checkpoints will be saved in: {conf['save_dir']}")

    # --- Training Setup ---
    # *** Suggestion: Use AdamW optimizer ***
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, runner.model.parameters()),
                                  lr=conf['lr'], weight_decay=conf['weight_decay'])
    print(f"Using AdamW optimizer with LR={conf['lr']:.1e}, WeightDecay={conf['weight_decay']:.1e}")

    scheduler = None
    if val_loader is not None and args.use_lr_scheduler:
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                                patience=args.lr_scheduler_patience, verbose=True)
         print(f"Enabled ReduceLROnPlateau LR scheduler (patience={args.lr_scheduler_patience}).")
    elif args.use_lr_scheduler:
         warnings.warn("LR scheduler requested but no validation loader available. Scheduler disabled.")

    best_val_loss = np.inf
    best_epoch = -1
    train_losses = []
    val_losses = []
    gen_validities = []

    print(f"\n--- Starting Training ({args.model_type} on {args.dataset_name}) ---")
    # --- Main Training Loop ---
    for epoch in range(1, conf['max_epochs'] + 1):
        runner.model.train()
        epoch_train_loss = 0
        processed_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{conf['max_epochs']} (Train)", leave=False)

        for i, data_batch in enumerate(pbar):
            optimizer.zero_grad()
            inp_node_features = data_batch.x.to(device)
            inp_adj_features = data_batch.adj.to(device)
            loss = 0
            try:
                # Forward pass and loss calculation
                if args.model_type == 'GraphDF' or args.model_type == 'GraphAF':
                    if hasattr(runner.model, 'dis_log_prob'):
                        out_z = runner.model(inp_node_features, inp_adj_features)
                        loss = runner.model.dis_log_prob(out_z)
                    else: raise NotImplementedError(f"{args.model_type} model missing 'dis_log_prob'")
                elif args.model_type == 'GraphEBM':
                     calc_loss_fn = getattr(runner.model, 'calculate_loss', getattr(runner, 'calculate_loss', None))
                     if calc_loss_fn: loss = calc_loss_fn(inp_node_features, inp_adj_features, conf['train_ebm'])
                     else: raise NotImplementedError("EBM loss calculation method not found.")
                else: raise NotImplementedError(f"Loss calculation not defined for {args.model_type}.")

                if torch.isnan(loss) or torch.isinf(loss):
                     warnings.warn(f"NaN/Inf loss detected: epoch {epoch}, batch {i}. Skipping update.")
                     continue # Skip this batch update

                loss.backward()

                # *** Suggestion: Apply Gradient Clipping ***
                if conf['grad_clip_value'] is not None and conf['grad_clip_value'] > 0:
                    # Clip gradients in-place
                    clip_val = torch.nn.utils.clip_grad_norm_(
                        runner.model.parameters(),
                        conf['grad_clip_value']
                    )
                    # Optional: Log clipping value if it's high
                    # if clip_val > conf['grad_clip_value'] * 0.9:
                    #     print(f"  Grad norm clipped: {clip_val:.2f} (max: {conf['grad_clip_value']})")

                optimizer.step()

                epoch_train_loss += loss.item()
                processed_batches += 1
                pbar.set_postfix({'Loss': loss.item()})

            except Exception as train_e:
                 warnings.warn(f"Error during training batch {i} (epoch {epoch}): {train_e}")
                 # Consider adding more specific error handling or logging stack trace
                 # import traceback; traceback.print_exc()
                 optimizer.zero_grad() # Clear potentially bad gradients
                 continue # Skip to next batch
        pbar.close()

        if processed_batches == 0:
            warnings.warn(f"Epoch {epoch}: No batches processed successfully. Check for errors.")
            avg_train_loss = float('nan')
        else:
            avg_train_loss = epoch_train_loss / processed_batches
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch}/{conf['max_epochs']} | Avg Train Loss: {avg_train_loss:.4f}")

        # --- Validation & Checkpointing Step ---
        if epoch % conf['save_interval'] == 0:
            avg_val_loss = run_validation(runner.model, val_loader, device, args.model_type, conf.get('train_ebm'))
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch}/{conf['max_epochs']} | Avg Valid Loss: {avg_val_loss:.4f}")

            current_validity = -1.0
            if args.validate_generation:
                 gen_args_val = {
                     "model_conf_dict": conf['model'],
                     "temperature": args.temperature_df, # Use the specific arg
                     "num_min_nodes": args.gen_min_nodes
                 }
                 current_validity = run_generation_and_eval(runner, gen_args_val, num_graphs_to_gen=args.num_val_gen_samples)
                 gen_validities.append(current_validity)
                 # Only print if validation ran successfully (not -1.0)
                 if current_validity >= 0:
                     print(f"Epoch {epoch}/{conf['max_epochs']} | Generation Validity Check: {current_validity*100:.1f}%")
                 else:
                      print(f"Epoch {epoch}/{conf['max_epochs']} | Generation Validity Check: Skipped/Error")


            # Save checkpoint
            ckpt_path = osp.join(conf['save_dir'], f'{args.model_type.lower()}_ckpt_epoch_{epoch}.pth')
            model_state_to_save = runner.model.module.state_dict() if isinstance(runner.model, nn.DataParallel) else runner.model.state_dict()
            torch.save({
                'epoch': epoch, 'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss, 'train_loss': avg_train_loss,
                'gen_validity': current_validity
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            # Save best model based on validation loss
            if avg_val_loss != float('inf') and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_ckpt_path = osp.join(conf['save_dir'], f'{args.model_type.lower()}_ckpt_best_val.pth')
                torch.save(model_state_to_save, best_ckpt_path)
                print(f"*** New best validation loss ({best_val_loss:.4f}). Saved best model to {best_ckpt_path} ***")

            # LR Scheduler Step
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif scheduler:
                scheduler.step()

    print("--- Training Finished ---")
    if best_epoch != -1:
        print(f"Best validation loss ({best_val_loss:.4f}) achieved at epoch {best_epoch}.")
    else:
        print("Validation was not performed or no improvement detected.")

    # --- Save Training History ---
    # (Keep history saving logic as before)
    num_epochs_run = len(train_losses)
    num_val_epochs = len(val_losses)
    num_gen_epochs = len(gen_validities)
    padded_val_losses = ([None] * (num_epochs_run - num_val_epochs)) + val_losses if num_val_epochs < num_epochs_run else val_losses
    padded_gen_validity = ([None] * (num_epochs_run - num_gen_epochs)) + gen_validities if num_gen_epochs < num_epochs_run else gen_validities
    history = {'train_loss': train_losses, 'val_loss': padded_val_losses, 'gen_validity': padded_gen_validity,
               'best_val_epoch': best_epoch, 'best_val_loss': best_val_loss if best_epoch != -1 else None}
    history_path = osp.join(conf['save_dir'], f'{args.model_type.lower()}_training_history.json')
    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
        print(f"Saved training history to {history_path}")
    except Exception as e: print(f"Error saving training history: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train graph generation models on pre-processed & augmented AIG dataset.")

    # --- General Arguments ---
    parser.add_argument('--model_type', type=str, default='GraphDF', choices=['GraphDF', 'GraphAF', 'GraphEBM'], help='Model to train.')
    parser.add_argument('--data_root', default="./data_pyg_augmented/",
                        help="Root directory where dataset subfolders were created by the processing script.")
    parser.add_argument('--dataset_name', default="aig_ds",
                        help="Name of the dataset used during processing.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for training.') # Default to cuda if available
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save model checkpoints.')

    # --- General Training Hyperparameters (Reflecting Suggestions) ---
    parser.add_argument('--lr', type=float, default=base_conf['lr'], help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=base_conf['weight_decay'], help="Weight decay (L2 penalty).")
    parser.add_argument('--batch_size', type=int, default=base_conf['batch_size'], help="Batch size.")
    parser.add_argument('--max_epochs', type=int, default=base_conf['max_epochs'], help="Maximum training epochs.")
    parser.add_argument('--save_interval', type=int, default=base_conf['save_interval'], help="Save checkpoints every N epochs.")
    parser.add_argument('--grad_clip_value', type=float, default=base_conf['grad_clip_value'], help='Max norm for gradient clipping. None or 0 to disable.')
    parser.add_argument('--use_lr_scheduler', action='store_true', default=True, help='Enable ReduceLROnPlateau LR scheduler.') # Default to True
    parser.add_argument('--lr_scheduler_patience', type=int, default=5, help='Patience for ReduceLROnPlateau scheduler.') # Increased patience

    # --- Validation & Generation Check Arguments ---
    parser.add_argument('--validate_generation', action='store_true', default=True, help='Generate/evaluate graphs during validation.') # Default to True
    parser.add_argument('--num_val_gen_samples', type=int, default=100, help='Number of graphs for validity check.') # Increase samples
    parser.add_argument('--gen_min_nodes', type=int, default=5, help='Min nodes for generated graphs during validation.')
    parser.add_argument('--temperature_df', type=float, nargs=2, default=[0.7, 0.7], help='Temperature [node, edge] (for GDF/GAF generation check).') # Slightly higher temp?

    # --- GraphAF/GraphDF Specific Model Hyperparameters ---
    parser.add_argument('--edge_unroll', type=int, default=base_conf['model']['edge_unroll'], help="Max look-back distance for edge connections.")
    parser.add_argument('--num_flow_layer', type=int, default=base_conf['model']['num_flow_layer'], help="Number of flow layers.")
    parser.add_argument('--num_rgcn_layer', type=int, default=base_conf['model']['num_rgcn_layer'], help="Number of RGCN layers.")
    parser.add_argument('--gaf_nhid', type=int, default=base_conf['model']['nhid'], help="Hidden dim for GAF/GDF.")
    parser.add_argument('--gaf_nout', type=int, default=base_conf['model']['nout'], help="Output dim (embedding size) for GAF/GDF.")
    parser.add_argument('--deq_coeff', type=float, default=base_conf['model']['deq_coeff'], help="Dequantization coefficient.")
    parser.add_argument('--st_type', type=str, default=base_conf['model']['st_type'], choices=['exp', 'sigmoid', 'softplus'], help="ST network type.")

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
