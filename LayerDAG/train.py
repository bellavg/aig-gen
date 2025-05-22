# train.py
import dgl.sparse as dglsp
import pandas as pd
import time
import torch
import torch.nn as nn
import wandb  # For logging
import multiprocessing as mp # Import multiprocessing

from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

# Options: 'highest' (default), 'high', 'medium'
# 'high' or 'medium' can leverage Tensor Cores for float32 matrix multiplications


from setup_utils import set_seed, load_yaml
# Ensure your dataset components are correctly imported
from src.dataset import (
    load_dataset,
    LayerDAGNodeCountDataset,
    LayerDAGNodePredDataset,
    LayerDAGEdgePredDataset,
    collate_node_count,
    collate_node_pred,
    collate_edge_pred
)
from src.model import DiscreteDiffusion, EdgeDiscreteDiffusion, LayerDAG


@torch.no_grad()
def eval_node_count(device, val_loader, model, is_conditional):
    """
    Evaluates the node count prediction model.
    Args:
        device: The device to run evaluation on.
        val_loader: DataLoader for the validation set.
        model: The node count prediction model.
        is_conditional (bool): Flag indicating if the model is conditional.
    Returns:
        Tuple of (average NLL, accuracy).
    """
    model.eval()
    total_nll = 0
    total_count = 0
    true_count = 0
    for batch_data in tqdm(val_loader, desc="Eval Node Count"):
        # Unpack batch data based on whether it's conditional
        if is_conditional:
            batch_size, batch_edge_index, batch_x_n, batch_abs_level, \
                batch_rel_level, batch_y, batch_n2g_index, batch_label = batch_data
            batch_y = batch_y.to(device)
        else:
            batch_size, batch_edge_index, batch_x_n, batch_abs_level, \
                batch_rel_level, batch_n2g_index, batch_label = batch_data
            batch_y = None  # Explicitly set to None if not conditional

        num_nodes = len(batch_x_n)
        # Ensure edge_index is not empty before creating spmatrix
        if batch_edge_index.numel() > 0:
            batch_A = dglsp.spmatrix(
                batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
        else:
            # Create an empty sparse matrix if there are no edges
            batch_A = dglsp.spmatrix(
                torch.empty((2, 0), dtype=torch.long, device=device),  # Ensure it's on the correct device
                shape=(num_nodes, num_nodes)).to(device)

        batch_x_n = batch_x_n.to(device)
        batch_abs_level = batch_abs_level.to(device)
        batch_rel_level = batch_rel_level.to(device)
        batch_A_n2g = dglsp.spmatrix(
            batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
        batch_label = batch_label.to(device)

        batch_logits = model(batch_A, batch_x_n, batch_abs_level,
                             batch_rel_level, batch_A_n2g, batch_y)

        batch_nll = -batch_logits.log_softmax(dim=-1)
        # Clamp label to be within the range of predicted logits
        batch_label_clamped = batch_label.clamp(max=batch_nll.shape[-1] - 1)
        # Ensure batch_size is not zero before indexing
        if batch_size > 0:
            batch_nll = batch_nll[torch.arange(batch_size, device=device), batch_label_clamped]
            total_nll += batch_nll.sum().item()

        batch_probs = batch_logits.softmax(dim=-1)
        batch_preds = batch_probs.multinomial(1).squeeze(-1)
        true_count += (batch_preds == batch_label).sum().item()  # Compare with original label

        total_count += batch_size

    return total_nll / total_count if total_count > 0 else float('inf'), \
        true_count / total_count if total_count > 0 else 0


def main_node_count(device, train_loader, val_loader, model, config, patience, is_conditional):
    """
    Main training loop for the node count prediction model.
    Args:
        device: The device to run training on.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        model: The node count prediction model.
        config: Configuration dictionary for node count training.
        patience (int): Number of epochs to wait for improvement before early stopping.
        is_conditional (bool): Flag indicating if the model is conditional.
    Returns:
        The state dictionary of the best performing model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])

    best_val_nll = float('inf')
    best_val_acc = 0
    best_state_dict = deepcopy(model.state_dict())
    num_patient_epochs = 0

    for epoch in range(config['num_epochs']):
        model.train()
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} Node Count"):
            if is_conditional:
                batch_size, batch_edge_index, batch_x_n, batch_abs_level, \
                    batch_rel_level, batch_y, batch_n2g_index, batch_label = batch_data
                batch_y = batch_y.to(device)
            else:
                batch_size, batch_edge_index, batch_x_n, batch_abs_level, \
                    batch_rel_level, batch_n2g_index, batch_label = batch_data
                batch_y = None

            num_nodes = len(batch_x_n)

            # ---- ADDED DEBUG ----
            if batch_edge_index.numel() > 0:
                max_val_in_edge_index = batch_edge_index.max().item()
                min_val_in_edge_index = batch_edge_index.min().item()
                if max_val_in_edge_index >= num_nodes or min_val_in_edge_index < 0:
                    error_msg = (
                        f"CRITICAL ERROR in batch for Node Count training:\n"
                        f"  batch_edge_index.max() ({max_val_in_edge_index}) >= num_nodes ({num_nodes}) OR "
                        f"batch_edge_index.min() ({min_val_in_edge_index}) < 0.\n"
                        f"  batch_x_n.shape: {batch_x_n.shape}\n"
                        f"  batch_edge_index.shape: {batch_edge_index.shape}\n"
                    )
                    print(error_msg)
                    # To find the problematic graph(s) within the batch, you'd need to inspect
                    # the components of `batch_data` that formed this `batch_edge_index` and `batch_x_n`
                    # before collation, or save `batch_data` itself.
                    # For now, we'll raise an error to halt execution.
                    # Consider more sophisticated error handling or skipping the batch if appropriate.
                    raise ValueError("Invalid edge index detected for spmatrix, halting.")
            # ---- END ADDED DEBUG ----

            if batch_edge_index.numel() > 0:
                batch_A = dglsp.spmatrix(batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
            else:
                batch_A = dglsp.spmatrix(
                    torch.empty((2, 0), dtype=torch.long, device=device),
                    shape=(num_nodes, num_nodes)).to(device)

            batch_x_n = batch_x_n.to(device)
            batch_abs_level = batch_abs_level.to(device)
            batch_rel_level = batch_rel_level.to(device)
            batch_A_n2g = dglsp.spmatrix(batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
            batch_label = batch_label.to(device)

            batch_pred = model(batch_A, batch_x_n, batch_abs_level,
                               batch_rel_level, batch_A_n2g, batch_y)

            loss = criterion(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch + 1 % 10 == 0:
                print(f"Epoch {epoch + 1}, Node Count Loss: {loss.item():.4f}")

            wandb.log({'node_count/loss': loss.item()})


        val_nll, val_acc = eval_node_count(device, val_loader, model, is_conditional)
        wandb.log({
            'node_count/epoch': epoch,
            'node_count/val_nll': val_nll,
            'node_count/val_acc': val_acc,
        })

        if val_acc > best_val_acc:  # Prioritize accuracy, then NLL for tie-breaking
            best_val_acc = val_acc
            best_val_nll = val_nll  # Update NLL if accuracy improved
            best_state_dict = deepcopy(model.state_dict())
            num_patient_epochs = 0
        elif val_acc == best_val_acc and val_nll < best_val_nll:  # If accuracy is same, check NLL
            best_val_nll = val_nll
            best_state_dict = deepcopy(model.state_dict())
            num_patient_epochs = 0
        else:
            num_patient_epochs += 1

        wandb.log({
            'node_count/best_val_nll': best_val_nll,
            'node_count/best_val_acc': best_val_acc,
            'node_count/num_patient_epochs': num_patient_epochs
        })

        if (patience is not None) and (num_patient_epochs >= patience):
            print(f"Node Count: Early stopping after {patience} epochs without improvement.")
            break
    if epoch == config['num_epochs'] - 1:
        print(f"Node Count: Max epochs reached.")

    print(f"Node Count: Best validation accuracy: {best_val_acc:.4f}, Best NLL: {best_val_nll:.4f}")
    return best_state_dict


@torch.no_grad()
def eval_node_pred(device, val_loader, model, is_conditional):
    """
    Evaluates the node prediction model.
    Args:
        device: The device to run evaluation on.
        val_loader: DataLoader for the validation set.
        model: The node prediction model.
        is_conditional (bool): Flag indicating if the model is conditional.
    Returns:
        Average NLL.
    """
    model.eval()
    total_nll = 0
    total_num_attribute_predictions = 0
    for batch_data in tqdm(val_loader, desc="Eval Node Pred"):
        if is_conditional:
            batch_size, batch_edge_index, batch_x_n, batch_abs_level, \
                batch_rel_level, batch_n2g_index, batch_z_t, batch_t, batch_y, \
                query2g, num_query_cumsum, batch_z = batch_data
            batch_y = batch_y.to(device)
        else:
            batch_size, batch_edge_index, batch_x_n, batch_abs_level, \
                batch_rel_level, batch_n2g_index, batch_z_t, batch_t, \
                query2g, num_query_cumsum, batch_z = batch_data
            batch_y = None

        num_nodes = len(batch_x_n)
        if batch_edge_index.numel() > 0:
            batch_A = dglsp.spmatrix(
                batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
        else:
            batch_A = dglsp.spmatrix(
                torch.empty((2, 0), dtype=torch.long, device=device),
                shape=(num_nodes, num_nodes)).to(device)

        batch_x_n = batch_x_n.to(device)
        batch_abs_level = batch_abs_level.to(device)
        batch_rel_level = batch_rel_level.to(device)
        batch_A_n2g = dglsp.spmatrix(
            batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
        batch_z_t = batch_z_t.to(device)
        batch_t = batch_t.to(device)
        query2g = query2g.to(device)
        num_query_cumsum = num_query_cumsum.to(device)
        batch_z = batch_z.to(device)

        batch_logits_list = model(batch_A, batch_x_n, batch_abs_level,
                                  batch_rel_level, batch_A_n2g, batch_z_t, batch_t,
                                  query2g, num_query_cumsum, batch_y)

        num_feature_dims = len(batch_logits_list)
        if num_feature_dims == 0:
            continue
        current_batch_num_queries = batch_logits_list[0].shape[0]
        if current_batch_num_queries == 0:
            continue

        for d in range(num_feature_dims):
            batch_logits_d = batch_logits_list[d]
            ground_truth_d = batch_z[:, d]

            batch_nll_d = -batch_logits_d.log_softmax(dim=-1)
            batch_nll_d = batch_nll_d[torch.arange(current_batch_num_queries, device=device), ground_truth_d]
            total_nll += batch_nll_d.sum().item()

        total_num_attribute_predictions += current_batch_num_queries * num_feature_dims

    return total_nll / total_num_attribute_predictions if total_num_attribute_predictions > 0 else float('inf')


def main_node_pred(device, train_loader, val_loader, model, config, patience, is_conditional):
    """
    Main training loop for the node prediction model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])

    best_val_nll = float('inf')
    best_state_dict = deepcopy(model.state_dict())
    num_patient_epochs = 0

    for epoch in range(config['num_epochs']):
        model.train()
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} Node Pred"):
            if is_conditional:
                batch_size, batch_edge_index, batch_x_n, batch_abs_level, \
                    batch_rel_level, batch_n2g_index, batch_z_t, batch_t, \
                    batch_y, query2g, num_query_cumsum, batch_z = batch_data
                batch_y = batch_y.to(device)
            else:
                batch_size, batch_edge_index, batch_x_n, batch_abs_level, \
                    batch_rel_level, batch_n2g_index, batch_z_t, batch_t, \
                    query2g, num_query_cumsum, batch_z = batch_data
                batch_y = None

            num_nodes = len(batch_x_n)
            # ---- ADD THIS VALIDATION ----
            if batch_edge_index.numel() > 0:
                max_val_in_edge_index = batch_edge_index.max().item()
                min_val_in_edge_index = batch_edge_index.min().item()
                if max_val_in_edge_index >= num_nodes or min_val_in_edge_index < 0:
                    error_msg = (
                        f"CRITICAL ERROR in batch for Node Prediction training:\n"
                        f"  batch_edge_index.max() ({max_val_in_edge_index}) >= num_nodes ({num_nodes}) OR "
                        f"batch_edge_index.min() ({min_val_in_edge_index}) < 0.\n"
                        f"  batch_x_n.shape: {batch_x_n.shape}\n"
                        f"  batch_edge_index.shape: {batch_edge_index.shape}\n"
                    )
                    print(error_msg)
                    # Consider raising ValueError to stop execution cleanly before CUDA error
                    raise ValueError("Invalid edge index for dglsp.spmatrix in main_node_pred.")
            # ---- END VALIDATION ----

            if batch_edge_index.numel() > 0:
                batch_A = dglsp.spmatrix(
                    batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
            else:
                batch_A = dglsp.spmatrix(
                    torch.empty((2, 0), dtype=torch.long, device=device),
                    shape=(num_nodes, num_nodes)).to(device)

            batch_x_n = batch_x_n.to(device)
            batch_abs_level = batch_abs_level.to(device)
            batch_rel_level = batch_rel_level.to(device)
            batch_A_n2g = dglsp.spmatrix(
                batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
            batch_z_t = batch_z_t.to(device)
            batch_t = batch_t.to(device)
            query2g = query2g.to(device)
            num_query_cumsum = num_query_cumsum.to(device)
            batch_z = batch_z.to(device)

            batch_pred_logits_list = model(batch_A, batch_x_n, batch_abs_level,
                                           batch_rel_level, batch_A_n2g, batch_z_t,
                                           batch_t, query2g, num_query_cumsum, batch_y)

            loss = 0
            num_feature_dims = len(batch_pred_logits_list)
            if num_feature_dims > 0 and batch_pred_logits_list[0].shape[0] > 0:
                for d in range(num_feature_dims):
                    loss = loss + criterion(batch_pred_logits_list[d], batch_z[:, d])
                loss /= num_feature_dims
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            wandb.log({'node_pred/loss': loss.item()})

        val_nll = eval_node_pred(device, val_loader, model, is_conditional)
        wandb.log({
            'node_pred/epoch': epoch,
            'node_pred/val_nll': val_nll,
        })

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state_dict = deepcopy(model.state_dict())
            num_patient_epochs = 0
        else:
            num_patient_epochs += 1

        wandb.log({
            'node_pred/best_val_nll': best_val_nll,
            'node_pred/num_patient_epochs': num_patient_epochs
        })

        if (patience is not None) and (num_patient_epochs >= patience):
            print(f"Node Pred: Early stopping after {patience} epochs without improvement.")
            break
    if epoch == config['num_epochs'] - 1:
        print(f"Node Pred: Max epochs reached.")

    print(f"Node Pred: Best validation NLL: {best_val_nll:.4f}")
    return best_state_dict


@torch.no_grad()
def eval_edge_pred(device, val_loader, model, is_conditional):
    """
    Evaluates the edge prediction model.
    """
    model.eval()
    total_nll = 0
    total_queries = 0
    for batch_data in tqdm(val_loader, desc="Eval Edge Pred"):
        if is_conditional:
            batch_edge_index, batch_noisy_edge_index, batch_x_n, \
                batch_abs_level, batch_rel_level, batch_t, batch_y, \
                batch_query_src, batch_query_dst, batch_label = batch_data
            batch_y = batch_y.to(device)
        else:
            batch_edge_index, batch_noisy_edge_index, batch_x_n, \
                batch_abs_level, batch_rel_level, batch_t, \
                batch_query_src, batch_query_dst, batch_label = batch_data
            batch_y = None

        num_nodes = len(batch_x_n)

        if batch_edge_index.numel() == 0 and batch_noisy_edge_index.numel() == 0:
            combined_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        elif batch_edge_index.numel() == 0:
            combined_edge_index = batch_noisy_edge_index
        elif batch_noisy_edge_index.numel() == 0:
            combined_edge_index = batch_edge_index
        else:
            combined_edge_index = torch.cat([batch_edge_index, batch_noisy_edge_index], dim=1)

        if combined_edge_index.numel() > 0:
            batch_A = dglsp.spmatrix(
                combined_edge_index,
                shape=(num_nodes, num_nodes)).to(device)
        else:
            batch_A = dglsp.spmatrix(
                torch.empty((2, 0), dtype=torch.long, device=device),
                shape=(num_nodes, num_nodes)).to(device)

        batch_x_n = batch_x_n.to(device)
        batch_abs_level = batch_abs_level.to(device)
        batch_rel_level = batch_rel_level.to(device)
        batch_t = batch_t.to(device)
        batch_query_src = batch_query_src.to(device)
        batch_query_dst = batch_query_dst.to(device)
        batch_label = batch_label.to(device)

        batch_logits = model(batch_A, batch_x_n, batch_abs_level,
                             batch_rel_level, batch_t, batch_query_src,
                             batch_query_dst, batch_y)

        current_batch_num_queries = batch_logits.shape[0]
        if current_batch_num_queries > 0:
            batch_nll = -batch_logits.log_softmax(dim=-1)
            batch_nll = batch_nll[
                torch.arange(current_batch_num_queries, device=device), batch_label]
            total_nll += batch_nll.sum().item()
            total_queries += current_batch_num_queries

    return total_nll / total_queries if total_queries > 0 else float('inf')


def main_edge_pred(device, train_loader, val_loader, model, config, patience, is_conditional):
    """
    Main training loop for the edge prediction model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])

    best_val_nll = float('inf')
    best_state_dict = deepcopy(model.state_dict())
    num_patient_epochs = 0

    for epoch in range(config['num_epochs']):
        model.train()
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} Edge Pred"):
            if is_conditional:
                batch_edge_index, batch_noisy_edge_index, batch_x_n, \
                    batch_abs_level, batch_rel_level, batch_t, \
                    batch_y, batch_query_src, batch_query_dst, batch_label = batch_data
                batch_y = batch_y.to(device)
            else:
                batch_edge_index, batch_noisy_edge_index, batch_x_n, \
                    batch_abs_level, batch_rel_level, batch_t, \
                    batch_query_src, batch_query_dst, batch_label = batch_data
                batch_y = None

            num_nodes = len(batch_x_n)
            if batch_edge_index.numel() == 0 and batch_noisy_edge_index.numel() == 0:
                combined_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            elif batch_edge_index.numel() == 0:
                combined_edge_index = batch_noisy_edge_index
            elif batch_noisy_edge_index.numel() == 0:
                combined_edge_index = batch_edge_index
            else:
                combined_edge_index = torch.cat([batch_edge_index, batch_noisy_edge_index], dim=1)

            if combined_edge_index.numel() > 0:
                batch_A = dglsp.spmatrix(
                    combined_edge_index,
                    shape=(num_nodes, num_nodes)).to(device)
            else:
                batch_A = dglsp.spmatrix(
                    torch.empty((2, 0), dtype=torch.long, device=device),
                    shape=(num_nodes, num_nodes)).to(device)

            batch_x_n = batch_x_n.to(device)
            batch_abs_level = batch_abs_level.to(device)
            batch_rel_level = batch_rel_level.to(device)
            batch_t = batch_t.to(device)
            batch_query_src = batch_query_src.to(device)
            batch_query_dst = batch_query_dst.to(device)
            batch_label = batch_label.to(device)

            batch_pred_logits = model(batch_A, batch_x_n, batch_abs_level,
                                      batch_rel_level, batch_t, batch_query_src,
                                      batch_query_dst, batch_y)

            if batch_pred_logits.shape[0] > 0:
                loss = criterion(batch_pred_logits, batch_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                wandb.log({'edge_pred/loss': loss.item()})
            else:
                wandb.log({'edge_pred/loss': 0.0})

        val_nll = eval_edge_pred(device, val_loader, model, is_conditional)
        wandb.log({
            'edge_pred/epoch': epoch,
            'edge_pred/val_nll': val_nll,
        })

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state_dict = deepcopy(model.state_dict())
            num_patient_epochs = 0
        else:
            num_patient_epochs += 1

        wandb.log({
            'edge_pred/best_val_nll': best_val_nll,
            'edge_pred/num_patient_epochs': num_patient_epochs
        })

        if (patience is not None) and (num_patient_epochs >= patience):
            print(f"Edge Pred: Early stopping after {patience} epochs without improvement.")
            break
    if epoch == config['num_epochs'] - 1:
        print(f"Edge Pred: Max epochs reached.")

    print(f"Edge Pred: Best validation NLL: {best_val_nll:.4f}")
    return best_state_dict


def main(args):
    # ADD THIS LINE AT THE BEGINNING OF YOUR main FUNCTION OR THE SCRIPT
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
            print("Multiprocessing start method set to 'spawn'.")
        except RuntimeError:
            print("Could not set multiprocessing start method to 'spawn'. This might lead to CUDA errors in subprocesses.")


    torch.set_num_threads(args.num_threads)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    set_seed(args.seed)

    config = load_yaml(args.config_file)
    if config is None:
        print("ERROR: Config is None. YAML file might not have been loaded correctly.")
        exit(1)
    if 'general' not in config:
        print("ERROR: 'general' key not found in config.")
        exit(1)

    dataset_name = config['general']['dataset']
    config_df = pd.json_normalize(config, sep='/')  # For wandb logging

    ts = time.strftime('%b%d-%H%M%S', time.gmtime())

    wandb.init(
        project=f'LayerDAG_{dataset_name}',
        name=f'{ts}_{dataset_name}',
        config=config_df.to_dict(orient='records')[0]
    )

    is_conditional = config['general']['conditional']

    # --- Prepare dataset loading arguments ---
    load_dataset_kwargs = {'conditional': is_conditional}
    if dataset_name == 'aig':
        if 'path_to_pt_file' not in config['general']:
            raise ValueError("'path_to_pt_file' missing in config for AIG dataset.")
        if 'num_node_categories' not in config['general']:
            raise ValueError("'num_node_categories' missing in config for AIG dataset.")
        load_dataset_kwargs['path'] = config['general']['path_to_pt_file']
        load_dataset_kwargs['num_categories_actual'] = config['general']['num_node_categories']
    elif dataset_name == 'tpu_tile':
        pass
    else:
        raise ValueError(f"Unsupported dataset name in config: {dataset_name}")

    print(f"Loading dataset: {dataset_name} (Conditional: {is_conditional})")
    train_set, val_set, _ = load_dataset(dataset_name, **load_dataset_kwargs)
    print(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}")

    # --- Create LayerDAG specific datasets ---
    print("Preparing LayerDAG-specific datasets...")
    train_node_count_dataset = LayerDAGNodeCountDataset(train_set, conditional=is_conditional)
    val_node_count_dataset = LayerDAGNodeCountDataset(val_set, conditional=is_conditional)
    print(f"Node count dataset: Max layer size from train_set: {train_node_count_dataset.max_layer_size}")

    train_node_pred_dataset = LayerDAGNodePredDataset(train_set, conditional=is_conditional, get_marginal=True)
    val_node_pred_dataset = LayerDAGNodePredDataset(val_set, conditional=is_conditional, get_marginal=False)

    if not hasattr(train_node_pred_dataset, 'x_n_marginal') or not train_node_pred_dataset.x_n_marginal:
        raise ValueError("x_n_marginal is empty or not set in train_node_pred_dataset. Check dataset processing.")

    node_diffusion_config = {
        'marginal_list': train_node_pred_dataset.x_n_marginal,
        'T': config['node_pred']['T']
    }
    node_diffusion = DiscreteDiffusion(**node_diffusion_config)
    train_node_pred_dataset.node_diffusion = node_diffusion
    val_node_pred_dataset.node_diffusion = node_diffusion

    train_edge_pred_dataset = LayerDAGEdgePredDataset(train_set, conditional=is_conditional)
    val_edge_pred_dataset = LayerDAGEdgePredDataset(val_set, conditional=is_conditional)

    # --- MODIFICATION START: Initialize edge_diffusion ---
    edge_diffusion_config = {
        'avg_in_deg': train_edge_pred_dataset.avg_in_deg,
        'T': config['edge_pred']['T']
    }
    edge_diffusion = EdgeDiscreteDiffusion(**edge_diffusion_config)
    train_edge_pred_dataset.edge_diffusion = edge_diffusion
    val_edge_pred_dataset.edge_diffusion = edge_diffusion
    # --- MODIFICATION END ---

    # --- Create DataLoaders ---
    print("Creating DataLoaders...")
    node_count_train_loader = DataLoader(
        train_node_count_dataset,
        batch_size=config['node_count']['loader']['batch_size'],
        shuffle=True,
        num_workers=config['node_count']['loader']['num_workers'],
        collate_fn=collate_node_count,
        pin_memory=False
    )
    node_count_val_loader = DataLoader(
        val_node_count_dataset,
        batch_size=config['node_count']['loader']['batch_size'],
        shuffle=False,
        num_workers=config['node_count']['loader']['num_workers'],
        collate_fn=collate_node_count,
        pin_memory=False
    )

    node_pred_train_loader = DataLoader(
        train_node_pred_dataset,
        batch_size=config['node_pred']['loader']['batch_size'],
        shuffle=True,
        num_workers=config['node_pred']['loader']['num_workers'],
        collate_fn=collate_node_pred,
        pin_memory=False
    )
    node_pred_val_loader = DataLoader(
        val_node_pred_dataset,
        batch_size=config['node_pred']['loader']['batch_size'],
        shuffle=False,
        num_workers=config['node_pred']['loader']['num_workers'],
        collate_fn=collate_node_pred,
        pin_memory=False
    )

    edge_pred_train_loader = DataLoader(
        train_edge_pred_dataset,
        batch_size=config['edge_pred']['loader']['batch_size'],
        shuffle=True,
        num_workers=config['edge_pred']['loader']['num_workers'],
        collate_fn=collate_edge_pred,
        pin_memory=False
    )
    edge_pred_val_loader = DataLoader(
        val_edge_pred_dataset,
        batch_size=config['edge_pred']['loader']['batch_size'],
        shuffle=False,
        num_workers=config['edge_pred']['loader']['num_workers'],
        collate_fn=collate_edge_pred,
        pin_memory=False
    )
    print("DataLoaders created.")

    # --- Model Configuration & Initialization ---
    model_num_x_n_cat = train_set.num_categories

    model_config = {
        'num_x_n_cat': model_num_x_n_cat,
        'node_count_encoder_config': config['node_count']['model'],
        'max_layer_size': train_node_count_dataset.max_layer_size,
        'node_pred_graph_encoder_config': config['node_pred']['graph_encoder'],
        'node_predictor_config': config['node_pred']['predictor'],
        'edge_pred_graph_encoder_config': config['edge_pred']['graph_encoder'],
        'edge_predictor_config': config['edge_pred']['predictor'],
        'max_level': max(train_node_pred_dataset.input_level.max().item(),
                         val_node_pred_dataset.input_level.max().item())
    }
    if isinstance(model_config['num_x_n_cat'], int):
        model_config['num_x_n_cat'] = torch.LongTensor([model_config['num_x_n_cat']])
    elif isinstance(model_config['num_x_n_cat'], list):
        model_config['num_x_n_cat'] = torch.LongTensor(model_config['num_x_n_cat'])

    print("Initializing LayerDAG model...")
    model = LayerDAG(device=device,
                     node_diffusion=node_diffusion,
                     is_model_conditional = False,
                     edge_diffusion=edge_diffusion,  # Now edge_diffusion is defined
                     **model_config)
    model.to(device)
    print("LayerDAG model initialized and moved to device.")

    patience_val = config['general'].get('patience', 10)

    # --- Training Sub-models ---
    print("\n--- Training Node Count Model ---")
    node_count_state_dict = main_node_count(
        device,
        node_count_train_loader,
        node_count_val_loader,
        model.node_count_model, config['node_count'], patience_val, is_conditional)
    model.node_count_model.load_state_dict(node_count_state_dict)
    print("--- Finished Training Node Count Model ---\n")

    print("--- Training Node Prediction Model ---")
    node_pred_state_dict = main_node_pred(
        device,
        node_pred_train_loader,
        node_pred_val_loader,
        model.node_pred_model, config['node_pred'], patience_val, is_conditional)
    model.node_pred_model.load_state_dict(node_pred_state_dict)
    print("--- Finished Training Node Prediction Model ---\n")

    print("--- Training Edge Prediction Model ---")
    edge_pred_state_dict = main_edge_pred(
        device,
        edge_pred_train_loader,
        edge_pred_val_loader,
        model.edge_pred_model, config['edge_pred'], patience_val, is_conditional)
    model.edge_pred_model.load_state_dict(edge_pred_state_dict)
    print("--- Finished Training Edge Prediction Model ---\n")

    # --- Save Model ---
    save_path = f'model_{dataset_name}_{ts}.pth'
    # --- MODIFICATION START: Include edge_diffusion_config in saved checkpoint ---
    torch.save({
        'dataset': dataset_name,
        'node_diffusion_config': node_diffusion_config,
        'edge_diffusion_config': edge_diffusion_config,  # Add this
        'model_config': model_config,
        'is_conditional': is_conditional,
        'model_state_dict': model.state_dict()
    }, save_path)
    # --- MODIFICATION END ---
    print(f"Model saved to {save_path}")
    wandb.save(save_path)
    wandb.finish()


if __name__ == '__main__':
    # IT IS CRITICAL TO PUT THIS INSIDE THE if __name__ == '__main__': BLOCK
    # OR AT THE VERY TOP OF THE SCRIPT IF IT'S ALWAYS EXECUTED DIRECTLY.
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
            print("Multiprocessing start method set to 'spawn' (from __main__).")
        except RuntimeError as e:
            print(f"Warning: Could not set multiprocessing start method to 'spawn' in __main__: {e}")

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:  # Check for Volta or newer
        print("Setting float32 matmul precision to 'high' for Tensor Cores.")
        torch.set_float32_matmul_precision('high')

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Train LayerDAG model.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of CPU threads for PyTorch (DGL recommendation is often 1 for GNNs to avoid overhead).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    main(args)