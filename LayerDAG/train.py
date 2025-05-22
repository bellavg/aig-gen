# train.py
import dgl.sparse as dglsp
import pandas as pd
import time
import torch
import torch.nn as nn
import wandb  # For logging
import os  # Added for dataset caching

from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import for AMP
from torch.cuda.amp import GradScaler, autocast

import torch  # Set matmul precision for Tensor Cores

# Options: 'highest' (default), 'high', 'medium'
# 'high' or 'medium' can leverage Tensor Cores for float32 matrix multiplications
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:  # Check for Volta or newer
    print("Setting float32 matmul precision to 'high' for Tensor Cores.")
    torch.set_float32_matmul_precision('high')

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

        # For evaluation, dtypes should generally be float32 unless specific ops benefit from float16
        # and are handled carefully. Here, we'll use float32 for sparse matrix values in eval.
        sparse_val_dtype_eval = torch.float32
        num_edges_eval = batch_edge_index.shape[1]

        if batch_edge_index.numel() > 0:
            vals_eval = torch.ones(num_edges_eval, dtype=sparse_val_dtype_eval, device=device)
            batch_A = dglsp.spmatrix(
                batch_edge_index, val=vals_eval, shape=(num_nodes, num_nodes)).to(device)
        else:
            batch_A = dglsp.spmatrix(
                torch.empty((2, 0), dtype=torch.long, device=device),
                val=torch.empty((0,), dtype=sparse_val_dtype_eval, device=device),
                shape=(num_nodes, num_nodes)).to(device)

        batch_x_n = batch_x_n.to(device)
        batch_abs_level = batch_abs_level.to(device)
        batch_rel_level = batch_rel_level.to(device)
        batch_A_n2g = dglsp.spmatrix(
            batch_n2g_index, shape=(batch_size, num_nodes)).to(
            device)  # A_n2g is typically binary, no explicit vals needed for it
        batch_label = batch_label.to(device)

        batch_logits = model(batch_A, batch_x_n, batch_abs_level,
                             batch_rel_level, batch_A_n2g, batch_y)

        batch_nll = -batch_logits.log_softmax(dim=-1)
        batch_label_clamped = batch_label.clamp(max=batch_nll.shape[-1] - 1)
        if batch_size > 0:
            batch_nll = batch_nll[torch.arange(batch_size, device=device), batch_label_clamped]
            total_nll += batch_nll.sum().item()

        batch_probs = batch_logits.softmax(dim=-1)
        batch_preds = batch_probs.multinomial(1).squeeze(-1)
        true_count += (batch_preds == batch_label).sum().item()

        total_count += batch_size

    return total_nll / total_count if total_count > 0 else float('inf'), \
        true_count / total_count if total_count > 0 else 0


def main_node_count(device, train_loader, val_loader, model, config, patience, is_conditional):
    """
    Main training loop for the node count prediction model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

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
            num_edges = batch_edge_index.shape[1]

            # Determine dtype for sparse matrix values based on AMP state
            sparse_val_dtype = torch.float16 if scaler.is_enabled() else torch.float32

            if batch_edge_index.numel() > 0:
                vals = torch.ones(num_edges, dtype=sparse_val_dtype, device=device)
                batch_A = dglsp.spmatrix(batch_edge_index, val=vals, shape=(num_nodes, num_nodes)).to(device)
            else:
                batch_A = dglsp.spmatrix(
                    torch.empty((2, 0), dtype=torch.long, device=device),
                    val=torch.empty((0,), dtype=sparse_val_dtype, device=device),
                    shape=(num_nodes, num_nodes)).to(device)

            batch_x_n = batch_x_n.to(device)
            batch_abs_level = batch_abs_level.to(device)
            batch_rel_level = batch_rel_level.to(device)
            # A_n2g is typically binary and used for pooling, its implicit values' dtype might not conflict
            # or DGL handles it. If issues arise here, it might also need explicit vals.
            batch_A_n2g = dglsp.spmatrix(batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
            batch_label = batch_label.to(device)

            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                batch_pred = model(batch_A, batch_x_n, batch_abs_level,
                                   batch_rel_level, batch_A_n2g, batch_y)
                loss = criterion(batch_pred, batch_label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            wandb.log({'node_count/loss': loss.item()})

        val_nll, val_acc = eval_node_count(device, val_loader, model, is_conditional)
        wandb.log({
            'node_count/epoch': epoch,
            'node_count/val_nll': val_nll,
            'node_count/val_acc': val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_nll = val_nll
            best_state_dict = deepcopy(model.state_dict())
            num_patient_epochs = 0
        elif val_acc == best_val_acc and val_nll < best_val_nll:
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
        sparse_val_dtype_eval = torch.float32
        num_edges_eval = batch_edge_index.shape[1]

        if batch_edge_index.numel() > 0:
            vals_eval = torch.ones(num_edges_eval, dtype=sparse_val_dtype_eval, device=device)
            batch_A = dglsp.spmatrix(
                batch_edge_index, val=vals_eval, shape=(num_nodes, num_nodes)).to(device)
        else:
            batch_A = dglsp.spmatrix(
                torch.empty((2, 0), dtype=torch.long, device=device),
                val=torch.empty((0,), dtype=sparse_val_dtype_eval, device=device),
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
    scaler = GradScaler(enabled=(device.type == 'cuda'))

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
            num_edges = batch_edge_index.shape[1]
            sparse_val_dtype = torch.float16 if scaler.is_enabled() else torch.float32

            if batch_edge_index.numel() > 0:
                vals = torch.ones(num_edges, dtype=sparse_val_dtype, device=device)
                batch_A = dglsp.spmatrix(
                    batch_edge_index, val=vals, shape=(num_nodes, num_nodes)).to(device)
            else:
                batch_A = dglsp.spmatrix(
                    torch.empty((2, 0), dtype=torch.long, device=device),
                    val=torch.empty((0,), dtype=sparse_val_dtype, device=device),
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

            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                batch_pred_logits_list = model(batch_A, batch_x_n, batch_abs_level,
                                               batch_rel_level, batch_A_n2g, batch_z_t,
                                               batch_t, query2g, num_query_cumsum, batch_y)
                loss = 0
                num_feature_dims = len(batch_pred_logits_list)
                if num_feature_dims > 0 and batch_pred_logits_list[0].shape[0] > 0:
                    for d_idx in range(num_feature_dims):
                        loss = loss + criterion(batch_pred_logits_list[d_idx], batch_z[:, d_idx])
                    loss /= num_feature_dims
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

            if loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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
        sparse_val_dtype_eval = torch.float32

        # Handle combined_edge_index for batch_A
        num_edges_combined = 0
        if batch_edge_index.numel() > 0:
            num_edges_combined += batch_edge_index.shape[1]
        if batch_noisy_edge_index.numel() > 0:
            num_edges_combined += batch_noisy_edge_index.shape[1]

        if batch_edge_index.numel() == 0 and batch_noisy_edge_index.numel() == 0:
            combined_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        elif batch_edge_index.numel() == 0:
            combined_edge_index = batch_noisy_edge_index
        elif batch_noisy_edge_index.numel() == 0:
            combined_edge_index = batch_edge_index
        else:
            combined_edge_index = torch.cat([batch_edge_index, batch_noisy_edge_index], dim=1)

        if combined_edge_index.numel() > 0:
            vals_eval = torch.ones(combined_edge_index.shape[1], dtype=sparse_val_dtype_eval, device=device)
            batch_A = dglsp.spmatrix(
                combined_edge_index, val=vals_eval, shape=(num_nodes, num_nodes)).to(device)
        else:
            batch_A = dglsp.spmatrix(
                torch.empty((2, 0), dtype=torch.long, device=device),
                val=torch.empty((0,), dtype=sparse_val_dtype_eval, device=device),
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
    scaler = GradScaler(enabled=(device.type == 'cuda'))

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
            sparse_val_dtype = torch.float16 if scaler.is_enabled() else torch.float32

            # Handle combined_edge_index for batch_A
            if batch_edge_index.numel() == 0 and batch_noisy_edge_index.numel() == 0:
                combined_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            elif batch_edge_index.numel() == 0:
                combined_edge_index = batch_noisy_edge_index
            elif batch_noisy_edge_index.numel() == 0:
                combined_edge_index = batch_edge_index
            else:
                combined_edge_index = torch.cat([batch_edge_index, batch_noisy_edge_index], dim=1)

            num_edges_combined = combined_edge_index.shape[1]

            if combined_edge_index.numel() > 0:
                vals = torch.ones(num_edges_combined, dtype=sparse_val_dtype, device=device)
                batch_A = dglsp.spmatrix(
                    combined_edge_index, val=vals, shape=(num_nodes, num_nodes)).to(device)
            else:
                batch_A = dglsp.spmatrix(
                    torch.empty((2, 0), dtype=torch.long, device=device),
                    val=torch.empty((0,), dtype=sparse_val_dtype, device=device),
                    shape=(num_nodes, num_nodes)).to(device)

            batch_x_n = batch_x_n.to(device)
            batch_abs_level = batch_abs_level.to(device)
            batch_rel_level = batch_rel_level.to(device)
            batch_t = batch_t.to(device)
            batch_query_src = batch_query_src.to(device)
            batch_query_dst = batch_query_dst.to(device)
            batch_label = batch_label.to(device)

            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                batch_pred_logits = model(batch_A, batch_x_n, batch_abs_level,
                                          batch_rel_level, batch_t, batch_query_src,
                                          batch_query_dst, batch_y)
                if batch_pred_logits.shape[0] > 0:
                    loss = criterion(batch_pred_logits, batch_label)
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

            if batch_pred_logits.shape[0] > 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
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
    config_df = pd.json_normalize(config, sep='/')
    ts = time.strftime('%b%d-%H%M%S', time.gmtime())
    cache_dir = "dataset_cache"
    os.makedirs(cache_dir, exist_ok=True)

    wandb.init(
        project=f'LayerDAG_{dataset_name}',
        name=f'{ts}_{dataset_name}',
        config=config_df.to_dict(orient='records')[0]
    )

    is_conditional = config['general']['conditional']
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
    print("Preparing LayerDAG-specific datasets (with caching)...")

    train_node_count_dataset_path = os.path.join(cache_dir, f"train_node_count_{dataset_name}_{is_conditional}.pt")
    if os.path.exists(train_node_count_dataset_path) and not args.force_preprocess:
        print(f"Loading cached train_node_count_dataset from {train_node_count_dataset_path}...")
        train_node_count_dataset = torch.load(train_node_count_dataset_path)
    else:
        print("Preprocessing train_node_count_dataset...")
        train_node_count_dataset = LayerDAGNodeCountDataset(train_set, conditional=is_conditional)
        print(f"Saving train_node_count_dataset to {train_node_count_dataset_path}...")
        torch.save(train_node_count_dataset, train_node_count_dataset_path)

    val_node_count_dataset_path = os.path.join(cache_dir, f"val_node_count_{dataset_name}_{is_conditional}.pt")
    if os.path.exists(val_node_count_dataset_path) and not args.force_preprocess:
        print(f"Loading cached val_node_count_dataset from {val_node_count_dataset_path}...")
        val_node_count_dataset = torch.load(val_node_count_dataset_path)
    else:
        print("Preprocessing val_node_count_dataset...")
        val_node_count_dataset = LayerDAGNodeCountDataset(val_set, conditional=is_conditional)
        print(f"Saving val_node_count_dataset to {val_node_count_dataset_path}...")
        torch.save(val_node_count_dataset, val_node_count_dataset_path)
    print(f"Node count dataset: Max layer size from train_set: {train_node_count_dataset.max_layer_size}")

    train_node_pred_dataset_path = os.path.join(cache_dir, f"train_node_pred_{dataset_name}_{is_conditional}.pt")
    if os.path.exists(train_node_pred_dataset_path) and not args.force_preprocess:
        print(f"Loading cached train_node_pred_dataset from {train_node_pred_dataset_path}...")
        train_node_pred_dataset = torch.load(train_node_pred_dataset_path)
    else:
        print("Preprocessing train_node_pred_dataset...")
        train_node_pred_dataset = LayerDAGNodePredDataset(train_set, conditional=is_conditional, get_marginal=True)
        print(f"Saving train_node_pred_dataset to {train_node_pred_dataset_path}...")
        torch.save(train_node_pred_dataset, train_node_pred_dataset_path)

    val_node_pred_dataset_path = os.path.join(cache_dir, f"val_node_pred_{dataset_name}_{is_conditional}.pt")
    if os.path.exists(val_node_pred_dataset_path) and not args.force_preprocess:
        print(f"Loading cached val_node_pred_dataset from {val_node_pred_dataset_path}...")
        val_node_pred_dataset = torch.load(val_node_pred_dataset_path)
    else:
        print("Preprocessing val_node_pred_dataset...")
        val_node_pred_dataset = LayerDAGNodePredDataset(val_set, conditional=is_conditional, get_marginal=False)
        print(f"Saving val_node_pred_dataset to {val_node_pred_dataset_path}...")
        torch.save(val_node_pred_dataset, val_node_pred_dataset_path)

    if not hasattr(train_node_pred_dataset, 'x_n_marginal') or not train_node_pred_dataset.x_n_marginal:
        if hasattr(train_node_pred_dataset, 'input_x_n') and train_node_pred_dataset.input_x_n.numel() > 0:
            print("Recomputing x_n_marginal for cached train_node_pred_dataset as it was missing...")
            input_x_n_for_marginal = train_node_pred_dataset.input_x_n
            if input_x_n_for_marginal.ndim == 1:
                input_x_n_for_marginal = input_x_n_for_marginal.unsqueeze(-1)
            num_feats = input_x_n_for_marginal.shape[-1]
            x_n_marginal = []
            num_actual_categories_per_feat = [train_set.num_categories] * num_feats
            dummy_category_val = train_set.dummy_category
            for f_idx in range(num_feats):
                input_x_n_f = input_x_n_for_marginal[:, f_idx]
                actual_nodes_mask = (input_x_n_f != dummy_category_val)
                actual_nodes_input_x_n_f = input_x_n_f[actual_nodes_mask]
                num_actual_types_this_feat = num_actual_categories_per_feat[f_idx]
                marginal_f = torch.zeros(num_actual_types_this_feat)
                if actual_nodes_input_x_n_f.numel() > 0:
                    unique_actual_vals, counts_actual_vals = torch.unique(actual_nodes_input_x_n_f, return_counts=True)
                    for val_idx, val_actual in enumerate(unique_actual_vals):
                        if 0 <= val_actual.item() < num_actual_types_this_feat:
                            marginal_f[val_actual.item()] += counts_actual_vals[val_idx].item()
                if marginal_f.sum() > 0:
                    marginal_f /= marginal_f.sum()
                else:
                    marginal_f.fill_(1.0 / num_actual_types_this_feat if num_actual_types_this_feat > 0 else 0)
                x_n_marginal.append(marginal_f)
            train_node_pred_dataset.x_n_marginal = x_n_marginal
            if not train_node_pred_dataset.x_n_marginal:
                raise ValueError("x_n_marginal is still empty for train_node_pred_dataset after attempting recompute.")
        else:
            raise ValueError(
                "x_n_marginal is empty or not set in train_node_pred_dataset, and cannot recompute from cache.")

    node_diffusion_config = {'marginal_list': train_node_pred_dataset.x_n_marginal, 'T': config['node_pred']['T']}
    node_diffusion = DiscreteDiffusion(**node_diffusion_config)
    train_node_pred_dataset.node_diffusion = node_diffusion
    val_node_pred_dataset.node_diffusion = node_diffusion

    train_edge_pred_dataset_path = os.path.join(cache_dir, f"train_edge_pred_{dataset_name}_{is_conditional}.pt")
    if os.path.exists(train_edge_pred_dataset_path) and not args.force_preprocess:
        print(f"Loading cached train_edge_pred_dataset from {train_edge_pred_dataset_path}...")
        train_edge_pred_dataset = torch.load(train_edge_pred_dataset_path)
    else:
        print("Preprocessing train_edge_pred_dataset...")
        train_edge_pred_dataset = LayerDAGEdgePredDataset(train_set, conditional=is_conditional)
        print(f"Saving train_edge_pred_dataset to {train_edge_pred_dataset_path}...")
        torch.save(train_edge_pred_dataset, train_edge_pred_dataset_path)

    val_edge_pred_dataset_path = os.path.join(cache_dir, f"val_edge_pred_{dataset_name}_{is_conditional}.pt")
    if os.path.exists(val_edge_pred_dataset_path) and not args.force_preprocess:
        print(f"Loading cached val_edge_pred_dataset from {val_edge_pred_dataset_path}...")
        val_edge_pred_dataset = torch.load(val_edge_pred_dataset_path)
    else:
        print("Preprocessing val_edge_pred_dataset...")
        val_edge_pred_dataset = LayerDAGEdgePredDataset(val_set, conditional=is_conditional)
        print(f"Saving val_edge_pred_dataset to {val_edge_pred_dataset_path}...")
        torch.save(val_edge_pred_dataset, val_edge_pred_dataset_path)

    edge_diffusion_config = {'avg_in_deg': train_edge_pred_dataset.avg_in_deg, 'T': config['edge_pred']['T']}
    edge_diffusion = EdgeDiscreteDiffusion(**edge_diffusion_config)
    train_edge_pred_dataset.edge_diffusion = edge_diffusion
    val_edge_pred_dataset.edge_diffusion = edge_diffusion
    print("Finished preparing LayerDAG-specific datasets.")

    print("Creating DataLoaders...")
    node_count_train_loader = DataLoader(train_node_count_dataset,
                                         batch_size=config['node_count']['loader']['batch_size'], shuffle=True,
                                         num_workers=config['node_count']['loader']['num_workers'],
                                         collate_fn=collate_node_count,
                                         pin_memory=True if device.type == "cuda" else False)
    node_count_val_loader = DataLoader(val_node_count_dataset, batch_size=config['node_count']['loader']['batch_size'],
                                       shuffle=False, num_workers=config['node_count']['loader']['num_workers'],
                                       collate_fn=collate_node_count,
                                       pin_memory=True if device.type == "cuda" else False)
    node_pred_train_loader = DataLoader(train_node_pred_dataset, batch_size=config['node_pred']['loader']['batch_size'],
                                        shuffle=True, num_workers=config['node_pred']['loader']['num_workers'],
                                        collate_fn=collate_node_pred,
                                        pin_memory=True if device.type == "cuda" else False)
    node_pred_val_loader = DataLoader(val_node_pred_dataset, batch_size=config['node_pred']['loader']['batch_size'],
                                      shuffle=False, num_workers=config['node_pred']['loader']['num_workers'],
                                      collate_fn=collate_node_pred, pin_memory=True if device.type == "cuda" else False)
    edge_pred_train_loader = DataLoader(train_edge_pred_dataset, batch_size=config['edge_pred']['loader']['batch_size'],
                                        shuffle=True, num_workers=config['edge_pred']['loader']['num_workers'],
                                        collate_fn=collate_edge_pred,
                                        pin_memory=True if device.type == "cuda" else False)
    edge_pred_val_loader = DataLoader(val_edge_pred_dataset, batch_size=config['edge_pred']['loader']['batch_size'],
                                      shuffle=False, num_workers=config['edge_pred']['loader']['num_workers'],
                                      collate_fn=collate_edge_pred, pin_memory=True if device.type == "cuda" else False)
    print("DataLoaders created.")

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
    model = LayerDAG(device=device, node_diffusion=node_diffusion, edge_diffusion=edge_diffusion,
                     is_model_conditional=is_conditional, **model_config)
    model.to(device)
    print("LayerDAG model initialized and moved to device.")

    patience_val = config['general'].get('patience', 10)

    print("\n--- Training Node Count Model ---")
    node_count_state_dict = main_node_count(device, node_count_train_loader, node_count_val_loader,
                                            model.node_count_model, config['node_count'], patience_val, is_conditional)
    model.node_count_model.load_state_dict(node_count_state_dict)
    print("--- Finished Training Node Count Model ---\n")

    print("--- Training Node Prediction Model ---")
    node_pred_state_dict = main_node_pred(device, node_pred_train_loader, node_pred_val_loader, model.node_pred_model,
                                          config['node_pred'], patience_val, is_conditional)
    model.node_pred_model.load_state_dict(node_pred_state_dict)
    print("--- Finished Training Node Prediction Model ---\n")

    print("--- Training Edge Prediction Model ---")
    edge_pred_state_dict = main_edge_pred(device, edge_pred_train_loader, edge_pred_val_loader, model.edge_pred_model,
                                          config['edge_pred'], patience_val, is_conditional)
    model.edge_pred_model.load_state_dict(edge_pred_state_dict)
    print("--- Finished Training Edge Prediction Model ---\n")

    save_path = f'model_{dataset_name}_{ts}.pth'
    torch.save({'dataset': dataset_name, 'node_diffusion_config': node_diffusion_config,
                'edge_diffusion_config': edge_diffusion_config, 'model_config': model_config,
                'is_conditional': is_conditional, 'model_state_dict': model.state_dict()}, save_path)
    print(f"Model saved to {save_path}")
    wandb.save(save_path)
    wandb.finish()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Train LayerDAG model.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of CPU threads for PyTorch (DGL recommendation is often 1 for GNNs to avoid overhead).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--force_preprocess", action='store_true',
                        help="Force preprocessing of datasets even if cache files exist.")
    args = parser.parse_args()
    main(args)
