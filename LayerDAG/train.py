# train.py
import dgl.sparse as dglsp
import pandas as pd
import time
import torch
import torch.nn as nn
import wandb  # For logging

from copy import deepcopy
from torch.utils.data import DataLoader  # Ensure DataLoader is imported
from tqdm import tqdm

from setup_utils import set_seed, load_yaml
# Ensure your dataset components are correctly imported
# This structure assumes your __init__.py in src/dataset correctly exports these
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
        batch_A = dglsp.spmatrix(
            batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
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
            batch_A = dglsp.spmatrix(batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
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
            print(f"Node Count: New best val_acc: {best_val_acc:.4f} (NLL: {best_val_nll:.4f}) at epoch {epoch + 1}")
        elif val_acc == best_val_acc and val_nll < best_val_nll:  # If accuracy is same, check NLL
            best_val_nll = val_nll
            best_state_dict = deepcopy(model.state_dict())
            num_patient_epochs = 0  # Reset patience as NLL improved for same accuracy
            print(f"Node Count: New best val_nll: {best_val_nll:.4f} (Acc: {best_val_acc:.4f}) at epoch {epoch + 1}")
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
    total_num_attribute_predictions = 0  # Changed from total_count to be more descriptive
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
        batch_A = dglsp.spmatrix(
            batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
        batch_x_n = batch_x_n.to(device)
        batch_abs_level = batch_abs_level.to(device)
        batch_rel_level = batch_rel_level.to(device)
        batch_A_n2g = dglsp.spmatrix(
            batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
        batch_z_t = batch_z_t.to(device)
        batch_t = batch_t.to(device)
        query2g = query2g.to(device)
        num_query_cumsum = num_query_cumsum.to(device)
        batch_z = batch_z.to(device)  # Ground truth node attributes

        batch_logits_list = model(batch_A, batch_x_n, batch_abs_level,
                                  batch_rel_level, batch_A_n2g, batch_z_t, batch_t,
                                  query2g, num_query_cumsum, batch_y)

        num_feature_dims = len(batch_logits_list)
        current_batch_num_queries = batch_logits_list[0].shape[0]

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
    Args:
        device: The device to run training on.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        model: The node prediction model.
        config: Configuration dictionary for node prediction training.
        patience (int): Number of epochs to wait for improvement before early stopping.
        is_conditional (bool): Flag indicating if the model is conditional.
    Returns:
        The state dictionary of the best performing model.
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
            batch_A = dglsp.spmatrix(
                batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
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
            for d in range(num_feature_dims):
                loss = loss + criterion(batch_pred_logits_list[d], batch_z[:, d])
            if num_feature_dims > 0:
                loss /= num_feature_dims

            optimizer.zero_grad()
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
            print(f"Node Pred: New best val_nll: {best_val_nll:.4f} at epoch {epoch + 1}")
        else:
            num_patient_epochs += 1

        wandb.log({
            'node_pred/best_val_nll': best_val_nll,
            'node_pred/num_patient_epochs': num_patient_epochs
        })

        if (patience is not None) and (num_patient_epochs >= patience):
            print(f"Node Pred: Early stopping after {patience} epochs without improvement.")
            break

    print(f"Node Pred: Best validation NLL: {best_val_nll:.4f}")
    return best_state_dict


@torch.no_grad()
def eval_edge_pred(device, val_loader, model, is_conditional):
    """
    Evaluates the edge prediction model.
    Args:
        device: The device to run evaluation on.
        val_loader: DataLoader for the validation set.
        model: The edge prediction model.
        is_conditional (bool): Flag indicating if the model is conditional.
    Returns:
        Average NLL.
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
        combined_edge_index = torch.cat([batch_edge_index, batch_noisy_edge_index], dim=1)
        batch_A = dglsp.spmatrix(
            combined_edge_index,
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

        batch_nll = -batch_logits.log_softmax(dim=-1)
        current_batch_num_queries = batch_logits.shape[0]
        batch_nll = batch_nll[
            torch.arange(current_batch_num_queries, device=device), batch_label]
        total_nll += batch_nll.sum().item()
        total_queries += current_batch_num_queries

    return total_nll / total_queries if total_queries > 0 else float('inf')


def main_edge_pred(device, train_loader, val_loader, model, config, patience, is_conditional):
    """
    Main training loop for the edge prediction model.
    Args:
        device: The device to run training on.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        model: The edge prediction model.
        config: Configuration dictionary for edge prediction training.
        patience (int): Number of epochs to wait for improvement before early stopping.
        is_conditional (bool): Flag indicating if the model is conditional.
    Returns:
        The state dictionary of the best performing model.
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
            combined_edge_index = torch.cat([batch_edge_index, batch_noisy_edge_index], dim=1)
            batch_A = dglsp.spmatrix(
                combined_edge_index,
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
            loss = criterion(batch_pred_logits, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'edge_pred/loss': loss.item()})

        val_nll = eval_edge_pred(device, val_loader, model, is_conditional)
        wandb.log({
            'edge_pred/epoch': epoch,
            'edge_pred/val_nll': val_nll,
        })

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state_dict = deepcopy(model.state_dict())
            num_patient_epochs = 0
            print(f"Edge Pred: New best val_nll: {best_val_nll:.4f} at epoch {epoch + 1}")
        else:
            num_patient_epochs += 1

        wandb.log({
            'edge_pred/best_val_nll': best_val_nll,
            'edge_pred/num_patient_epochs': num_patient_epochs
        })

        if (patience is not None) and (num_patient_epochs >= patience):
            print(f"Edge Pred: Early stopping after {patience} epochs without improvement.")
            break

    print(f"Edge Pred: Best validation NLL: {best_val_nll:.4f}")
    return best_state_dict


def main(args):
    torch.set_num_threads(args.num_threads)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    set_seed(args.seed)

    config = load_yaml(args.config_file)
    print("--- Debugging Config ---")
    if config is None:
        print("ERROR: Config is None. YAML file might not have been loaded correctly.")
        exit(1)
    print(f"Full config object type: {type(config)}")
    # print(f"Full config content: {config}") # Can be very verbose
    if 'general' in config:
        print(f"config['general'] type: {type(config['general'])}")
        print(f"config['general'] content: {config['general']}")
        print(f"Keys in config['general']: {config['general'].keys()}")
    else:
        print("ERROR: 'general' key not found in config.")
        exit(1)  # Exit if 'general' key is missing
    print("--- End Debugging Config ---")

    dataset_name = config['general']['dataset']
    config_df = pd.json_normalize(config, sep='/')

    ts = time.strftime('%b%d-%H%M%S', time.gmtime())

    wandb.init(
        project=f'LayerDAG_{dataset_name}',
        name=f'{ts}_{dataset_name}',
        config=config_df.to_dict(orient='records')[0]
    )

    is_conditional = config['general']['conditional']

    load_dataset_kwargs = {'conditional': is_conditional}
    if dataset_name == 'aig':
        load_dataset_kwargs['path'] = config['general']['path_to_pt_file']
        load_dataset_kwargs['num_node_categories'] = config['general']['num_node_categories']
    elif dataset_name == 'tpu_tile':
        pass  # tpu_tile specific args if any
    else:
        raise ValueError(f"Unsupported dataset name in config: {dataset_name}")

    print(f"Loading dataset: {dataset_name} (Conditional: {is_conditional})")
    train_set, val_set, _ = load_dataset(dataset_name, **load_dataset_kwargs)
    print(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}")

    # Create LayerDAG specific datasets
    print("Preparing LayerDAG-specific datasets...")
    train_node_count_dataset = LayerDAGNodeCountDataset(train_set, conditional=is_conditional)
    val_node_count_dataset = LayerDAGNodeCountDataset(val_set, conditional=is_conditional)
    print(f"Node count dataset: Max layer size from train_set: {train_node_count_dataset.max_layer_size}")

    train_node_pred_dataset = LayerDAGNodePredDataset(train_set, conditional=is_conditional, get_marginal=True)
    val_node_pred_dataset = LayerDAGNodePredDataset(val_set, conditional=is_conditional, get_marginal=False)

    if not train_node_pred_dataset.x_n_marginal:
        raise ValueError("x_n_marginal is empty in train_node_pred_dataset. Check dataset processing.")
    print(
        f"Node prediction dataset: Marginal distribution computed. Max level from train: {train_node_pred_dataset.input_level.max().item()}")

    node_diffusion_config = {
        'marginal_list': train_node_pred_dataset.x_n_marginal,
        'T': config['node_pred']['T']
    }
    node_diffusion = DiscreteDiffusion(**node_diffusion_config)
    train_node_pred_dataset.node_diffusion = node_diffusion
    val_node_pred_dataset.node_diffusion = node_diffusion

    train_edge_pred_dataset = LayerDAGEdgePredDataset(train_set, conditional=is_conditional)
    val_edge_pred_dataset = LayerDAGEdgePredDataset(val_set, conditional=is_conditional)
    print(f"Edge prediction dataset: Avg in-degree from train: {train_edge_pred_dataset.avg_in_deg:.4f}")

    edge_diffusion_config = {
        'avg_in_deg': train_edge_pred_dataset.avg_in_deg,
        'T': config['edge_pred']['T']
    }
    edge_diffusion = EdgeDiscreteDiffusion(**edge_diffusion_config)
    train_edge_pred_dataset.edge_diffusion = edge_diffusion
    val_edge_pred_dataset.edge_diffusion = edge_diffusion

    # --- Create DataLoaders ---
    print("Creating DataLoaders...")
    node_count_train_loader = DataLoader(
        train_node_count_dataset,
        batch_size=config['node_count']['loader']['batch_size'],
        shuffle=True,
        num_workers=config['node_count']['loader']['num_workers'],
        collate_fn=collate_node_count,
        pin_memory=True if device_str == "cuda:0" else False  # Optional: for faster data transfer to GPU
    )
    node_count_val_loader = DataLoader(
        val_node_count_dataset,
        batch_size=config['node_count']['loader']['batch_size'],
        shuffle=False,
        num_workers=config['node_count']['loader']['num_workers'],
        collate_fn=collate_node_count,
        pin_memory=True if device_str == "cuda:0" else False
    )

    node_pred_train_loader = DataLoader(
        train_node_pred_dataset,
        batch_size=config['node_pred']['loader']['batch_size'],
        shuffle=True,
        num_workers=config['node_pred']['loader']['num_workers'],
        collate_fn=collate_node_pred,
        pin_memory=True if device_str == "cuda:0" else False
    )
    node_pred_val_loader = DataLoader(
        val_node_pred_dataset,
        batch_size=config['node_pred']['loader']['batch_size'],
        shuffle=False,
        num_workers=config['node_pred']['loader']['num_workers'],
        collate_fn=collate_node_pred,
        pin_memory=True if device_str == "cuda:0" else False
    )

    edge_pred_train_loader = DataLoader(
        train_edge_pred_dataset,
        batch_size=config['edge_pred']['loader']['batch_size'],
        shuffle=True,
        num_workers=config['edge_pred']['loader']['num_workers'],
        collate_fn=collate_edge_pred,
        pin_memory=True if device_str == "cuda:0" else False
    )
    edge_pred_val_loader = DataLoader(
        val_edge_pred_dataset,
        batch_size=config['edge_pred']['loader']['batch_size'],
        shuffle=False,
        num_workers=config['edge_pred']['loader']['num_workers'],
        collate_fn=collate_edge_pred,
        pin_memory=True if device_str == "cuda:0" else False
    )
    print("DataLoaders created.")

    model_num_x_n_cat = train_set.num_categories
    print(f"Model config: num_x_n_cat (node types including dummy) = {model_num_x_n_cat}")

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

    print("Initializing LayerDAG model...")
    model = LayerDAG(device=device,
                     node_diffusion=node_diffusion,
                     edge_diffusion=edge_diffusion,
                     is_conditional=is_conditional,
                     **model_config)
    model.to(device)
    print("LayerDAG model initialized and moved to device.")

    patience_val = config['general'].get('patience', 10)

    print("\n--- Training Node Count Model ---")
    node_count_state_dict = main_node_count(
        device,
        node_count_train_loader,  # Pass DataLoader
        node_count_val_loader,  # Pass DataLoader
        model.node_count_model, config['node_count'], patience_val, is_conditional)
    model.node_count_model.load_state_dict(node_count_state_dict)
    print("--- Finished Training Node Count Model ---\n")

    print("--- Training Node Prediction Model ---")
    node_pred_state_dict = main_node_pred(
        device,
        node_pred_train_loader,  # Pass DataLoader
        node_pred_val_loader,  # Pass DataLoader
        model.node_pred_model, config['node_pred'], patience_val, is_conditional)
    model.node_pred_model.load_state_dict(node_pred_state_dict)
    print("--- Finished Training Node Prediction Model ---\n")

    print("--- Training Edge Prediction Model ---")
    edge_pred_state_dict = main_edge_pred(
        device,
        edge_pred_train_loader,  # Pass DataLoader
        edge_pred_val_loader,  # Pass DataLoader
        model.edge_pred_model, config['edge_pred'], patience_val, is_conditional)
    model.edge_pred_model.load_state_dict(edge_pred_state_dict)
    print("--- Finished Training Edge Prediction Model ---\n")

    save_path = f'model_{dataset_name}_{ts}.pth'
    torch.save({
        'dataset': dataset_name,
        'node_diffusion_config': node_diffusion_config,
        'edge_diffusion_config': edge_diffusion_config,
        'model_config': model_config,
        'is_conditional': is_conditional,
        'model_state_dict': model.state_dict()
    }, save_path)
    print(f"Model saved to {save_path}")
    wandb.save(save_path)
    wandb.finish()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Train LayerDAG model.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--num_threads", type=int, default=16, help="Number of CPU threads for PyTorch.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    main(args)
