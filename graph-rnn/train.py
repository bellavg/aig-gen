import argparse
import yaml
import torch
import os
import time
import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# Import the new DirectedGraphDataSet
# Ensure extension_data_updated.py is the correct filename if you saved my previous changes there
from extension_data import DirectedGraphDataSet
# Removed: from data import GraphDataSet

# Make sure this model.py is the one with node attribute prediction
from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP


def train_mlp_step(graph_rnn, edge_mlp, data_batch, criterion_edges, optim_graph_rnn, optim_edge_mlp,
                   scheduler_graph_rnn, scheduler_mlp, device, use_edge_features,
                   criterion_node_attr=None, node_attribute_loss_weight=0.0):
    """
    Train GraphRNN with MLP edge model.
    Adapted for data format from DirectedGraphDataSet.
    """
    graph_rnn.train()
    edge_mlp.train()
    graph_rnn.zero_grad()
    edge_mlp.zero_grad()

    s = data_batch['x_adj'].float().to(device)
    node_attrs_target = data_batch['node_attr_onehot'].float().to(device)
    lens = data_batch['len'].cpu()

    num_node_classes = getattr(graph_rnn, 'num_node_classes', 0)
    batch_size = s.shape[0]
    m_window_size = s.shape[2]
    edge_feature_len = s.shape[3]

    sos_adj_frame = torch.ones([batch_size, 1, m_window_size, edge_feature_len], device=device)
    x_adj_input = torch.cat((sos_adj_frame, s[:, :-1, :, :]), dim=1)

    sos_attr_frame = torch.zeros([batch_size, 1, num_node_classes], device=device)
    prev_node_attr_input = torch.cat((sos_attr_frame, node_attrs_target[:, :-1, :]), dim=1)

    target_node_attr_s = node_attrs_target
    y_adj_target = s
    current_lens = lens

    graph_rnn.reset_hidden()
    hidden_for_mlp, node_attribute_logits = graph_rnn(x_adj_input, prev_node_attr_input, current_lens)
    y_edge_pred = edge_mlp(hidden_for_mlp, return_logits=use_edge_features)

    packed_y_edge_pred = pack_padded_sequence(y_edge_pred, current_lens, batch_first=True, enforce_sorted=False).data
    packed_y_adj_target = pack_padded_sequence(y_adj_target, current_lens, batch_first=True, enforce_sorted=False).data

    if use_edge_features:
        loss_edges = criterion_edges(
            packed_y_edge_pred.reshape(-1, edge_feature_len),
            torch.argmax(packed_y_adj_target.reshape(-1, edge_feature_len), dim=-1)
        )
    else:
        loss_edges = criterion_edges(
            packed_y_edge_pred.reshape(-1, 1),
            packed_y_adj_target.reshape(-1, 1)
        )

    total_loss = loss_edges
    loss_node_attributes_val = 0.0

    if node_attribute_logits is not None and criterion_node_attr and num_node_classes > 0:
        packed_node_attr_logits = pack_padded_sequence(node_attribute_logits, current_lens, batch_first=True,
                                                       enforce_sorted=False).data
        packed_target_node_attr_onehot = pack_padded_sequence(target_node_attr_s, current_lens, batch_first=True,
                                                              enforce_sorted=False).data
        packed_target_node_attr_indices = torch.argmax(packed_target_node_attr_onehot, dim=-1)

        loss_node_attributes = criterion_node_attr(packed_node_attr_logits, packed_target_node_attr_indices)
        total_loss = total_loss + node_attribute_loss_weight * loss_node_attributes
        loss_node_attributes_val = loss_node_attributes.item()

    total_loss.backward()
    optim_graph_rnn.step()
    optim_edge_mlp.step()

    return total_loss.item(), loss_edges.item(), loss_node_attributes_val


def train_rnn_step(graph_rnn, edge_rnn, data_batch,
                   criterion_edges, criterion_node_attr,
                   optim_graph_rnn, optim_edge_model,
                   scheduler_graph_rnn, scheduler_edge_model,
                   device, use_edge_features, num_node_classes,  # num_node_classes here is from config
                   node_attribute_loss_weight):
    graph_rnn.train()
    edge_rnn.train()
    graph_rnn.zero_grad()
    edge_rnn.zero_grad()

    x_adj_seq = data_batch['x_adj'].float().to(device)
    node_attr_seq = data_batch['node_attr_onehot'].float().to(device)
    lens = data_batch['len'].cpu()

    batch_size = x_adj_seq.shape[0]
    m_adj_len = x_adj_seq.shape[2]
    edge_feature_len = x_adj_seq.shape[3]

    sos_adj_frame = torch.ones([batch_size, 1, m_adj_len, edge_feature_len], device=device)
    x_adj_for_graph_rnn_input = torch.cat((sos_adj_frame, x_adj_seq[:, :-1, :, :]), dim=1)

    sos_attr_frame = torch.zeros([batch_size, 1, num_node_classes], device=device)
    prev_node_attr_for_graph_rnn_input = torch.cat((sos_attr_frame, node_attr_seq[:, :-1, :]), dim=1)

    graph_rnn.reset_hidden()
    h_for_edge_rnn, pred_node_attr_logits = graph_rnn(x_adj_for_graph_rnn_input,
                                                      prev_node_attr_for_graph_rnn_input,
                                                      lens)

    packed_pred_node_attr_logits = pack_padded_sequence(pred_node_attr_logits, lens, batch_first=True,
                                                        enforce_sorted=False).data
    packed_target_node_attr_onehot = pack_padded_sequence(node_attr_seq, lens, batch_first=True,
                                                          enforce_sorted=False).data
    packed_target_node_attr_indices = torch.argmax(packed_target_node_attr_onehot, dim=-1)
    loss_node_attributes = criterion_node_attr(packed_pred_node_attr_logits, packed_target_node_attr_indices)

    hidden_packed_for_edge_rnn = pack_padded_sequence(h_for_edge_rnn, lens, batch_first=True, enforce_sorted=False).data
    edge_rnn.set_first_layer_hidden(hidden_packed_for_edge_rnn)

    packed_x_adj_seq_data = pack_padded_sequence(x_adj_seq, lens, batch_first=True, enforce_sorted=False).data
    sos_edge_frame_for_packed = torch.ones([packed_x_adj_seq_data.shape[0], 1, edge_feature_len], device=device)
    x_edge_rnn_input_packed = torch.cat((sos_edge_frame_for_packed, packed_x_adj_seq_data[:, :-1, :]), dim=1)
    y_edge_rnn_target_packed = packed_x_adj_seq_data

    m_edge_pred_window = graph_rnn.input_size  # This is M

    edge_seq_lens_for_packing = []
    for l_nodes_in_graph in lens:  # lens contains actual number of nodes for each graph in batch
        for i_node_idx_in_graph in range(l_nodes_in_graph.item()):  # Iterate 0 to num_nodes-1 for current graph
            # For node i_node_idx_in_graph, it has min(i_node_idx_in_graph, M) predecessors in M-window
            # The S_i vector for node i_node_idx_in_graph has min(i_node_idx_in_graph, M) actual entries.
            edge_seq_lens_for_packing.append(min(i_node_idx_in_graph, m_edge_pred_window))

    y_edge_rnn_pred = edge_rnn(x_edge_rnn_input_packed,
                               torch.tensor(edge_seq_lens_for_packing, device=device, dtype=torch.long),
                               return_logits=use_edge_features)

    valid_preds_list = []
    valid_targets_list = []
    current_pos_in_packed_data = 0  # Keep track of where we are in the packed (concatenated) data

    # Iterate through each graph in the batch
    for graph_idx_in_batch in range(batch_size):
        num_nodes_this_graph = lens[graph_idx_in_batch].item()

        # Get the predictions and targets for all nodes of the current graph from the packed batch
        # y_edge_rnn_pred and y_edge_rnn_target_packed are sequences of [num_edges_in_m_window, edge_feature_len]
        # for each node, concatenated across all nodes in the batch.

        # Identify the start and end index for the current graph's nodes in the packed data
        # The number of nodes processed so far for previous graphs in the batch:
        # current_pos_in_packed_data is the sum of (nodes in graph_0 + nodes in graph_1 + ... + nodes in graph_{graph_idx_in_batch-1})

        # Slice the predictions and targets for the current graph
        # The y_edge_rnn_pred is already structured such that the first lens[0] items are for graph 0,
        # next lens[1] items for graph 1, etc.
        # So, current_pos_in_packed_data correctly tracks the start of each graph's node data.

        graph_preds_all_nodes = y_edge_rnn_pred[
                                current_pos_in_packed_data: current_pos_in_packed_data + num_nodes_this_graph]
        graph_targets_all_nodes = y_edge_rnn_target_packed[
                                  current_pos_in_packed_data: current_pos_in_packed_data + num_nodes_this_graph]

        # For each node in the current graph
        for node_j_in_graph in range(num_nodes_this_graph):
            # Number of valid edge predictions for this specific node_j_in_graph
            # This is min(node_j_in_graph, M)
            actual_len_edges_for_this_node = min(node_j_in_graph, m_edge_pred_window)

            if actual_len_edges_for_this_node > 0:
                # Get the M-window predictions for this node
                node_preds_m_window = graph_preds_all_nodes[node_j_in_graph]  # Shape [M, edge_feature_len]
                # Get the M-window targets for this node
                node_targets_m_window = graph_targets_all_nodes[node_j_in_graph]  # Shape [M, edge_feature_len]

                # Slice to get only the valid part of the M-window
                valid_preds_list.append(node_preds_m_window[:actual_len_edges_for_this_node, :])
                valid_targets_list.append(node_targets_m_window[:actual_len_edges_for_this_node, :])

        current_pos_in_packed_data += num_nodes_this_graph

    if not valid_preds_list:  # If all graphs had 0 or 1 node, or M=0
        loss_edges = torch.tensor(0.0, device=device, requires_grad=True)  # Ensure it's a tensor that requires grad
    else:
        y_edge_rnn_pred_for_loss = torch.cat(valid_preds_list, dim=0)
        y_edge_rnn_target_for_loss = torch.cat(valid_targets_list, dim=0)

        if use_edge_features:  # True if edge_feature_len > 1
            # Target for CrossEntropyLoss should be class indices
            y_edge_rnn_target_for_loss_indices = torch.argmax(y_edge_rnn_target_for_loss, dim=-1)
            loss_edges = criterion_edges(y_edge_rnn_pred_for_loss, y_edge_rnn_target_for_loss_indices)
        else:  # Binary edges
            loss_edges = criterion_edges(y_edge_rnn_pred_for_loss, y_edge_rnn_target_for_loss)

    total_loss = loss_edges + node_attribute_loss_weight * loss_node_attributes

    total_loss.backward()
    optim_graph_rnn.step()
    optim_edge_model.step()

    return total_loss.item(), loss_edges.item(), loss_node_attributes.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Path of the config file to use for training')
    parser.add_argument('-r', '--restore', dest='restore_path', required=False, default=None,
                        help='Checkpoint to continue training from')
    parser.add_argument('--gpu', dest='gpu_id', required=False, default=0, type=int,
                        help='Id of the GPU to use')
    args = parser.parse_args()

    base_path = os.path.dirname(args.config_file) if args.config_file and os.path.isfile(args.config_file) else '.'

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    checkpoint_dir = os.path.join(base_path, config['train']['checkpoint_dir'])
    log_dir = os.path.join(base_path, config['train']['log_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # These are the definitive values from config for model construction and data loading
    num_node_classes_for_model = config['model']['GraphRNN']['num_node_classes']
    edge_feature_len_for_model = config['model']['GraphRNN']['edge_feature_len']
    m_param = config['data']['m']

    if num_node_classes_for_model <= 0:
        raise ValueError("'num_node_classes' in config['model']['GraphRNN'] must be > 0.")
    if edge_feature_len_for_model <= 0:  # Should be 3 for (NO_EDGE, REG, INV)
        raise ValueError("'edge_feature_len' in config['model']['GraphRNN'] must be > 0.")

    if config['model']['edge_model'] == 'rnn':
        node_model = GraphLevelRNN(
            input_size=m_param,
            output_size=config['model']['EdgeRNN']['hidden_size'],
            **config['model']['GraphRNN']
        ).to(device)
        edge_model = EdgeLevelRNN(
            **config['model']['EdgeRNN']
        ).to(device)
        step_fn = train_rnn_step
    else:  # MLP edge model
        node_model = GraphLevelRNN(
            input_size=m_param,
            output_size=None,
            **config['model']['GraphRNN']
        ).to(device)
        edge_model = EdgeLevelMLP(
            input_size_from_graph_rnn=config['model']['GraphRNN']['hidden_size'],
            mlp_hidden_size=config['model']['EdgeMLP']['hidden_size'],
            num_edges_to_predict=m_param,
            edge_feature_len=edge_feature_len_for_model
        ).to(device)
        step_fn = train_mlp_step
        print("Warning: MLP edge model path is being used.")

    use_edge_features_for_loss = edge_feature_len_for_model > 1  # This will be true

    if use_edge_features_for_loss:
        criterion_edges = torch.nn.CrossEntropyLoss().to(device)
    else:  # Should not happen with edge_feature_len = 3
        criterion_edges = torch.nn.BCELoss().to(device)

    criterion_node_attr = torch.nn.CrossEntropyLoss().to(device)

    optim_node_model = torch.optim.Adam(list(node_model.parameters()), lr=config['train']['lr'])
    optim_edge_model = torch.optim.Adam(list(edge_model.parameters()), lr=config['train']['lr'])

    scheduler_node_model = MultiStepLR(optim_node_model,
                                       milestones=config['train']['lr_schedule_milestones'],
                                       gamma=config['train']['lr_schedule_gamma'])
    scheduler_edge_model = MultiStepLR(optim_edge_model,
                                       milestones=config['train']['lr_schedule_milestones'],
                                       gamma=config['train']['lr_schedule_gamma'])

    writer = SummaryWriter(log_dir)
    global_step = 0

    if args.restore_path:
        print("Restoring from checkpoint: {}".format(args.restore_path))
        state = torch.load(args.restore_path, map_location=device)
        global_step = state.get("global_step", 0)
        node_model.load_state_dict(state["node_model"])
        edge_model.load_state_dict(state["edge_model"])
        optim_node_model.load_state_dict(state["optim_node_model"])
        optim_edge_model.load_state_dict(state["optim_edge_model"])
        scheduler_node_model.load_state_dict(state["scheduler_node_model"])
        scheduler_edge_model.load_state_dict(state["scheduler_edge_model"])
        if "criterion_edges" in state:
            criterion_edges.load_state_dict(state["criterion_edges"])
        if "criterion_node_attr" in state and criterion_node_attr is not None:
            criterion_node_attr.load_state_dict(state["criterion_node_attr"])
        print(f"Restored to global_step: {global_step}")

    pyg_file_path = config['data'].get('pyg_file_path', 'aig_directed.pt')
    if not os.path.isabs(pyg_file_path) and base_path and base_path != '.':
        pyg_file_path = os.path.join(base_path, pyg_file_path)

    print(f"Attempting to load dataset from: {os.path.abspath(pyg_file_path)}")

    dataset = DirectedGraphDataSet(
        dataset_type=config['model']['mode'],  # e.g. 'aig-custom-topsort'
        m=m_param,
        pyg_file_path=pyg_file_path,
        num_node_features=num_node_classes_for_model,  # Should be 4
        num_edge_features=edge_feature_len_for_model,  # Should be 3
        training=True,
        train_split=config['data']['train_split']
    )

    data_loader = DataLoader(dataset,
                             batch_size=config['train']['batch_size'],
                             shuffle=True,  # Shuffle is important for training
                             num_workers=config['train'].get('num_workers', 0),
                             pin_memory=config['train'].get('pin_memory', False)
                             )

    print(f"Starting training from global_step: {global_step}")
    node_model.train()
    edge_model.train()

    done = False
    train_loss_sum_print_iter = 0
    edge_loss_sum_print_iter = 0
    node_attr_loss_sum_print_iter = 0
    start_time_print_iter = time.time()

    steps_per_epoch = len(data_loader) if len(data_loader) > 0 else 1
    total_epochs_approx = (config['train'][
                               'steps'] - global_step + steps_per_epoch - 1) // steps_per_epoch if steps_per_epoch > 0 else 0
    print(f"Approximate total epochs to run: {total_epochs_approx} ({steps_per_epoch} steps per epoch)")

    current_epoch = 0
    # Flag to ensure debug prints only happen once for the first batch of the first epoch
    first_batch_inspected = False

    while not done:
        current_epoch += 1
        print(f"Epoch {current_epoch}/{total_epochs_approx if total_epochs_approx > 0 else 'N/A'} ---")
        epoch_train_loss_sum = 0
        epoch_edge_loss_sum = 0
        epoch_node_attr_loss_sum = 0

        for batch_idx, data_batch in enumerate(data_loader):
            # DEBUG PRINTS for the first batch of the first epoch
            if not first_batch_inspected and current_epoch == 1 and batch_idx == 0:
                print("\n" + "=" * 20 + " DEBUG: Inspecting First Data Batch " + "=" * 20)
                if 'x_adj' in data_batch:
                    print(f"data_batch['x_adj'].shape: {data_batch['x_adj'].shape}")
                    if data_batch['x_adj'].numel() > 0:  # Check if tensor is not empty
                        # Print for first graph, first node's M-window, first 5 edge slots
                        print(
                            f"data_batch['x_adj'][0, 0, :5, :]:\n{data_batch['x_adj'][0, 0, :min(5, data_batch['x_adj'].shape[2]), :]}")
                    else:
                        print("data_batch['x_adj'] is empty.")
                else:
                    print("data_batch does not contain 'x_adj'")

                if 'node_attr_onehot' in data_batch:
                    print(f"data_batch['node_attr_onehot'].shape: {data_batch['node_attr_onehot'].shape}")
                    if data_batch['node_attr_onehot'].numel() > 0:
                        # Print for first graph, attributes for first 5 nodes
                        print(
                            f"data_batch['node_attr_onehot'][0, :5, :]:\n{data_batch['node_attr_onehot'][0, :min(5, data_batch['node_attr_onehot'].shape[1]), :]}")
                    else:
                        print("data_batch['node_attr_onehot'] is empty.")
                else:
                    print("data_batch does not contain 'node_attr_onehot'")

                if 'len' in data_batch and data_batch['len'].numel() > 0:
                    print(f"data_batch['len'][0] (length of first graph): {data_batch['len'][0].item()}")
                else:
                    print("data_batch does not contain 'len' or it's empty.")
                print("=" * 60 + "\n")
                first_batch_inspected = True

            if global_step >= config['train']['steps']:
                done = True
                break
            global_step += 1

            if step_fn == train_rnn_step:
                current_total_loss, current_edge_loss, current_node_attr_loss = step_fn(
                    node_model, edge_model, data_batch,
                    criterion_edges, criterion_node_attr,
                    optim_node_model, optim_edge_model,
                    scheduler_node_model, scheduler_edge_model,
                    device, use_edge_features_for_loss, num_node_classes_for_model,
                    config['train'].get('node_attribute_loss_weight', 1.0)
                )
            else:  # train_mlp_step
                current_total_loss, current_edge_loss, current_node_attr_loss = step_fn(
                    node_model, edge_model, data_batch, criterion_edges,
                    optim_node_model, optim_edge_model,
                    scheduler_node_model, scheduler_edge_model,
                    device, use_edge_features_for_loss,
                    criterion_node_attr, config['train'].get('node_attribute_loss_weight', 1.0)
                )

            train_loss_sum_print_iter += current_total_loss
            edge_loss_sum_print_iter += current_edge_loss
            node_attr_loss_sum_print_iter += current_node_attr_loss

            epoch_train_loss_sum += current_total_loss
            epoch_edge_loss_sum += current_edge_loss
            epoch_node_attr_loss_sum += current_node_attr_loss

            writer.add_scalar('loss_iter/total_loss', current_total_loss, global_step)
            writer.add_scalar('loss_iter/edge_loss', current_edge_loss, global_step)
            if num_node_classes_for_model > 0 and (
                    current_node_attr_loss > 0 or step_fn == train_rnn_step or step_fn == train_mlp_step):  # Ensure node loss is logged if applicable
                writer.add_scalar('loss_iter/node_attribute_loss', current_node_attr_loss, global_step)

            if global_step % config['train']['print_iter'] == 0:
                avg_total_loss_print = train_loss_sum_print_iter / config['train']['print_iter']
                avg_edge_loss_print = edge_loss_sum_print_iter / config['train']['print_iter']
                avg_node_attr_loss_print = node_attr_loss_sum_print_iter / config['train']['print_iter']

                elapsed_time_print_iter = time.time() - start_time_print_iter
                time_per_iter_print = elapsed_time_print_iter / config['train']['print_iter'] if config['train'][
                                                                                                     'print_iter'] > 0 else 0
                eta_seconds = (config['train'][
                                   'steps'] - global_step) * time_per_iter_print if time_per_iter_print > 0 else 0

                log_message = (f"Epoch {current_epoch} | Step {global_step}/{config['train']['steps']} | "
                               f"Total Loss: {avg_total_loss_print:.4f} | "
                               f"Edge Loss: {avg_edge_loss_print:.4f} | ")
                if num_node_classes_for_model > 0:
                    log_message += f"Node Attr Loss: {avg_node_attr_loss_print:.4f} | "
                log_message += (f"Time/Iter: {time_per_iter_print:.4f}s | "
                                f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}")
                print(log_message)

                train_loss_sum_print_iter = 0
                edge_loss_sum_print_iter = 0
                node_attr_loss_sum_print_iter = 0
                start_time_print_iter = time.time()

            if global_step % config['train']['checkpoint_iter'] == 0 or global_step >= config['train']['steps']:
                state = {
                    "global_step": global_step, "config": config,
                    "node_model": node_model.state_dict(), "edge_model": edge_model.state_dict(),
                    "optim_node_model": optim_node_model.state_dict(),
                    "optim_edge_model": optim_edge_model.state_dict(),
                    "scheduler_node_model": scheduler_node_model.state_dict(),
                    "scheduler_edge_model": scheduler_edge_model.state_dict(),
                    "criterion_edges": criterion_edges.state_dict(),
                }
                if criterion_node_attr is not None:
                    state["criterion_node_attr"] = criterion_node_attr.state_dict()

                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pth")
                print(f"Saving checkpoint to {checkpoint_path}...")
                torch.save(state, checkpoint_path)
                print("Checkpoint saved.")

        if steps_per_epoch > 0:
            avg_epoch_total_loss = epoch_train_loss_sum / steps_per_epoch
            avg_epoch_edge_loss = epoch_edge_loss_sum / steps_per_epoch
            avg_epoch_node_attr_loss = epoch_node_attr_loss_sum / steps_per_epoch

            writer.add_scalar('loss_epoch/total_loss', avg_epoch_total_loss, current_epoch)
            writer.add_scalar('loss_epoch/edge_loss', avg_epoch_edge_loss, current_epoch)
            if num_node_classes_for_model > 0:
                writer.add_scalar('loss_epoch/node_attribute_loss', avg_epoch_node_attr_loss, current_epoch)

            print(f"--- End of Epoch {current_epoch} --- Avg Total Loss: {avg_epoch_total_loss:.4f}, "
                  f"Avg Edge Loss: {avg_epoch_edge_loss:.4f}, Avg Node Attr Loss: {avg_epoch_node_attr_loss:.4f}")

        scheduler_node_model.step()
        scheduler_edge_model.step()
        writer.add_scalar('lr/node_model_lr', scheduler_node_model.get_last_lr()[0], current_epoch)
        writer.add_scalar('lr/edge_model_lr', scheduler_edge_model.get_last_lr()[0], current_epoch)

    writer.close()
    print("Training finished.")
