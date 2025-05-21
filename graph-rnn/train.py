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
from extension_data import DirectedGraphDataSet  # Assuming extension_data_no_shuffle.py is the filename
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
    graph_rnn.train()  # Set to train mode
    edge_mlp.train()  # Set to train mode
    graph_rnn.zero_grad()
    edge_mlp.zero_grad()

    # Unpack data from DirectedGraphDataSet
    # s: adjacency sequence (S_i), lens: actual number of nodes
    s = data_batch['x_adj'].float().to(device)  # Shape: [B, max_N, M, edge_feature_len]
    node_attrs_target = data_batch['node_attr_onehot'].float().to(device)  # Shape: [B, max_N, num_node_classes]
    lens = data_batch['len'].cpu()

    num_node_classes = getattr(graph_rnn, 'num_node_classes', 0)
    batch_size = s.shape[0]
    m_window_size = s.shape[2]
    edge_feature_len = s.shape[3]

    # Prepare inputs for GraphLevelRNN
    # SOS for adjacency sequence
    sos_adj_frame = torch.ones([batch_size, 1, m_window_size, edge_feature_len], device=device)
    # Input S_{-1}, S0..S_{N-2}
    x_adj_input = torch.cat((sos_adj_frame, s[:, :-1, :, :]), dim=1)

    # SOS for previous node attributes
    sos_attr_frame = torch.zeros([batch_size, 1, num_node_classes], device=device)
    # Input A_{-1}, A0..A_{N-2}
    prev_node_attr_input = torch.cat((sos_attr_frame, node_attrs_target[:, :-1, :]), dim=1)

    # Target for node attributes is A0..A_{N-1}
    target_node_attr_s = node_attrs_target

    # Target for edges is S0..S_{N-1}
    y_adj_target = s

    current_lens = lens

    graph_rnn.reset_hidden()
    # GraphLevelRNN returns hidden state for edge model and logits for current node's attributes
    hidden_for_mlp, node_attribute_logits = graph_rnn(x_adj_input, prev_node_attr_input, current_lens)

    # EdgeLevelMLP predicts all M edge features at once based on hidden_for_mlp
    y_edge_pred = edge_mlp(hidden_for_mlp, return_logits=use_edge_features)  # Shape: [B, max_N, M, edge_feature_len]

    # --- Edge Loss Calculation ---
    # Pack predictions and targets based on actual node lengths
    # We are predicting M edge features for each of the N nodes.
    packed_y_edge_pred = pack_padded_sequence(y_edge_pred, current_lens, batch_first=True, enforce_sorted=False).data
    packed_y_adj_target = pack_padded_sequence(y_adj_target, current_lens, batch_first=True, enforce_sorted=False).data

    # packed_y_edge_pred and packed_y_adj_target have shape [sum_of_lengths * M, edge_feature_len]
    # if we flatten the M dimension. Or [sum_of_lengths, M, edge_feature_len].
    # For CrossEntropyLoss, target should be class indices.
    # The MLP predicts M edge slots simultaneously. Loss is over all these slots for valid nodes.

    if use_edge_features:  # CrossEntropyLoss
        # Reshape for CrossEntropy: pred [N_total_slots, C], target [N_total_slots]
        # N_total_slots = sum_of_lengths * M
        # C = edge_feature_len
        loss_edges = criterion_edges(
            packed_y_edge_pred.reshape(-1, edge_feature_len),
            torch.argmax(packed_y_adj_target.reshape(-1, edge_feature_len), dim=-1)
        )
    else:  # BCELoss (edge_feature_len == 1)
        loss_edges = criterion_edges(
            packed_y_edge_pred.reshape(-1, 1),
            packed_y_adj_target.reshape(-1, 1)
        )

    total_loss = loss_edges

    # --- Node Attribute Loss Calculation ---
    if node_attribute_logits is not None and criterion_node_attr and num_node_classes > 0:
        packed_node_attr_logits = pack_padded_sequence(node_attribute_logits, current_lens, batch_first=True,
                                                       enforce_sorted=False).data
        packed_target_node_attr_onehot = pack_padded_sequence(target_node_attr_s, current_lens, batch_first=True,
                                                              enforce_sorted=False).data

        packed_target_node_attr_indices = torch.argmax(packed_target_node_attr_onehot, dim=-1)

        loss_node_attributes = criterion_node_attr(packed_node_attr_logits, packed_target_node_attr_indices)
        total_loss = total_loss + node_attribute_loss_weight * loss_node_attributes
    else:
        loss_node_attributes = torch.tensor(0.0, device=device)

    total_loss.backward()
    optim_graph_rnn.step()
    optim_edge_mlp.step()

    # Schedulers are typically stepped per epoch, but original code steps per iteration.
    # scheduler_graph_rnn.step()
    # scheduler_mlp.step()

    return total_loss.item(), loss_edges.item(), loss_node_attributes.item()


def train_rnn_step(graph_rnn, edge_rnn, data_batch,
                   criterion_edges, criterion_node_attr,
                   optim_graph_rnn, optim_edge_model,
                   scheduler_graph_rnn, scheduler_edge_model,
                   device, use_edge_features, num_node_classes,
                   node_attribute_loss_weight):
    """ Train GraphRNN with RNN edge model, including node attribute prediction. """
    graph_rnn.train()  # Set to train mode
    edge_rnn.train()  # Set to train mode
    graph_rnn.zero_grad()
    edge_rnn.zero_grad()

    # Unpack data from DataLoader (output of DirectedGraphDataSet)
    x_adj_seq = data_batch['x_adj'].float().to(device)
    node_attr_seq = data_batch['node_attr_onehot'].float().to(device)
    lens = data_batch['len'].cpu()

    batch_size = x_adj_seq.shape[0]
    # max_num_nodes = x_adj_seq.shape[1] # Not directly used, lens is more important
    m_adj_len = x_adj_seq.shape[2]  # This is M from config['data']['m']
    edge_feature_len = x_adj_seq.shape[3]  # This is from config['model']['GraphRNN']['edge_feature_len']

    # 1. Prepare inputs for GraphLevelRNN
    sos_adj_frame = torch.ones([batch_size, 1, m_adj_len, edge_feature_len], device=device)
    x_adj_for_graph_rnn_input = torch.cat((sos_adj_frame, x_adj_seq[:, :-1, :, :]), dim=1)

    sos_attr_frame = torch.zeros([batch_size, 1, num_node_classes], device=device)
    prev_node_attr_for_graph_rnn_input = torch.cat((sos_attr_frame, node_attr_seq[:, :-1, :]), dim=1)

    graph_rnn.reset_hidden()
    h_for_edge_rnn, pred_node_attr_logits = graph_rnn(x_adj_for_graph_rnn_input,
                                                      prev_node_attr_for_graph_rnn_input,
                                                      lens)

    # 2. Node Attribute Loss Calculation
    packed_pred_node_attr_logits = pack_padded_sequence(pred_node_attr_logits, lens, batch_first=True,
                                                        enforce_sorted=False).data
    packed_target_node_attr_onehot = pack_padded_sequence(node_attr_seq, lens, batch_first=True,
                                                          enforce_sorted=False).data
    packed_target_node_attr_indices = torch.argmax(packed_target_node_attr_onehot, dim=-1)
    loss_node_attributes = criterion_node_attr(packed_pred_node_attr_logits, packed_target_node_attr_indices)

    # 3. Edge Prediction Loss Calculation
    hidden_packed_for_edge_rnn = pack_padded_sequence(h_for_edge_rnn, lens, batch_first=True, enforce_sorted=False).data
    edge_rnn.set_first_layer_hidden(hidden_packed_for_edge_rnn)

    packed_x_adj_seq_data = pack_padded_sequence(x_adj_seq, lens, batch_first=True, enforce_sorted=False).data
    sos_edge_frame_for_packed = torch.ones([packed_x_adj_seq_data.shape[0], 1, edge_feature_len], device=device)
    x_edge_rnn_input_packed = torch.cat((sos_edge_frame_for_packed, packed_x_adj_seq_data[:, :-1, :]), dim=1)
    y_edge_rnn_target_packed = packed_x_adj_seq_data

    # Calculate sequence lengths for each edge sequence (M for most, less for early ones)
    # graph_rnn.input_size is M (max number of edges to predict for a node)
    m_edge_pred_window = graph_rnn.input_size  # This is M from config['data']['m']

    edge_seq_lens_for_packing = []
    for l_nodes_in_graph in lens:
        for i_node_idx_in_graph in range(1, l_nodes_in_graph.item() + 1):
            edge_seq_lens_for_packing.append(min(i_node_idx_in_graph, m_edge_pred_window))

    # EdgeLevelRNN's forward pass expects x_edge_lens for its internal packing
    # The input to EdgeLevelRNN is x_edge_rnn_input_packed, which is already a flat batch.
    # The lengths passed to EdgeLevelRNN should correspond to the actual number of edge slots to predict for each node.
    # These are the values in edge_seq_lens_for_packing.
    y_edge_rnn_pred = edge_rnn(x_edge_rnn_input_packed,
                               torch.tensor(edge_seq_lens_for_packing, device=device, dtype=torch.long),
                               return_logits=use_edge_features)

    # Pack predictions and targets for loss calculation, using the actual edge sequence lengths
    # y_edge_rnn_pred is [sum_of_node_lengths, M_target_edge_slots, edge_feature_len]
    # y_edge_rnn_target_packed is [sum_of_node_lengths, M_target_edge_slots, edge_feature_len]
    # We need to pack them using edge_seq_lens_for_packing to only consider valid edge predictions.

    # Create a PackedSequence for y_edge_rnn_pred
    # First, ensure y_edge_rnn_pred is padded according to max(edge_seq_lens_for_packing) if EdgeLevelRNN doesn't already return it padded
    # The current EdgeLevelRNN returns a padded tensor if x_edge_lens was used for unpacking.
    # So, y_edge_rnn_pred should be [sum_of_node_lengths, max_edge_seq_len, edge_feature_len]

    # We need to pack based on the *true* lengths of edge sequences for each node.
    # These true lengths are in edge_seq_lens_for_packing.

    # Correct packing for loss:
    # The y_edge_rnn_pred and y_edge_rnn_target_packed are already shaped correctly for element-wise comparison
    # up to the actual length of each edge sequence. We need to select these valid parts.

    # Flatten predictions and targets, then select valid parts for loss
    # y_edge_rnn_pred: [total_nodes_in_batch, M, edge_feature_len]
    # y_edge_rnn_target_packed: [total_nodes_in_batch, M, edge_feature_len]

    # Create masks or select directly
    valid_preds_list = []
    valid_targets_list = []

    current_pos = 0
    for i, l_nodes_in_graph in enumerate(lens):  # Iterate through batch
        # For each graph in the batch
        graph_preds = y_edge_rnn_pred[current_pos: current_pos + l_nodes_in_graph]  # Preds for this graph
        graph_targets = y_edge_rnn_target_packed[current_pos: current_pos + l_nodes_in_graph]  # Targets for this graph

        graph_edge_seq_lens = []
        for i_node_idx_in_graph in range(1, l_nodes_in_graph.item() + 1):
            graph_edge_seq_lens.append(min(i_node_idx_in_graph, m_edge_pred_window))

        for node_j in range(l_nodes_in_graph):  # For each node in this graph
            actual_len_edges_for_this_node = graph_edge_seq_lens[node_j]
            if actual_len_edges_for_this_node > 0:
                valid_preds_list.append(graph_preds[node_j, :actual_len_edges_for_this_node, :])
                valid_targets_list.append(graph_targets[node_j, :actual_len_edges_for_this_node, :])
        current_pos += l_nodes_in_graph

    if not valid_preds_list:  # Handle cases with no valid edges (e.g., all graphs have 0 or 1 node)
        loss_edges = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        y_edge_rnn_pred_for_loss = torch.cat(valid_preds_list,
                                             dim=0)  # Shape [total_valid_edge_slots, edge_feature_len]
        y_edge_rnn_target_for_loss = torch.cat(valid_targets_list,
                                               dim=0)  # Shape [total_valid_edge_slots, edge_feature_len]

        if use_edge_features:  # CrossEntropyLoss
            if y_edge_rnn_target_for_loss.shape[-1] > 1 and criterion_edges.__class__.__name__ == 'CrossEntropyLoss':
                y_edge_rnn_target_for_loss_indices = torch.argmax(y_edge_rnn_target_for_loss, dim=-1)
            else:  # Should not happen if use_edge_features is True and CELoss is used with one-hot target
                y_edge_rnn_target_for_loss_indices = y_edge_rnn_target_for_loss.squeeze(-1).long()  # Fallback
            loss_edges = criterion_edges(y_edge_rnn_pred_for_loss, y_edge_rnn_target_for_loss_indices)
        else:  # BCELoss (edge_feature_len == 1)
            loss_edges = criterion_edges(y_edge_rnn_pred_for_loss, y_edge_rnn_target_for_loss)

    # 4. Total Loss and Backpropagation
    total_loss = loss_edges + node_attribute_loss_weight * loss_node_attributes

    total_loss.backward()
    optim_graph_rnn.step()
    optim_edge_model.step()

    # Schedulers are typically stepped per epoch
    # scheduler_graph_rnn.step()
    # scheduler_edge_model.step()

    return total_loss.item(), loss_edges.item(), loss_node_attributes.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Path of the config file to use for training')
    parser.add_argument('-r', '--restore', dest='restore_path', required=False, default=None,
                        help='Checkpoint to continue training from')
    parser.add_argument('--gpu', dest='gpu_id', required=False, default=0, type=int,
                        help='Id of the GPU to use')
    args = parser.parse_args()

    base_path = os.path.dirname(args.config_file) if args.config_file else '.'  # Handle no config file for default path

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    checkpoint_dir = os.path.join(base_path, config['train']['checkpoint_dir'])
    log_dir = os.path.join(base_path, config['train']['log_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model parameters from config
    num_node_features = config['model']['GraphRNN'][
        'num_node_classes']  # num_node_classes in model is num_features for data
    edge_feature_len = config['model']['GraphRNN']['edge_feature_len']
    m_param = config['data']['m']

    if num_node_features <= 0:
        raise ValueError("'num_node_classes' in config must be > 0 for typed node prediction.")
    if edge_feature_len <= 0:
        raise ValueError("'edge_feature_len' in config must be > 0 for typed edge prediction.")

    # Create models
    if config['model']['edge_model'] == 'rnn':
        node_model = GraphLevelRNN(input_size=m_param,  # M-window size
                                   output_size=config['model']['EdgeRNN']['hidden_size'],  # For EdgeRNN init
                                   num_node_classes=num_node_features,
                                   **config['model']['GraphRNN']).to(
            device)  # Passes embedding_size, hidden_size, num_layers, edge_feature_len
        edge_model = EdgeLevelRNN(**config['model']['EdgeRNN']).to(
            device)  # Passes embedding_size, hidden_size, num_layers, edge_feature_len
        step_fn = train_rnn_step
    else:  # MLP edge model
        node_model = GraphLevelRNN(input_size=m_param,
                                   output_size=None,  # MLP takes hidden state directly
                                   num_node_classes=num_node_features,
                                   **config['model']['GraphRNN']).to(device)
        edge_model = EdgeLevelMLP(input_size_from_graph_rnn=config['model']['GraphRNN']['hidden_size'],
                                  mlp_hidden_size=config['model']['EdgeMLP']['hidden_size'],
                                  num_edges_to_predict=m_param,  # MLP predicts M edge slots
                                  edge_feature_len=edge_feature_len
                                  ).to(device)
        step_fn = train_mlp_step
        print("Warning: MLP edge model path is being used.")

    use_edge_features_for_loss = edge_feature_len > 1

    if use_edge_features_for_loss:
        criterion_edges = torch.nn.CrossEntropyLoss().to(device)
    else:
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
        global_step = state["global_step"]
        node_model.load_state_dict(state["node_model"])
        edge_model.load_state_dict(state["edge_model"])
        optim_node_model.load_state_dict(state["optim_node_model"])
        optim_edge_model.load_state_dict(state["optim_edge_model"])
        scheduler_node_model.load_state_dict(state["scheduler_node_model"])
        scheduler_edge_model.load_state_dict(state["scheduler_edge_model"])
        if "criterion_edges" in state:  # backward compatibility
            criterion_edges.load_state_dict(state["criterion_edges"])
        if "criterion_node_attr" in state and criterion_node_attr is not None:
            criterion_node_attr.load_state_dict(state["criterion_node_attr"])
        print(f"Restored to global_step: {global_step}")

    # DATA LOADING
    # Get pyg_file_path from config, default to "aig_directed.pt" if not specified
    pyg_file_path = config['data'].get('pyg_file_path', 'aig_directed.pt')
    if not os.path.isabs(pyg_file_path) and base_path:  # If relative path, make it relative to config file dir
        pyg_file_path = os.path.join(base_path, pyg_file_path)

    print(f"Attempting to load dataset from: {os.path.abspath(pyg_file_path)}")

    dataset = DirectedGraphDataSet(
        dataset_type=config['model']['mode'],  # e.g. 'aig-custom-topsort'
        m=m_param,
        pyg_file_path=pyg_file_path,
        num_node_features=num_node_features,  # From config['model']['GraphRNN']['num_node_classes']
        num_edge_features=edge_feature_len,  # From config['model']['GraphRNN']['edge_feature_len']
        training=True,  # For the training dataset
        train_split=config['data']['train_split']
    )

    data_loader = DataLoader(dataset,
                             batch_size=config['train']['batch_size'],
                             shuffle=True)  # Shuffle batches from the training split

    # Optional: Create a validation dataset and loader
    # val_dataset = DirectedGraphDataSet(
    #     dataset_type=config['model']['mode'],
    #     m=m_param,
    #     pyg_file_path=pyg_file_path,
    #     num_node_features=num_node_features,
    #     num_edge_features=edge_feature_len,
    #     training=False, # For the validation/test dataset
    #     train_split=config['data']['train_split']
    # )
    # val_data_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    print(f"Starting training from global_step: {global_step}")
    node_model.train()
    edge_model.train()

    done = False
    train_loss_sum = 0
    edge_loss_sum = 0
    node_attr_loss_sum = 0
    start_time_epoch = time.time()  # For epoch timing

    # Determine number of epochs based on steps and dataset size
    # This is an approximation if batch sizes vary or dataset isn't perfectly divisible
    steps_per_epoch = len(data_loader)
    total_epochs = (config['train'][
                        'steps'] - global_step + steps_per_epoch - 1) // steps_per_epoch if steps_per_epoch > 0 else 0
    print(f"Approximate total epochs to run: {total_epochs} ({steps_per_epoch} steps per epoch)")

    current_epoch = 0

    while not done:
        current_epoch += 1
        print(f"Epoch {current_epoch}/{total_epochs if total_epochs > 0 else 'N/A (dynamic steps)'} ---")
        epoch_train_loss_sum = 0
        epoch_edge_loss_sum = 0
        epoch_node_attr_loss_sum = 0

        for batch_idx, data_batch in enumerate(data_loader):
            if global_step >= config['train']['steps']:
                done = True
                break
            global_step += 1

            if step_fn == train_rnn_step:
                current_total_loss, current_edge_loss, current_node_attr_loss = step_fn(
                    node_model, edge_model, data_batch,
                    criterion_edges, criterion_node_attr,
                    optim_node_model, optim_edge_model,
                    scheduler_node_model, scheduler_edge_model,  # Schedulers passed but typically stepped per epoch
                    device, use_edge_features_for_loss, num_node_features,
                    config['train'].get('node_attribute_loss_weight', 1.0)
                )
            else:  # MLP step
                current_total_loss, current_edge_loss, current_node_attr_loss = step_fn(
                    node_model, edge_model, data_batch, criterion_edges,
                    optim_node_model, optim_edge_model,
                    scheduler_node_model, scheduler_edge_model,
                    device, use_edge_features_for_loss,
                    criterion_node_attr, config['train'].get('node_attribute_loss_weight', 1.0)
                )

            train_loss_sum += current_total_loss
            edge_loss_sum += current_edge_loss
            node_attr_loss_sum += current_node_attr_loss  # Will be 0 if MLP path doesn't return it properly

            epoch_train_loss_sum += current_total_loss
            epoch_edge_loss_sum += current_edge_loss
            epoch_node_attr_loss_sum += current_node_attr_loss

            writer.add_scalar('loss_iter/total_loss', current_total_loss, global_step)
            writer.add_scalar('loss_iter/edge_loss', current_edge_loss, global_step)
            if num_node_features > 0 and (current_node_attr_loss > 0 or step_fn == train_rnn_step):  # Log if meaningful
                writer.add_scalar('loss_iter/node_attribute_loss', current_node_attr_loss, global_step)

            if global_step % config['train']['print_iter'] == 0:
                avg_total_loss_print = train_loss_sum / config['train']['print_iter']
                avg_edge_loss_print = edge_loss_sum / config['train']['print_iter']
                avg_node_attr_loss_print = node_attr_loss_sum / config['train']['print_iter']

                elapsed_time_iter = time.time() - start_time_epoch  # Time for print_iter iterations
                time_per_iter_print = elapsed_time_iter / config['train']['print_iter']
                eta_seconds = (config['train']['steps'] - global_step) * time_per_iter_print

                log_message = (f"Epoch {current_epoch} | Step {global_step}/{config['train']['steps']} | "
                               f"Total Loss: {avg_total_loss_print:.4f} | "
                               f"Edge Loss: {avg_edge_loss_print:.4f} | ")
                if num_node_features > 0:
                    log_message += f"Node Attr Loss: {avg_node_attr_loss_print:.4f} | "
                log_message += (f"Time/Iter: {time_per_iter_print:.4f}s | "
                                f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}")
                print(log_message)

                train_loss_sum = 0
                edge_loss_sum = 0
                node_attr_loss_sum = 0
                start_time_epoch = time.time()  # Reset timer for next print_iter block

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

        # End of Epoch
        if steps_per_epoch > 0:
            avg_epoch_total_loss = epoch_train_loss_sum / steps_per_epoch
            avg_epoch_edge_loss = epoch_edge_loss_sum / steps_per_epoch
            avg_epoch_node_attr_loss = epoch_node_attr_loss_sum / steps_per_epoch

            writer.add_scalar('loss_epoch/total_loss', avg_epoch_total_loss, current_epoch)
            writer.add_scalar('loss_epoch/edge_loss', avg_epoch_edge_loss, current_epoch)
            if num_node_features > 0:
                writer.add_scalar('loss_epoch/node_attribute_loss', avg_epoch_node_attr_loss, current_epoch)

            print(f"--- End of Epoch {current_epoch} --- Avg Total Loss: {avg_epoch_total_loss:.4f}, "
                  f"Avg Edge Loss: {avg_epoch_edge_loss:.4f}, Avg Node Attr Loss: {avg_epoch_node_attr_loss:.4f}")

        # Step schedulers per epoch
        scheduler_node_model.step()
        scheduler_edge_model.step()
        writer.add_scalar('lr/node_model_lr', scheduler_node_model.get_last_lr()[0], current_epoch)
        writer.add_scalar('lr/edge_model_lr', scheduler_edge_model.get_last_lr()[0], current_epoch)

    writer.close()
    print("Training finished.")
