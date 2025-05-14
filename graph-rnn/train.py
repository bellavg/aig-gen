import argparse
import yaml
import torch
import os
import time
import datetime
import torch.nn as nn  # Added for nn.CrossEntropyLoss

from data import GraphDataSet  # You'll need to modify this to provide new data format
from extension_data import DirectedGraphDataSet  # Same as above
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# Make sure this model.py is the one with node attribute prediction
from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP


def train_mlp_step(graph_rnn, edge_mlp, data, criterion_edges, optim_graph_rnn, optim_edge_mlp,
                   scheduler_graph_rnn, scheduler_mlp, device, use_edge_features,
                   # Added for consistency, but MLP path is not fully updated for node attributes
                   criterion_node_attr=None, node_attribute_loss_weight=0.0
                   ):
    """
    Train GraphRNN with MLP edge model.
    NOTE: This function is NOT fully updated to support node attribute prediction
    as per the user's focus on the RNN edge model. It's kept for structural
    consistency but would require similar modifications as train_rnn_step.
    """
    graph_rnn.zero_grad()
    edge_mlp.zero_grad()

    # Original MLP step expects 'x' in data, which was the adjacency sequence.
    # This part needs to be adapted if MLP is to be used with node attributes.
    # For now, we'll assume it might error or not work correctly with the new data format.
    s, lens = data.get('x', data.get('x_adj')).float().to(device), data['len'].cpu()  # Try to be robust

    num_node_classes = getattr(graph_rnn, 'num_node_classes', 0)  # Get from model if exists

    # Prepare inputs for GraphLevelRNN if it's the new version
    # This is a placeholder and needs proper SOS tokens and alignment like in train_rnn_step
    if num_node_classes > 0 and 'prev_node_attr' in data:
        prev_node_attr_s = data['prev_node_attr'].float().to(device)
        # Placeholder for SOS attributes for MLP path
        sos_attr_frame_mlp = torch.zeros([s.shape[0], 1, num_node_classes], device=device)
        if prev_node_attr_s.shape[1] == s.shape[1] - 1:  # if already shifted
            prev_node_attr_input_s = torch.cat((sos_attr_frame_mlp, prev_node_attr_s), dim=1)
        else:  # needs shifting
            prev_node_attr_input_s = torch.cat((sos_attr_frame_mlp, prev_node_attr_s[:, :-1, :]), dim=1)
    else:
        # Fallback or error if attributes are expected but not provided correctly
        if num_node_classes > 0:
            print(
                "Warning: MLP step called with attribute-aware GraphLevelRNN but prev_node_attr not in data or num_node_classes is 0.")
            # Create dummy prev_node_attributes if model expects it
            prev_node_attr_input_s = torch.zeros([s.shape[0], s.shape[1], num_node_classes], device=device)

    # If s does not have edge features, just add a dummy dimension 1
    if len(s.shape) == 3:  # [B, N, M]
        s = s.unsqueeze(3)  # [B, N, M, 1]

    # Teacher forcing for adjacency part (original logic)
    one_frame_adj = torch.ones([s.shape[0], 1, s.shape[2], s.shape[3]], device=device)
    zero_frame_adj = torch.zeros([s.shape[0], 1, s.shape[2], s.shape[3]], device=device)

    x_adj_input = torch.cat((one_frame_adj, s[:, :-1, :, :]), dim=1)  # Input S-1, S0..SN-2
    y_adj_target = s  # Target S0..SN-1

    current_lens = lens  # Original lens for GraphRNN

    graph_rnn.reset_hidden()
    if num_node_classes > 0:
        # Assuming GraphLevelRNN now returns a tuple
        hidden_for_mlp, node_attribute_logits = graph_rnn(x_adj_input, prev_node_attr_input_s, current_lens)
    else:  # Original GraphLevelRNN
        hidden_for_mlp = graph_rnn(x_adj_input, current_lens)
        node_attribute_logits = None

    y_edge_pred = edge_mlp(hidden_for_mlp, return_logits=use_edge_features)

    # Pack and pad for edge loss (original logic for y_adj_target)
    # The target for MLP is the current node's full adjacency vector
    y_adj_target_packed = pack_padded_sequence(y_adj_target, current_lens, batch_first=True, enforce_sorted=False).data
    y_adj_target_padded, _ = pad_packed_sequence(
        pack_padded_sequence(y_adj_target, current_lens, batch_first=True, enforce_sorted=False), batch_first=True)

    if use_edge_features:  # For CrossEntropyLoss on edges
        # y_edge_pred shape: [B, N, M, edge_feat_len]
        # y_adj_target_padded shape: [B, N, M, edge_feat_len]
        # Need to align for loss, typically MLP predicts all M edges at once.
        # The loss is usually on the packed sequence of these predictions.
        # This part might need careful review based on how MLP output and targets are structured for loss.
        # Original MLP output was [B, N, M*edge_feat_len], then reshaped.
        # Current EdgeLevelMLP output: [B, N, M, edge_feat_len]

        # Assuming loss is computed on packed data.
        # We need to pack y_edge_pred as well.
        y_edge_pred_packed = pack_padded_sequence(y_edge_pred, current_lens, batch_first=True,
                                                  enforce_sorted=False).data
        y_adj_target_packed_for_loss = y_adj_target_packed  # Already packed

        # CrossEntropy expects [N_elements, C] and [N_elements]
        # If edge_feature_len > 1, it's multi-class for each of M positions.
        # This requires reshaping: (B*N_packed)*M, C and (B*N_packed)*M
        # This part is complex and error-prone without knowing exact MLP target structure.
        # For now, let's assume a simpler case or that criterion_edges handles it.
        # If use_edge_features:
        #   y_adj_target_packed_for_loss = torch.argmax(y_adj_target_packed_for_loss, dim=-1) # if target is one-hot
        #   y_edge_pred_packed = y_edge_pred_packed.permute(0,2,1) # B,C,N if loss expects C in dim 1
        #   loss_edges = criterion_edges(y_edge_pred_packed.reshape(-1, y_edge_pred_packed.shape[-1]), y_adj_target_packed_for_loss.reshape(-1))

        # Sticking to a more direct application for now, assuming criterion handles shapes
        loss_edges = criterion_edges(y_edge_pred_packed, y_adj_target_packed_for_loss)


    else:  # For BCELoss on edges (edge_feature_len == 1)
        y_edge_pred_packed = pack_padded_sequence(y_edge_pred, current_lens, batch_first=True,
                                                  enforce_sorted=False).data
        loss_edges = criterion_edges(y_edge_pred_packed, y_adj_target_packed)

    total_loss = loss_edges
    # Node attribute loss (placeholder for MLP)
    if node_attribute_logits is not None and criterion_node_attr and 'target_node_attr' in data:
        target_node_attr_s = data['target_node_attr'].float().to(device)
        # pack node_attribute_logits and target_node_attr_s
        packed_node_attr_logits = pack_padded_sequence(node_attribute_logits, current_lens, batch_first=True,
                                                       enforce_sorted=False).data
        packed_target_node_attr = pack_padded_sequence(target_node_attr_s, current_lens, batch_first=True,
                                                       enforce_sorted=False).data

        if target_node_attr_s.shape[-1] > 1:  # if one-hot
            packed_target_node_attr_indices = torch.argmax(packed_target_node_attr, dim=-1)
        else:  # already indices
            packed_target_node_attr_indices = packed_target_node_attr.squeeze(-1).long()

        loss_node_attributes = criterion_node_attr(packed_node_attr_logits, packed_target_node_attr_indices)
        total_loss = total_loss + node_attribute_loss_weight * loss_node_attributes

    total_loss.backward()
    optim_graph_rnn.step()
    optim_edge_mlp.step()
    scheduler_graph_rnn.step()
    scheduler_mlp.step()

    return total_loss.item()


def train_rnn_step(graph_rnn, edge_rnn, data,
                   criterion_edges, criterion_node_attr,
                   optim_graph_rnn, optim_edge_model,
                   scheduler_graph_rnn, scheduler_edge_model,
                   device, use_edge_features, num_node_classes,
                   node_attribute_loss_weight):
    """ Train GraphRNN with RNN edge model, including node attribute prediction. """
    graph_rnn.zero_grad()
    edge_rnn.zero_grad()

    # Unpack data from DataLoader
    # x_adj_seq: Adjacency vectors S0, S1, ..., SN-1. Shape: [B, max_N, M, edge_feature_len]
    # node_attr_seq: Node attributes A0, A1, ..., AN-1. Shape: [B, max_N, num_node_classes] (one-hot)
    # lens: Actual number of nodes N for each graph. Shape: [B]
    x_adj_seq = data['x_adj'].float().to(device)
    node_attr_seq = data['node_attr_onehot'].float().to(device)  # Assuming one-hot
    lens = data['len'].cpu()  # Number of nodes in each graph

    batch_size = x_adj_seq.shape[0]
    max_num_nodes = x_adj_seq.shape[1]
    m_adj_len = x_adj_seq.shape[2]  # This is M, the input_size for GraphLevelRNN adj part
    edge_feature_len = x_adj_seq.shape[3]

    # 1. Prepare inputs for GraphLevelRNN
    # GraphLevelRNN processes N nodes for each graph.
    # Input for node i: (Adjacency of S_{i-1}, Attribute of A_{i-1})
    # Output for node i: (Hidden state h_i, Logits for A_i)

    # SOS token for adjacency sequence (e.g., all ones)
    sos_adj_frame = torch.ones([batch_size, 1, m_adj_len, edge_feature_len], device=device)
    # Adjacency input to GraphLevelRNN: [SOS_adj, S0, S1, ..., S_{N-2}]
    # x_adj_seq is [S0, ..., SN-1], so we take [:, :-1, :, :] for S0, ..., SN-2
    x_adj_for_graph_rnn_input = torch.cat((sos_adj_frame, x_adj_seq[:, :-1, :, :]), dim=1)

    # SOS token for previous node attribute sequence (e.g., all zeros)
    sos_attr_frame = torch.zeros([batch_size, 1, num_node_classes], device=device)
    # Attribute input to GraphLevelRNN: [SOS_attr, A0, A1, ..., A_{N-2}]
    # node_attr_seq is [A0, ..., AN-1], so we take [:, :-1, :] for A0, ..., AN-2
    prev_node_attr_for_graph_rnn_input = torch.cat((sos_attr_frame, node_attr_seq[:, :-1, :]), dim=1)

    graph_rnn.reset_hidden()
    # h_for_edge_rnn: Hidden states h0, h1, ..., h_{N-1}. Shape [B, max_N, G_RNN_hidden_or_output_size]
    # pred_node_attr_logits: Logits for A0, A1, ..., A_{N-1}. Shape [B, max_N, num_node_classes]
    h_for_edge_rnn, pred_node_attr_logits = graph_rnn(x_adj_for_graph_rnn_input,
                                                      prev_node_attr_for_graph_rnn_input,
                                                      lens)

    # 2. Node Attribute Loss Calculation
    # Targets are A0, A1, ..., A_{N-1} from node_attr_seq
    # Predictions are for A0, A1, ..., A_{N-1} from pred_node_attr_logits

    # Pack sequences for loss calculation (only consider actual nodes)
    packed_pred_node_attr_logits = pack_padded_sequence(pred_node_attr_logits, lens, batch_first=True,
                                                        enforce_sorted=False).data
    packed_target_node_attr_onehot = pack_padded_sequence(node_attr_seq, lens, batch_first=True,
                                                          enforce_sorted=False).data

    # Convert one-hot target attributes to class indices for CrossEntropyLoss
    packed_target_node_attr_indices = torch.argmax(packed_target_node_attr_onehot, dim=-1)

    loss_node_attributes = criterion_node_attr(packed_pred_node_attr_logits, packed_target_node_attr_indices)

    # 3. Edge Prediction Loss Calculation (similar to original, but using h_for_edge_rnn)
    # EdgeLevelRNN predicts edges for each node i (S_i) based on h_i.
    # Targets for edge prediction are S0, S1, ..., S_{N-1} from x_adj_seq.

    # Pack the hidden states h0, ..., h_{N-1} to feed to EdgeLevelRNN's set_first_layer_hidden
    # This hidden_packed corresponds to the hidden state for each actual node across the batch
    hidden_packed_for_edge_rnn = pack_padded_sequence(h_for_edge_rnn, lens, batch_first=True, enforce_sorted=False).data
    edge_rnn.set_first_layer_hidden(hidden_packed_for_edge_rnn)

    # Prepare edge sequence inputs and targets for EdgeLevelRNN
    # EdgeLevelRNN input for node i: [SOS_edge, e_{i,0}, e_{i,1}, ..., e_{i,M-1}] (teacher forcing from S_i)
    # EdgeLevelRNN target for node i: [e_{i,0}, e_{i,1}, ..., e_{i,M}]

    # x_adj_seq is [S0, S1, ..., S_{N-1}]. This is the target for edge predictions.
    # We need to pack it along the node dimension first.
    # packed_x_adj_seq_data shape: [sum_of_node_lengths, M, edge_feature_len]
    packed_x_adj_seq_data = pack_padded_sequence(x_adj_seq, lens, batch_first=True, enforce_sorted=False).data

    # Create SOS tokens for each edge sequence
    # sos_edge_frame_for_packed shape: [sum_of_node_lengths, 1, edge_feature_len]
    sos_edge_frame_for_packed = torch.ones([packed_x_adj_seq_data.shape[0], 1, edge_feature_len], device=device)

    # Input to EdgeLevelRNN: [SOS_edge, S_i[:, :-1, :]] for each packed node i
    # packed_x_adj_seq_data[:, :-1, :] means taking all but the last edge for input
    # x_edge_rnn_input_packed shape: [sum_of_node_lengths, M, edge_feature_len] (assuming M is edge seq len)
    # If M is the number of edges, then SOS is prepended to M edges.
    # Original code: x_edge_rnn = torch.cat((one_frame, seq_packed[:, :-1, :]), dim=1)
    # where seq_packed was [sum_of_node_lengths, M_edges, edge_feat_dim]
    # So, x_edge_rnn_input_packed should be [sum_of_node_lengths, M_edges_input_len, edge_feat_dim]
    # and y_edge_rnn_target_packed should be [sum_of_node_lengths, M_edges_target_len, edge_feat_dim]

    x_edge_rnn_input_packed = torch.cat((sos_edge_frame_for_packed, packed_x_adj_seq_data[:, :-1, :]), dim=1)
    y_edge_rnn_target_packed = packed_x_adj_seq_data  # Target is the original S_i sequence

    # Calculate sequence lengths for each edge sequence (M for most, less for early ones)
    # These are the lengths for the second RNN (EdgeLevelRNN)
    edge_seq_lens_packed = []
    # graph_rnn.input_size is M (max number of edges to predict for a node)
    # This M is different from m_adj_len if GraphLevelRNN's input_size param means something else.
    # Assuming graph_rnn.input_size refers to the 'm' parameter for edge prediction window.
    m_edge_pred_window = graph_rnn.input_size  # This should be M, the number of edges in S_i

    # lens are the number of nodes. For each node i (from 1 to N), it can connect to min(i, M) previous nodes.
    # The EdgeLevelRNN processes sequences of length min(node_idx_in_graph, M_edge_pred_window)
    # Example: Node 0 (1st node): connects to min(0,M)=0. EdgeRNN seq len for S0 is 0?
    # Paper: S_i is a vector of length min(i-1, M).
    # The code's `seq_packed_len` was `min(i, m)` where `i` is node index (1 to l)
    # Let's use the original logic for edge sequence lengths.
    for l_nodes in lens:  # l_nodes is N for a graph
        for i_node_idx in range(1, l_nodes.item() + 1):  # For node S_0 to S_{N-1} (indices 1 to N)
            edge_seq_lens_packed.append(min(i_node_idx, m_edge_pred_window))  # Length of S_i
    # This list needs to be sorted for pack_padded_sequence if enforce_sorted=True, but it's not for .data
    # However, EdgeLevelRNN forward call will use these lengths for its own packing.
    # The input x_edge_rnn_input_packed is already packed batch-wise for nodes.
    # The EdgeLevelRNN will internally pack again if x_edge_lens are provided to it.
    # The current EdgeLevelRNN takes x_edge_lens for its internal packing.
    # The length of x_edge_rnn_input_packed is sum_of_node_lengths.
    # The length of y_edge_rnn_target_packed is sum_of_node_lengths.
    # The edge_seq_lens_packed corresponds to these.

    # Sort edge_seq_lens_packed for robust packing if EdgeLevelRNN does it.
    # However, the input to EdgeLevelRNN is already a flat batch of edge sequences.
    # So, edge_seq_lens_packed should be the lengths for this flat batch.

    # The `seq_packed_len` in original code was used for packing `y_edge_rnn` and `y_edge_rnn_pred`
    # before loss. Here, `edge_seq_lens_packed` is for the EdgeLevelRNN's forward pass.
    # The input to EdgeLevelRNN is `x_edge_rnn_input_packed` which is [total_nodes, M_input, feat]
    # The `edge_seq_lens_packed` should be for this.
    # The length of `x_edge_rnn_input_packed` along dim 1 is M (or M-1 + SOS).
    # The `x_edge_lens` for `edge_rnn.forward` should be the actual number of edges to predict for each node.
    # These are `min(node_index_in_graph, M_edge_pred_window)`.
    # The `edge_seq_lens_packed` list IS these lengths.

    # Compute edge probabilities
    # y_edge_rnn_pred shape: [sum_of_node_lengths, M_target, edge_feature_len]
    y_edge_rnn_pred = edge_rnn(x_edge_rnn_input_packed,
                               torch.tensor(edge_seq_lens_packed, device=device),  # Pass actual edge counts
                               return_logits=use_edge_features)

    # Target for edge loss is y_edge_rnn_target_packed
    # Need to ensure y_edge_rnn_pred and y_edge_rnn_target_packed are aligned and properly shaped for loss.
    # Both are currently [sum_of_node_lengths, M, edge_feature_len].
    # The loss should be computed only over the valid parts of these sequences using edge_seq_lens_packed.

    # Pack y_edge_rnn_pred and y_edge_rnn_target_packed using edge_seq_lens_packed for the loss
    y_edge_rnn_pred_for_loss = pack_padded_sequence(y_edge_rnn_pred, torch.tensor(edge_seq_lens_packed, device=device),
                                                    batch_first=True, enforce_sorted=False).data
    y_edge_rnn_target_for_loss = pack_padded_sequence(y_edge_rnn_target_packed,
                                                      torch.tensor(edge_seq_lens_packed, device=device),
                                                      batch_first=True, enforce_sorted=False).data

    if use_edge_features:
        # CrossEntropyLoss expects logits [N, C] and targets [N]
        # y_..._for_loss are [total_valid_edges, edge_feature_len]
        # Target needs to be class indices if one-hot
        if y_edge_rnn_target_for_loss.shape[-1] > 1 and criterion_edges.__class__.__name__ == 'CrossEntropyLoss':
            y_edge_rnn_target_for_loss_indices = torch.argmax(y_edge_rnn_target_for_loss, dim=-1)
        else:  # Already indices or BCELoss will handle it
            y_edge_rnn_target_for_loss_indices = y_edge_rnn_target_for_loss.squeeze(-1)

        loss_edges = criterion_edges(y_edge_rnn_pred_for_loss, y_edge_rnn_target_for_loss_indices)
    else:  # BCELoss
        loss_edges = criterion_edges(y_edge_rnn_pred_for_loss, y_edge_rnn_target_for_loss)

    # 4. Total Loss and Backpropagation
    total_loss = loss_edges + node_attribute_loss_weight * loss_node_attributes

    total_loss.backward()
    optim_graph_rnn.step()
    optim_edge_model.step()  # This is for the edge_rnn
    scheduler_graph_rnn.step()
    scheduler_edge_model.step()  # This is for the edge_rnn

    return total_loss.item(), loss_edges.item(), loss_node_attributes.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Path of the config file to use for training')
    parser.add_argument('-r', '--restore', dest='restore_path', required=False, default=None,
                        help='Checkpoint to continue training from')
    parser.add_argument('--gpu', dest='gpu_id', required=False, default=0, type=int,
                        help='Id of the GPU to use')
    args = parser.parse_args()

    base_path = os.path.dirname(args.config_file)

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure checkpoint and log directories exist
    checkpoint_dir = os.path.join(base_path, config['train']['checkpoint_dir'])
    log_dir = os.path.join(base_path, config['train']['log_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # Get num_node_classes from config (NEW)
    # Ensure this path exists in your config YAML, e.g., config['model']['GraphRNN']['num_node_classes']
    num_node_classes = config.get('model', {}).get('GraphRNN', {}).get('num_node_classes', 0)
    if num_node_classes == 0:
        print(
            "Warning: 'num_node_classes' not found or is 0 in config. Node attribute prediction will be disabled or may fail.")

    # Create models
    if config['model']['edge_model'] == 'rnn':
        node_model = GraphLevelRNN(input_size=config['data']['m'],
                                   output_size=config['model']['EdgeRNN']['hidden_size'],
                                   num_node_classes=num_node_classes,  # NEW
                                   **config['model']['GraphRNN']).to(device)
        edge_model = EdgeLevelRNN(**config['model']['EdgeRNN']).to(device)
        step_fn = train_rnn_step
    else:  # MLP edge model
        # MLP path not fully updated for node attributes, but model instantiation needs num_node_classes
        node_model = GraphLevelRNN(input_size=config['data']['m'],
                                   output_size=None,
                                   num_node_classes=num_node_classes,  # NEW
                                   **config['model']['GraphRNN']).to(device)
        edge_model = EdgeLevelMLP(input_size_from_graph_rnn=config['model']['GraphRNN']['hidden_size'],
                                  # Corrected param name
                                  mlp_hidden_size=config['model']['EdgeMLP']['hidden_size'],
                                  num_edges_to_predict=config['data']['m'],
                                  edge_feature_len=config['model']['GraphRNN'].get('edge_feature_len', 1)
                                  # Use from GraphRNN or default
                                  ).to(device)
        step_fn = train_mlp_step
        print("Warning: MLP edge model path is not fully updated for node attribute prediction.")

    # Determine if edge features are multi-class (for CrossEntropyLoss on edges)
    use_edge_features = 'edge_feature_len' in config['model']['GraphRNN'] \
                        and config['model']['GraphRNN']['edge_feature_len'] > 1

    if use_edge_features:
        criterion_edges = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion_edges = torch.nn.BCELoss().to(device)

    # Criterion for node attributes (NEW) - always CrossEntropy for class prediction
    criterion_node_attr = torch.nn.CrossEntropyLoss().to(device) if num_node_classes > 0 else None

    # Optimizers
    optim_node_model = torch.optim.Adam(list(node_model.parameters()), lr=config['train']['lr'])
    optim_edge_model = torch.optim.Adam(list(edge_model.parameters()), lr=config['train']['lr'])

    # Schedulers
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
        criterion_edges.load_state_dict(state["criterion_edges"])  # Changed from "criterion"
        if "criterion_node_attr" in state and criterion_node_attr is not None:
            criterion_node_attr.load_state_dict(state["criterion_node_attr"])

    # DATA LOADING: This part is CRITICAL.
    # Your GraphDataSet/DirectedGraphDataSet must be updated to return dicts with
    # 'x_adj', 'node_attr_onehot', and 'len'.
    if 'mode' in config['model'] and 'directed' in config['model']['mode']:
        dataset = DirectedGraphDataSet(**config['data'],
                                       num_node_classes=num_node_classes)  # Pass num_node_classes if needed by data loader
    else:
        dataset = GraphDataSet(**config['data'], num_node_classes=num_node_classes)  # Pass num_node_classes if needed
    data_loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True)  # Added shuffle=True

    node_model.train()
    edge_model.train()

    done = False
    train_loss_sum = 0
    edge_loss_sum = 0
    node_attr_loss_sum = 0
    start_step = global_step
    start_time = time.time()

    # Get node attribute loss weight from config (NEW)
    node_attribute_loss_weight = config.get('train', {}).get('node_attribute_loss_weight', 1.0)  # Default to 1.0

    while not done:
        for batch_idx, data_batch in enumerate(data_loader):
            global_step += 1
            if global_step > config['train']['steps']:
                done = True
                break

            if step_fn == train_rnn_step:
                current_total_loss, current_edge_loss, current_node_attr_loss = step_fn(
                    node_model, edge_model, data_batch,
                    criterion_edges, criterion_node_attr,
                    optim_node_model, optim_edge_model,  # optim_edge_model used for edge_rnn
                    scheduler_node_model, scheduler_edge_model,  # scheduler_edge_model for edge_rnn
                    device, use_edge_features, num_node_classes,
                    node_attribute_loss_weight
                )
                train_loss_sum += current_total_loss
                edge_loss_sum += current_edge_loss
                if num_node_classes > 0:
                    node_attr_loss_sum += current_node_attr_loss
            else:  # MLP step (less tested with attributes)
                current_total_loss = step_fn(
                    node_model, edge_model, data_batch, criterion_edges,
                    optim_node_model, optim_edge_model,  # optim_edge_model used for edge_mlp
                    scheduler_node_model, scheduler_edge_model,  # scheduler_edge_model for edge_mlp
                    device, use_edge_features,
                    criterion_node_attr, node_attribute_loss_weight  # Pass new params
                )
                train_loss_sum += current_total_loss

            writer.add_scalar('loss/total_loss', current_total_loss, global_step)
            if step_fn == train_rnn_step:
                writer.add_scalar('loss/edge_loss', current_edge_loss, global_step)
                if num_node_classes > 0:
                    writer.add_scalar('loss/node_attribute_loss', current_node_attr_loss, global_step)

            if global_step % config['train']['print_iter'] == 0:
                running_time = time.time() - start_time
                time_per_iter = running_time / (global_step - start_step) if (global_step - start_step) > 0 else 0
                eta_seconds = (config['train']['steps'] - global_step) * time_per_iter if time_per_iter > 0 else 0

                avg_total_loss = train_loss_sum / config['train']['print_iter']
                log_message = "[{}] total_loss={:.4f}".format(global_step, avg_total_loss)
                if step_fn == train_rnn_step:
                    avg_edge_loss = edge_loss_sum / config['train']['print_iter']
                    log_message += " edge_loss={:.4f}".format(avg_edge_loss)
                    if num_node_classes > 0:
                        avg_node_attr_loss = node_attr_loss_sum / config['train']['print_iter']
                        log_message += " node_attr_loss={:.4f}".format(avg_node_attr_loss)

                log_message += " time_per_iter={:.4f}s eta={}".format(time_per_iter,
                                                                      datetime.timedelta(seconds=int(eta_seconds)))
                print(log_message)

                train_loss_sum = 0
                edge_loss_sum = 0
                node_attr_loss_sum = 0

            if global_step % config['train']['checkpoint_iter'] == 0 or global_step + 1 > config['train']['steps']:
                state = {
                    "global_step": global_step,
                    "config": config,
                    "node_model": node_model.state_dict(),
                    "edge_model": edge_model.state_dict(),
                    "optim_node_model": optim_node_model.state_dict(),
                    "optim_edge_model": optim_edge_model.state_dict(),
                    "scheduler_node_model": scheduler_node_model.state_dict(),
                    "scheduler_edge_model": scheduler_edge_model.state_dict(),
                    "criterion_edges": criterion_edges.state_dict(),
                }
                if criterion_node_attr is not None:  # Save node_attr criterion state if it exists
                    state["criterion_node_attr"] = criterion_node_attr.state_dict()

                print("Saving checkpoint...")
                torch.save(state, os.path.join(checkpoint_dir, "checkpoint-{}.pth".format(global_step)))

    writer.close()
    print("Training finished.")

