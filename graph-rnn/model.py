import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class GraphLevelRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 num_node_classes, output_size=None, edge_feature_len=1):
        """
        Arguments:
            input_size: Length of the padded adjacency vector (m). This is the number of
                        potential previous nodes an edge can connect to.
            embedding_size: Size of the input embedding fed to the GRU.
            hidden_size: Hidden size of the GRU.
            num_layers: Number of GRU layers.
            num_node_classes: Number of unique node classes/types for attribute prediction.
            output_size: Size of the final output for the edge-level RNN's initial hidden state.
                         Set to None if the GRU's hidden state is used directly.
            edge_feature_len: Number of features associated with each edge.
                Default is 1 (i.e., scalar value 0/1 indicating whether the
                edge is set or not).
        """
        super().__init__()
        self.input_size = input_size  # This is 'm' from the paper/configs
        self.edge_feature_len = edge_feature_len
        self.num_node_classes = num_node_classes

        # The input to this linear layer will be the flattened adjacency vector of the previous node
        # (connections to its 'm' predecessors) concatenated with the one-hot encoded
        # attribute of that *same* previous node.
        self.combined_input_dim = (input_size * edge_feature_len) + num_node_classes
        self.linear_in = nn.Linear(self.combined_input_dim, embedding_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        # Output layer for node attribute prediction (predicts attributes for the *current* node being generated)
        self.node_attribute_predictor = nn.Linear(hidden_size, num_node_classes)

        # Optional output layers if 'output_size' is specified (used to transform GRU hidden state
        # before passing to EdgeLevelRNN as its initial hidden state)
        if output_size:
            self.linear_out1_edge = nn.Linear(hidden_size, embedding_size)
            self.linear_out2_edge = nn.Linear(embedding_size, output_size)
        else:
            self.linear_out1_edge = None
            self.linear_out2_edge = None

        self.hidden = None  # GRU hidden state

    def reset_hidden(self):
        """ Resets the GRU hidden state to None (PyTorch will init with zeros). """
        self.hidden = None

    def forward(self, x_adj, prev_node_attributes, x_lens=None):
        """
        Processes one step of graph generation (i.e., for one node).
        Arguments:
            x_adj: Adjacency information of the *previous* node that was generated/processed.
                   Tensor of shape [batch, seq_len, self.input_size, self.edge_feature_len].
                   `seq_len` is typically 1 during iterative generation, or longer during training.
                   `self.input_size` is 'm' (max predecessors).
            prev_node_attributes: One-hot encoded attributes of the *previous* node.
                                  Tensor of shape [batch, seq_len, self.num_node_classes].
            x_lens: List of sequence lengths (number of nodes in each graph in the batch).
                    Used for packing sequences during training. On CPU.

        Returns:
            A tuple: (output_for_edge_rnn, node_attribute_logits)
            - output_for_edge_rnn: Hidden state to initialize the EdgeLevelRNN for the *current* node.
                                   Shape [batch, seq_len, hidden_size or output_size].
            - node_attribute_logits: Logits for predicting attributes of the *current* node.
                                     Shape [batch, seq_len, num_node_classes].
        """
        # Flatten the adjacency part of the input (from previous node)
        # x_adj shape: [batch, seq_len, self.input_size (m), self.edge_feature_len]
        x_adj_flat = torch.flatten(x_adj, start_dim=2, end_dim=3)
        # x_adj_flat shape: [batch, seq_len, self.input_size * self.edge_feature_len]

        # Concatenate previous node's flattened adjacency with its attributes
        # prev_node_attributes shape: [batch, seq_len, self.num_node_classes]
        combined_input = torch.cat((x_adj_flat, prev_node_attributes), dim=-1)
        # combined_input shape: [batch, seq_len, (self.input_size * self.edge_feature_len) + self.num_node_classes]

        # Pass through input embedding layer
        embedded_input = self.relu(self.linear_in(combined_input))
        # embedded_input shape: [batch, seq_len, embedding_size]

        # Pack data if lengths are provided (for training efficiency)
        if x_lens is not None:
            cpu_x_lens = x_lens.cpu() if x_lens.is_cuda else x_lens  # Ensure lengths are on CPU
            embedded_input = pack_padded_sequence(embedded_input, cpu_x_lens, batch_first=True, enforce_sorted=False)

        # Pass through GRU
        gru_output, self.hidden = self.gru(embedded_input, self.hidden)
        # gru_output (packed) shape: [sum_of_lengths, hidden_size]

        # Unpack (reintroduces padding) if data was packed
        if x_lens is not None:
            gru_output, _ = pad_packed_sequence(gru_output, batch_first=True)
        # gru_output (unpacked) shape: [batch, seq_len, hidden_size]

        # Predict node attributes for the current node from GRU's hidden state
        node_attribute_logits = self.node_attribute_predictor(gru_output)
        # node_attribute_logits shape: [batch, seq_len, num_node_classes]

        # Determine the output to be used as initial hidden state for the EdgeLevelRNN
        output_for_edge_rnn = gru_output
        if self.linear_out1_edge and self.linear_out2_edge:  # Optional transformation
            output_for_edge_rnn = self.relu(self.linear_out1_edge(gru_output))
            output_for_edge_rnn = self.linear_out2_edge(output_for_edge_rnn)

        return output_for_edge_rnn, node_attribute_logits


class EdgeLevelRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, edge_feature_len=1):
        """
        Arguments:
            embedding_size: Size of the input embedding fed to the GRU (for edge features).
            hidden_size: Hidden size of the GRU.
            num_layers: Number of GRU layers.
            edge_feature_len: Number of features associated with each edge.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.edge_feature_len = edge_feature_len

        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.linear_out1 = nn.Linear(hidden_size, embedding_size)
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None  # GRU hidden state

    def set_first_layer_hidden(self, h_graph_level):
        """
        Sets the hidden state of the first GRU layer using output from GraphLevelRNN.
        The hidden state of all other GRU layers will be reset to 0.

        Arguments:
            h_graph_level: Hidden state from GraphLevelRNN for the current node.
                           Expected shape for training (packed): [current_edge_batch_size, graph_level_output_size].
                           Expected shape for generation (single node): [1, graph_level_output_size] or [batch_size_1, graph_level_output_size].
                           The GRU expects hidden state of shape [num_layers, batch_for_gru, hidden_size_for_gru].
        """
        # h_graph_level is the output from GraphLevelRNN, its size should match self.hidden_size (or EdgeRNN's embedding if transformed)
        # It's used as the initial hidden state for the *first layer* of this EdgeLevelRNN's GRU.

        # Reshape h_graph_level to [1, batch_for_gru, hidden_size_for_gru] to represent the first layer's hidden state.
        if len(h_graph_level.shape) == 2:  # e.g., [current_edge_batch_size, hidden_size_for_gru]
            h_first_layer = h_graph_level.unsqueeze(0)  # Becomes [1, current_edge_batch_size, hidden_size_for_gru]
        elif len(h_graph_level.shape) == 3:  # e.g., [batch_size_1, 1, hidden_size_for_gru] during generation
            h_first_layer = h_graph_level.transpose(0, 1)  # Becomes [1, batch_size_1, hidden_size_for_gru]
        else:
            raise ValueError(f"Unexpected shape for h_graph_level: {h_graph_level.shape}")

        if self.num_layers > 1:
            # Create zero states for the remaining layers
            zeros = torch.zeros([self.num_layers - 1, h_first_layer.shape[1], h_first_layer.shape[2]],
                                device=h_first_layer.device)
            self.hidden = torch.cat([h_first_layer, zeros], dim=0)
        else:  # num_layers is 1
            self.hidden = h_first_layer

    def forward(self, x_edge_seq, x_edge_lens=None, return_logits=False):
        """
        Processes a sequence of edge connections for the current node.
        Arguments:
            x_edge_seq: Input tensor of shape [batch, seq_len_edges, edge_feature_len].
                        Represents the sequence of edge connections to previous nodes
                        (e.g., SOS token, then actual edge features).
            x_edge_lens: List of sequence lengths (number of edges to predict for each node in batch).
                         On CPU. Used for packing.
            return_logits: If True, output raw logits. Otherwise, apply activation.

        Returns:
            Predicted edge features/probabilities of shape [batch, seq_len_edges, edge_feature_len].
        """
        assert self.hidden is not None, "Hidden state not set for EdgeLevelRNN! Call set_first_layer_hidden."

        x_embedded = self.relu(self.linear_in(x_edge_seq))

        if x_edge_lens is not None:
            cpu_x_edge_lens = x_edge_lens.cpu() if x_edge_lens.is_cuda else x_edge_lens
            x_embedded = pack_padded_sequence(x_embedded, cpu_x_edge_lens, batch_first=True, enforce_sorted=False)

        gru_output, self.hidden = self.gru(x_embedded, self.hidden)

        if x_edge_lens is not None:
            gru_output, _ = pad_packed_sequence(gru_output, batch_first=True)

        x_out_transformed = self.relu(self.linear_out1(gru_output))
        edge_predictions = self.linear_out2(x_out_transformed)

        if not return_logits:
            if self.edge_feature_len == 1:  # Binary edge prediction
                edge_predictions = self.sigmoid(edge_predictions)
            # For edge_feature_len > 1 (multi-class edges), logits are typically returned
            # as CrossEntropyLoss (used in original train.py) applies softmax internally.
        return edge_predictions


class EdgeLevelMLP(nn.Module):
    def __init__(self, input_size_from_graph_rnn, mlp_hidden_size, num_edges_to_predict, edge_feature_len=1):
        """
        Arguments:
            input_size_from_graph_rnn: Size of the hidden state from GraphLevelRNN.
            mlp_hidden_size: Size of the MLP's hidden layer.
            num_edges_to_predict: Number of edge probabilities to output (typically 'm').
            edge_feature_len: Number of features for each edge.
        """
        super().__init__()
        self.edge_feature_len = edge_feature_len
        self.linear1 = nn.Linear(input_size_from_graph_rnn, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_edges_to_predict * edge_feature_len)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_graph_level, return_logits=False):
        """
        Arguments:
            h_graph_level: Hidden state from GraphLevelRNN.
                           Shape [batch, seq_len, input_size_from_graph_rnn].
                           `seq_len` is typically 1 for MLP edge prediction per node.
            return_logits: If True, output raw logits.

        Returns:
            Predicted edge features/probabilities.
            Shape [batch, seq_len, num_edges_to_predict, edge_feature_len].
        """
        hidden_mlp_state = self.relu(self.linear1(h_graph_level))
        edge_predictions_flat = self.linear2(hidden_mlp_state)
        # Shape: [batch, seq_len, num_edges_to_predict * edge_feature_len]

        if not return_logits:
            if self.edge_feature_len == 1:  # Binary edge prediction
                edge_predictions_flat = self.sigmoid(edge_predictions_flat)
            # For multi-class edges, logits are typically preferred for CrossEntropyLoss.

        # Reshape to separate edge predictions and their features
        batch_size = h_graph_level.shape[0]
        seq_len = h_graph_level.shape[1]  # Typically 1 if MLP predicts all edges at once for a node
        # num_edges_to_predict was passed in __init__
        num_edge_slots = edge_predictions_flat.shape[-1] // self.edge_feature_len

        edge_predictions = edge_predictions_flat.view(batch_size, seq_len, num_edge_slots, self.edge_feature_len)
        # Target shape: [batch, seq_len, num_edges_to_predict, self.edge_feature_len]

        return edge_predictions


d