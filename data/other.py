class SiameseDagformer(nn.Module):
    """
    A Siamese network using the GraphTransformer (Dagformer) as the backbone.

    This network processes two graphs, generates an embedding for each,
    combines them, and then passes them through a prediction head to produce
    the final output.
    """
    def __init__(self, dagformer_params, output_dim=16, hidden_dim_ratio=0.5):
        """
        Args:
            dagformer_params (dict): A dictionary of parameters to initialize the GraphTransformer.
            output_dim (int): The dimension of the final output vector.
            hidden_dim_ratio (float): The ratio of the hidden layer size in the prediction
                                      head relative to the dagformer's model dimension.
        """
        super().__init__()
        
        # 1. Shared Backbone: Instantiate a single Dagformer.
        #    This same instance will be used to process both input graphs.
        self.dagformer = GraphTransformer(**dagformer_params)
        
        d_model = dagformer_params.get('d_model', 512)
        
        # 2. Pooling Layer: We need a single vector representation for each graph.
        #    We'll use the pooling method specified in the dagformer params.
        self.pooling = self.dagformer.pooling

        # 3. Prediction Head: An MLP that takes the combined graph embeddings
        #    and maps them to the final desired output dimension.
        hidden_dim = int(d_model * hidden_dim_ratio)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data1, data2):
        """
        Forward pass for the Siamese network.

        Args:
            data1 (torch_geometric.data.Data): The first graph (or batch of graphs).
            data2 (torch_geometric.data.Data): The second graph (or batch of graphs).
        
        Returns:
            torch.Tensor: The final prediction vector of shape (batch_size, output_dim).
        """
        
        # --- Process Graph 1 ---
        # Get node-level embeddings from the shared backbone.
        node_embedding1 = self.dagformer.forward_encoder(data1)
        # Pool the node embeddings to get a single graph-level embedding.
        graph_embedding1 = self.pooling(node_embedding1, data1.batch)
        
        # --- Process Graph 2 ---
        # Get node-level embeddings for the second graph using the same backbone.
        node_embedding2 = self.dagformer.forward_encoder(data2)
        # Pool to get the graph-level embedding.
        graph_embedding2 = self.pooling(node_embedding2, data2.batch)
        
        # --- Combine and Predict ---
        # Combine the two graph embeddings. The absolute difference is a common
        # and effective choice for learning a distance or similarity metric.
        combined_embedding = torch.abs(graph_embedding1 - graph_embedding2)
        
        # Pass the combined representation through the final prediction head.
        prediction = self.prediction_head(combined_embedding)
        
        return prediction

# TO TRY 
# combined = torch.cat([graph_embedding1, graph_embedding2, torch.abs(graph_embedding1 - graph_embedding2), graph_embedding1 * graph_embedding2], dim=-1)
# Attention-Based Comparison for prediction head
# Node-Level Comparison - pre pooling cross attention 
