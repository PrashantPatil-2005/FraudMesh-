
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear


class FraudGAT(nn.Module):
    """
    Graph Attention Network for fraud detection on heterogeneous graph.
    
    Architecture:
    1. Separate linear projections for card and merchant node features
       (because they have different feature dimensions)
    2. Two HeteroConv layers using GATConv
       (message passing happens along both edge types)
    3. Final linear classifier on card nodes only
    
    Args:
        card_feature_dim: Number of input features for card nodes
        merchant_feature_dim: Number of input features for merchant nodes
        hidden_dim: Hidden layer dimension (default 64)
        num_heads: Number of attention heads (default 4)
        num_layers: Number of GAT layers (default 2)
        dropout: Dropout rate (default 0.3)
        output_dim: Number of output classes (default 2: fraud/not fraud)
    """
    
    def __init__(
        self,
        card_feature_dim,
        merchant_feature_dim,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
        output_dim=2
    ):
        super(FraudGAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # ================================================================
        # Input Projections
        # ================================================================
        # Project card and merchant features to same hidden dimension
        # This is necessary because they have different feature counts
        self.card_input = Linear(card_feature_dim, hidden_dim)
        self.merchant_input = Linear(merchant_feature_dim, hidden_dim)
        
        # ================================================================
        # HeteroConv Layers
        # ================================================================
        # Each HeteroConv layer contains GATConv for each edge type
        # GATConv(in_channels, out_channels, heads)
        # Output dim of multi-head GAT = out_channels * heads
        # So we need to account for this in subsequent layers
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Input dimension for this layer
            if i == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim * num_heads  # Previous layer output with multi-head
            
            # For last layer, use 1 head and output hidden_dim
            # For other layers, use num_heads
            if i == num_layers - 1:
                heads = 1
                out_dim = hidden_dim
            else:
                heads = num_heads
                out_dim = hidden_dim
            
            # Create HeteroConv with GATConv for each edge type
            conv = HeteroConv({
                # Card → Merchant messages
                ('card', 'transacts_at', 'merchant'): GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False  # Bipartite graph, no self-loops
                ),
                # Merchant → Card messages
                ('merchant', 'rev_transacts_at', 'card'): GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False
                ),
            }, aggr='sum')  # Aggregate messages from different edge types by sum
            
            self.convs.append(conv)
            
            # Batch normalization for each node type
            norm_dim = out_dim * heads if i < num_layers - 1 else out_dim
            self.norms.append(nn.ModuleDict({
                'card': nn.BatchNorm1d(norm_dim),
                'merchant': nn.BatchNorm1d(norm_dim)
            }))
        
        # ================================================================
        # Output Classifier (card nodes only)
        # ================================================================
        self.classifier = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass.
        
        Args:
            x_dict: Dictionary of node features
                {'card': Tensor, 'merchant': Tensor}
            edge_index_dict: Dictionary of edge indices
                {('card', 'transacts_at', 'merchant'): Tensor,
                 ('merchant', 'rev_transacts_at', 'card'): Tensor}
                 
        Returns:
            torch.Tensor: Logits for card nodes, shape [num_cards, 2]
        """
        # ================================================================
        # Input Projection
        # ================================================================
        x_dict = {
            'card': self.card_input(x_dict['card']),
            'merchant': self.merchant_input(x_dict['merchant'])
        }
        
        # ================================================================
        # Message Passing Layers
        # ================================================================
        for i in range(self.num_layers):
            # Apply HeteroConv (GATConv on each edge type)
            x_dict_new = self.convs[i](x_dict, edge_index_dict)
            
            # Apply batch normalization
            for node_type in x_dict_new:
                if node_type in self.norms[i]:
                    x_dict_new[node_type] = self.norms[i][node_type](
                        x_dict_new[node_type]
                    )
            
            # Apply activation and dropout
            x_dict_new = {
                key: F.dropout(F.relu(val), p=self.dropout, training=self.training)
                for key, val in x_dict_new.items()
            }
            
            # Residual connection (if dimensions match)
            for key in x_dict_new:
                if key in x_dict and x_dict[key].shape == x_dict_new[key].shape:
                    x_dict_new[key] = x_dict_new[key] + x_dict[key]
            
            x_dict = x_dict_new
        
        # ================================================================
        # Classification (card nodes only)
        # ================================================================
        card_embeddings = x_dict['card']
        logits = self.classifier(card_embeddings)
        
        return logits
    
    def get_embeddings(self, x_dict, edge_index_dict):
        """
        Get card node embeddings BEFORE the classifier.
        Useful for visualization (t-SNE, UMAP).
        
        Returns:
            torch.Tensor: Card embeddings, shape [num_cards, hidden_dim]
        """
        x_dict = {
            'card': self.card_input(x_dict['card']),
            'merchant': self.merchant_input(x_dict['merchant'])
        }
        
        for i in range(self.num_layers):
            x_dict_new = self.convs[i](x_dict, edge_index_dict)
            
            for node_type in x_dict_new:
                if node_type in self.norms[i]:
                    x_dict_new[node_type] = self.norms[i][node_type](
                        x_dict_new[node_type]
                    )
            
            x_dict_new = {
                key: F.dropout(F.relu(val), p=self.dropout, training=self.training)
                for key, val in x_dict_new.items()
            }
            
            for key in x_dict_new:
                if key in x_dict and x_dict[key].shape == x_dict_new[key].shape:
                    x_dict_new[key] = x_dict_new[key] + x_dict[key]
            
            x_dict = x_dict_new
        
        return x_dict['card']
