from mamba_ssm import Mamba, Mamba2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import seaborn as sns


class MamST(torch.nn.Module):
    """
    Mamba-sequence Transformer feature processor for graph-structured data.

    This module processes both node metadata and embeddings using Mamba blocks,
    effectively capturing spatio-temporal patterns in graph data.

    Args:
        num_node_features (int): Number of raw metadata features per node
        hidden_dim (int): Dimension of hidden representations
        d_state1 (int, optional): State dimension for first Mamba block. Default: 16
        d_state2 (int, optional): State dimension for second Mamba block. Default: 16
        d_conv1 (int, optional): Convolution width for first Mamba block. Default: 4
        d_conv2 (int, optional): Convolution width for second Mamba block. Default: 4
        dropout_rate1 (float, optional): Dropout rate after first Mamba block. Default: 0.5
        dropout_rate2 (float, optional): Dropout rate after second Mamba block. Default: 0.5
    """

    def __init__(self,
                 num_node_features,
                 hidden_dim,
                 d_state1=16,
                 d_state2=16,
                 d_conv1=4,
                 d_conv2=4,
                 dropout_rate1=0.5,
                 dropout_rate2=0.5
                 ):
        super(MamST, self).__init__()
        self.features = num_node_features

        # Metadata processing layers
        if self.features > 0:
            self.fc = nn.Linear(1, hidden_dim // num_node_features)
        else:
            self.fc = nn.Linear(1, hidden_dim)  # When features are set to 0, directly set it to hidden_im

        self.relu = nn.ReLU()
        self.meta_fc = nn.Linear(hidden_dim + 1, hidden_dim)

        # Dual Mamba architecture
        self.MC1 = Mamba(d_model=hidden_dim, d_state=d_state1, d_conv=d_conv1)
        self.MC2 = Mamba(d_model=768, d_state=d_state2, d_conv=d_conv2)
        # self.MC2 = Mamba2(d_model=768, d_state=64, d_conv=4)

        # Regularization
        self.dropout1 = nn.Dropout(dropout_rate1)
        self.dropout2 = nn.Dropout(dropout_rate2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, date_encoding: torch.Tensor = None) -> tuple:
        """
        Forward pass through the MamST processor.

        Args:
            x (torch.Tensor): Input node features of shape (num_nodes, num_features)
            edge_index (torch.Tensor): Graph connectivity in COO format (2, num_edges)
            date_encoding (torch.Tensor, optional): Temporal encoding vector of shape (num_nodes,). Default: None

        Returns:
            tuple: Processed metadata (torch.Tensor) and embeddings (torch.Tensor)
        """
        # Split features into metadata and embeddings
        if self.features > 0:
            embedding = x[:, self.features:]
            meta = x[:, :self.features]
        else:
            # If there is no metadata, simply use x as the embedding
            embedding = x

        # Process metadata features
        if self.features > 0:
            meta = meta.unsqueeze(2)
            meta = self.fc(meta)
            meta = self.relu(meta)
            meta = meta.view(meta.size(0), -1)

        # Incorporate temporal encoding
        if date_encoding is not None:
            date_encoding = date_encoding.unsqueeze(-1) if date_encoding.dim() == 1 else date_encoding
            if self.features > 0:
                meta = torch.cat([meta, date_encoding], dim=-1)
            else:
                meta = date_encoding

        if self.features > 0:
            meta = self.meta_fc(meta)
            meta = self.relu(meta)

        # Mamba processing for metadata
        if self.features > 0:
            meta = meta.unsqueeze(1)  # (batch_size, 1, feature_dim)
            meta = self.MC1(meta)
            meta = self.dropout1(meta)
            meta = meta.squeeze(1)  # (N, hidden_dim)

        # Mamba processing for embeddings
        embedding = embedding.unsqueeze(1)
        embedding = self.MC2(embedding)
        embedding = self.dropout2(embedding)
        embedding = embedding.squeeze(1)

        if self.features > 0:
            return meta, embedding
        else:
            return embedding