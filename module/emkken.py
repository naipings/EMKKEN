import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import numpy as np
import seaborn as sns
from mamst import MamST
from knu import KNU


class EMKKEN(torch.nn.Module):
    """
    Efficient Mamba-KAN Knowledge Evaluator for spatio-temporal graph learning.

    Integrates MamST (Mamba-sequence Transformer) and KNU (KANflex Neural Unit)
    into a unified architecture for end-to-end graph representation learning.

    Args:
        mamba_num_node_features (int): Input metadata dimension for MamST
        mamba_hidden_dim (int): Hidden dimension for MamST's Mamba blocks
        mamba_d_state1 (int, optional): State dimension for MamST's first Mamba. Default: 16
        mamba_d_state2 (int, optional): State dimension for MamST's second Mamba. Default: 16
        mamba_d_conv1 (int, optional): Conv kernel size for first Mamba. Default: 4
        mamba_d_conv2 (int, optional): Conv kernel size for second Mamba. Default: 4
        mamba_dropout_rate1 (float, optional): Dropout after first Mamba. Default: 0.5
        mamba_dropout_rate2 (float, optional): Dropout after second Mamba. Default: 0.5
        knu_output_dim (int, optional): Output classes for KNU. Default: 256
        knu_mlp_hidden_dim (int, optional): KAN hidden dimension. Default: 128
        knu_dropout_rate (float, optional): Dropout in KNU. Default: 0.5
    """
    def __init__(self,
                 # MamST parameters
                 mamba_num_node_features,
                 mamba_hidden_dim,
                 mamba_d_state1=16,
                 mamba_d_state2=16,
                 mamba_d_conv1=4,
                 mamba_d_conv2=4,
                 mamba_dropout_rate1=0.5,
                 mamba_dropout_rate2=0.5,

                 # KNU parameters
                 knu_output_dim=256,
                 knu_mlp_hidden_dim=128,
                 knu_dropout_rate=0.5
                 ):
        super(EMKKEN, self).__init__()

        # Feature processing module
        self.mamst = MamST(
            num_node_features=mamba_num_node_features,
            hidden_dim=mamba_hidden_dim,
            d_state1=mamba_d_state1,
            d_state2=mamba_d_state2,
            d_conv1=mamba_d_conv1,
            d_conv2=mamba_d_conv2,
            dropout_rate1=mamba_dropout_rate1,
            dropout_rate2=mamba_dropout_rate2
        )

        # Classification module
        self.knu = KNU(
            MamST=self.mamst,
            num_node_features=mamba_hidden_dim,
            output_dim=knu_output_dim,
            mlp_hidden_dim=knu_mlp_hidden_dim,
            dropout_rate=knu_dropout_rate
        )

    def forward(self, data) -> tuple:
        """
        End-to-end forward pass through the integrated architecture.

        Args:
            data (Data): Input graph data following PyG conventions

        Returns:
            tuple: Outputs from KNU module containing predictions, labels,
                   node identifiers, and regularization loss
        """
        return self.knu(data)
