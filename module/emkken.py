import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import numpy as np
import seaborn as sns
from mamst import MamST
from knu import KNU
from torch_geometric.data import Data  # 确保导入 Data 对象
from torchsummary import summary


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
                 mamba_hidden_dim=256,
                 mamba_d_state1=16,
                 mamba_d_state2=8,
                 mamba_d_conv1=4,
                 mamba_d_conv2=2,
                 mamba_dropout_rate1=0.5,
                 mamba_dropout_rate2=0.5,

                 # KNU parameters
                 knu_output_dim=64,
                 knu_mlp_hidden_dim=64,
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
            num_node_features=mamba_num_node_features,
            input_node_features=mamba_hidden_dim,
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


if __name__ == "__main__":
    # 初始化模型
    model = EMKKEN(mamba_num_node_features=4, mamba_hidden_dim=64)
    print('===============================================================')
    print('model', model)

    # 构建输入数据
    # 假设有一个图，包含 3 个节点和 2 条边
    num_nodes = 3
    num_edges = 2
    # 节点特征 (3 个节点，每个节点有 4 个特征)
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float)
    # 边索引 (2 条边)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).T
    # 图标签
    y = torch.tensor([0], dtype=torch.long)
    # 构建 Data 对象
    data = Data(x=x, edge_index=edge_index, y=y)

    # 打印模型参数
    print('===============================================================')
    print('Model Parameters:')
    for name, param in model.named_parameters():
        print(f'{name}: {param.shape}')

    print('===============================================================')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    # # 使用 torchsummary 打印模型结构
    # print('===============================================================')
    # print('Model Summary:')
    # summary(model=model, input_size=data, device="cuda")