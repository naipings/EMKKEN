import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import seaborn as sns
from mamst import MamST


class KNU(nn.Module):
    """
    KANflex Neural Unit with Kolmogorov-Arnold Networks (KAN) for graph-centric classification.

    This module processes center node features using KAN layers and combines them for
    final prediction, incorporating regularization through KAN's structural properties.

    Args:
        MamST (nn.Module): Pretrained Mamba-based spatio-temporal processor
        num_node_features (int): Number of raw metadata features per node
        input_node_features (int): Dimension of processed node features from MamST
        output_dim (int): Number of target classes
        mlp_hidden_dim (int): Hidden dimension for KAN layers
        dropout_rate (float, optional): Dropout rate after KAN layers. Default: 0.5
    """

    def __init__(self,
                 MamST,
                 num_node_features,
                 input_node_features,
                 output_dim,
                 mlp_hidden_dim,
                 dropout_rate=0.5
                 ):
        super(KNU, self).__init__()
        self.MamST = MamST
        self.num_node_features = num_node_features

        # KAN-based feature processors
        self.meta_kan = KAN([input_node_features, mlp_hidden_dim // 2], base_activation=nn.ReLU)
        self.embedding_kan = KAN([768, mlp_hidden_dim // 2], base_activation=nn.ReLU)

        # Prediction layers
        self.final_fc = nn.Linear(mlp_hidden_dim, output_dim)
        self.final_fc2 = nn.Linear(mlp_hidden_dim // 2, output_dim)
        self.softmax = nn.Softmax(dim=1)

        # Regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data) -> tuple:
        """
        Forward pass with graph data processing and classification.

        Args:
            data (Data): PyG-style graph data object containing:
                - x: Node features
                - edge_index: Graph connectivity
                - date_encoding: Temporal encoding
                - is_center: Center node mask
                - y: Node labels
                - valid_node_ids: Valid node identifiers
                - center_id: Center node ID

        Returns:
            tuple: Contains:
                - predictions (torch.Tensor): Class probabilities (num_centers, output_dim)
                - labels (torch.Tensor): Ground truth labels (num_centers,)
                - valid_ids (torch.Tensor): Valid node IDs
                - center_id (torch.Tensor): Center node ID
                - reg_loss (torch.Tensor): KAN regularization loss
        """
        # Feature extraction
        if hasattr(data, 'date_encoding') and data.date_encoding is not None:
            result = self.MamST(data.x, data.edge_index, data.date_encoding)
        else:
            result = self.MamST(data.x, data.edge_index)

        if self.num_node_features > 0:
            meta, embedding = result
        else:
            embedding = result

        if self.num_node_features > 0:
            # Center node selection
            center_mask = data.is_center.bool()
            center_meta = meta[center_mask]
            center_embedding = embedding[center_mask]

            # KAN-based feature transformation
            center_meta_out = self.meta_kan(center_meta)
            center_meta_out = self.dropout(center_meta_out)
            center_embedding_out = self.embedding_kan(center_embedding)
            center_embedding_out = self.dropout(center_embedding_out)

            # Feature fusion
            combined = torch.cat([center_meta_out, center_embedding_out], dim=1)

            # Final prediction
            logits = self.final_fc(combined)
            preds = self.softmax(logits)

            # Regularization loss calculation
            reg_loss = self.meta_kan.regularization_loss(1, 0) + self.embedding_kan.regularization_loss(1, 0)
        else:
            # If there is no metadata, use embedding directly
            center_mask = data.is_center.bool()
            center_embedding = embedding[center_mask]

            # KAN-based feature transformation
            center_embedding_out = self.embedding_kan(center_embedding)
            center_embedding_out = self.dropout(center_embedding_out)

            # Final prediction
            logits = self.final_fc2(center_embedding_out)
            preds = self.softmax(logits)

            # Regularization loss calculation
            reg_loss = self.embedding_kan.regularization_loss(1, 0)

        return preds, data.y[center_mask], data.valid_node_ids, data.center_id, reg_loss


class KANLinear(torch.nn.Module):
    """
    Linear layer with Kernel Adaptive Network (KAN) capabilities combining base weights
    and spline-parameterized adaptive components with grid adjustment functionality.

    Implements learnable basis functions through B-splines and enables dynamic grid adaptation
    based on input distribution.
    """

    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        """
        Initialize KAN linear layer with dual parameterization (base + spline components).

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            grid_size (int): Number of grid points for spline interpolation. Default: 5
            spline_order (int): Order of B-splines (degree = order-1). Default: 3
            scale_noise (float): Noise magnitude for spline weight initialization. Default: 0.1
            scale_base (float): Scaling factor for base weight initialization. Default: 1.0
            scale_spline (float): Scaling factor for spline weight initialization. Default: 1.0
            enable_standalone_scale_spline (bool): Use separate scalers for spline weights. Default: True
            base_activation (torch.nn.Module): Activation function for base component. Default: SiLU
            grid_eps (float): Blending factor between uniform/adaptive grid updates. Default: 0.02
            grid_range (list): Initial grid range for spline bases. Default: [-1, 1]
        """
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Initialize spline grid with buffer registration
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Parameter initialization
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        # Configuration parameters
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize model parameters using Kaiming uniform distribution for base weights
        and controlled noise initialization for spline weights.
        """
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # Noise initialization for spline weights
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline basis values for input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)

        Returns:
            torch.Tensor: B-spline basis tensor of shape
                (batch_size, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # Iterative basis construction using De Boor's algorithm
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute spline coefficients through least squares fitting.

        Args:
            x (torch.Tensor): Input points tensor of shape (batch_size, in_features)
            y (torch.Tensor): Output values tensor of shape (batch_size, in_features, out_features)

        Returns:
            torch.Tensor: Spline coefficient tensor of shape
                (out_features, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        # Solve least squares problem for spline coefficients
        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """
        Get spline weights with applied scalers when enabled.
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        """
        Forward propagation through KANLinear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features)

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features)
        """
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # Base component computation
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Spline component computation
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        Adapt spline grid based on input data distribution.

        Args:
            x (torch.Tensor): Input data for grid adaptation of shape (batch_size, in_features)
            margin (float): Padding factor for grid range extension. Default: 0.01
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        # Calculate current spline outputs for coefficient preservation
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # Sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        # Final grid construction with boundary extensions
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.

        Args:
            regularize_activation (float): Weight for L1 regularization of spline magnitudes
            regularize_entropy (float): Weight for entropy regularization of spline distributions

        Returns:
            torch.Tensor: Combined regularization loss value
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    """
    Stacked KANLinear layers forming a complete Kernel Adaptive Network.

    Supports dynamic grid adaptation and provides configurable regularization.
    """

    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        """
        Initialize KAN network with sequential KANLinear layers.

        Args:
            layers_hidden (list): Layer dimensions [input_dim, hidden_dims..., output_dim]
            grid_size (int): Shared grid size across all layers. Default: 5
            spline_order (int): Shared spline order across all layers. Default: 3
            scale_noise (float): Noise scale factor for spline init. Default: 0.1
            scale_base (float): Base weight initialization scale. Default: 1.0
            scale_spline (float): Spline weight initialization scale. Default: 1.0
            base_activation (torch.nn.Module): Activation for base components. Default: SiLU
            grid_eps (float): Grid adaptation blending factor. Default: 0.02
            grid_range (list): Initial grid range for all layers. Default: [-1, 1]
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    # enable_standalone_scale_spline=False,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        """
        Sequential forward pass through all KANLinear layers.

        Args:
            x (torch.Tensor): Input tensor
            update_grid (bool): Enable grid adaptation during forward pass. Default: False

        Returns:
            torch.Tensor: Network output tensor
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Aggregate regularization losses from all KANLinear layers.

        Args:
            regularize_activation (float): Per-layer activation loss weight
            regularize_entropy (float): Per-layer entropy loss weight

        Returns:
            torch.Tensor: Total regularization loss
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
