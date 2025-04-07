"""
KQI (Key Quality Indicator) Processing Pipeline

This script implements a comprehensive workflow for:
1. KQI distribution analysis and visualization
2. Outlier detection and filtering
3. Data normalization and classification
4. ML model training and evaluation

Main Components:
- Data preprocessing and quality control
- Statistical analysis with boxplot visualization
- Dataset balancing and synthetic data generation
- Neural network implementation with advanced features
- Model performance evaluation metrics
"""

import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE


def kqi_process(input_filename: str) -> tuple:
    """
    Process KQI data from CSV file and analyze distribution.

    Args:
        input_filename (str): Path to input CSV file

    Returns:
        tuple: Contains three elements:
            - raw_kqi (list): Original KQI values
            - log_kqi (list): Log-transformed KQI values
            - paper_ids (list): Corresponding paper IDs

    Processing Steps:
        1. Load CSV data and extract KQI values
        2. Handle zero values and invalid entries
        3. Apply log transformation for normalization
        4. Generate distribution visualizations
    """
    raw_kqi, log_kqi, paper_ids = [], [], []

    with open(input_filename, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            paper_id = row[0]
            kqi_value = float(row[5]) if row[5] else 0.0

            raw_kqi.append(kqi_value)
            if kqi_value > 0:
                log_kqi.append(math.log(kqi_value))
                paper_ids.append(paper_id)

    # Generate distribution visualizations
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.boxplot(raw_kqi)
    plt.title('Raw KQI Distribution')

    plt.subplot(1, 2, 2)
    plt.boxplot(log_kqi)
    plt.title('Log-Transformed KQI Distribution')
    plt.show()

    return raw_kqi, log_kqi, paper_ids


def calculate_boxplot_stats(paper_ids: list, data: list) -> tuple:
    """
    Calculate boxplot statistics and identify outliers.

    Args:
        paper_ids (list): Paper identifiers
        data (list): Processed KQI values

    Returns:
        tuple: Contains eight elements with statistical measures:
            - Quartile values (q1, q3, iqr)
            - Boundary values
            - Outlier/non-outlier lists
            - Filtered paper IDs

    Methodology:
        Uses Tukey's fences method with 1.5*IQR threshold
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers, non_outliers = [], []
    filtered_ids = []

    for idx, value in enumerate(data):
        if value < lower_bound or value > upper_bound:
            outliers.append(value)
        else:
            non_outliers.append(value)
            filtered_ids.append(paper_ids[idx])

    return q1, q3, iqr, lower_bound, upper_bound, outliers, non_outliers, filtered_ids


class EnhancedMLP(nn.Module):
    """
    Advanced MLP model with regularization and residual connections.

    Architecture Features:
    - Multiple hidden layers with configurable dimensions
    - Batch normalization and dropout regularization
    - Residual skip connections
    - Xavier weight initialization

    Args:
        input_dim (int): Dimension of input features
        hidden_dim (list): List of hidden layer dimensions
        output_dim (int): Number of output classes
        use_residual (bool): Enable residual connections
        dropout_rate (float): Dropout probability
    """

    def __init__(self, input_dim: int, hidden_dim: list,
                 output_dim: int, use_residual=True, dropout_rate=0.5):
        super().__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()

        # Dynamic layer construction
        for i, h_dim in enumerate(hidden_dim):
            in_features = input_dim if i == 0 else hidden_dim[i - 1]
            self.layers.append(nn.Sequential(
                nn.Linear(in_features, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            nn.init.xavier_normal_(self.layers[-1][0].weight)

        self.output_layer = nn.Linear(hidden_dim[-1], output_dim)
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional residual connections.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features)

        Returns:
            torch.Tensor: Output logits
        """
        residual = x
        for layer in self.layers:
            x = layer(x)
            if self.use_residual and x.shape == residual.shape:
                x += residual
            residual = x
        return self.output_layer(x)


def train_evaluation_pipeline(X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray) -> None:
    """
    Complete model training and evaluation workflow.

    Steps:
        1. Handle class imbalance with SMOTE
        2. Convert data to PyTorch tensors
        3. Initialize model and optimizer
        4. Training loop with loss monitoring
        5. Performance evaluation on test set
    """
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_resampled)
    y_train_tensor = torch.LongTensor(y_resampled)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Model configuration
    model = EnhancedMLP(input_dim=4, hidden_dim=[64, 128], output_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(4000):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        _, predicted = torch.max(preds, 1)

    print(classification_report(y_test, predicted.numpy()))
    print(f"Macro F1-Score: {f1_score(y_test, predicted.numpy(), average='macro')}")


# Main execution flow
if __name__ == "__main__":
    # Data processing
    input_file = '/path/to/non_data.csv'
    raw_kqi, log_kqi, paper_ids = kqi_process(input_file)

    # Outlier handling
    _, _, _, _, _, _, clean_kqi, clean_ids = calculate_boxplot_stats(paper_ids, log_kqi)

    # Model training
    data = pd.read_csv(input_file)
    features = data[['reference_count', 'hindex',
                     'reference_years_difference', 'hindex_ave']].values
    labels = data['kqi_class'].values

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    train_evaluation_pipeline(X_train, X_test, y_train, y_test)