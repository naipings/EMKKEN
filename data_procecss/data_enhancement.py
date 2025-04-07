from imblearn.over_sampling import SMOTE
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns


def add_noise_to_features(data: torch.Tensor, noise_factor: float = 0.01) -> torch.Tensor:
    """
    Add Gaussian noise to node features for regularization and improved generalization.

    Args:
        data (torch.Tensor): Input graph data containing node features
        noise_factor (float, optional): Standard deviation multiplier for noise. Default: 0.01

    Returns:
        torch.Tensor: Augmented data with noisy node features

    Note:
        Preserves original data structure while adding independent noise to each feature dimension
    """
    noise = torch.randn_like(data.x) * noise_factor
    data.x = data.x + noise
    return data


def random_drop_edges(data: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    """
    Randomly drop edges from graph structure to prevent over-smoothing and improve robustness.

    Args:
        data (torch.Tensor): Input graph data with edge indices
        p (float, optional): Probability of edge removal. Default: 0.2

    Returns:
        torch.Tensor: Modified graph data with subset of original edges

    Implementation:
        Applies Bernoulli sampling to create binary mask for edge selection
    """
    num_edges = data.edge_index.size(1)
    mask = torch.rand(num_edges) > p
    data.edge_index = data.edge_index[:, mask]
    return data


def normalize_features(data: torch.Tensor) -> torch.Tensor:
    """
    Standardize node features to zero mean and unit variance using sklearn's StandardScaler.

    Args:
        data (torch.Tensor): Input graph data with node features

    Returns:
        torch.Tensor: Data with standardized features

    Note:
        Maintains tensor format while using sklearn's scaler for numerical stability
    """
    scaler = StandardScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float)
    return data


def augment_data(data: torch.Tensor) -> torch.Tensor:
    """
    Apply full data augmentation pipeline to graph data.

    Composition of:
    1. Feature noise injection
    2. Stochastic edge dropout
    3. Feature standardization

    Args:
        data (torch.Tensor): Input graph data

    Returns:
        torch.Tensor: Augmented graph data

    Note:
        Operations applied in fixed order: Noise -> Edge Drop -> Normalization
    """
    data = add_noise_to_features(data)  # Add Gaussian feature noise
    data = random_drop_edges(data, p=0.2)  # Remove random edges
    data = normalize_features(data)  # Standardize features
    return data


def balance_and_augment(data_list: list) -> list:
    """
    Balance class distribution through minority class oversampling with data augmentation.

    Args:
        data_list (list): List of PyG Data objects with imbalanced class distribution

    Returns:
        list: Balanced list of Data objects with equal class representation

    Workflow:
        1. Analyze original class distribution
        2. Determine majority class count
        3. For each minority class:
            a. Add original samples
            b. Generate augmented samples until reaching majority count
        4. Shuffle final dataset

    Note:
        Uses clone() for safe tensor operations during augmentation
    """
    labels = [int(data.y[0].item()) for data in data_list]
    label_counter = Counter(labels)
    print("Original class distribution:", label_counter)

    max_count = max(label_counter.values())  # Target samples per class
    balanced_data = []

    for label, count in label_counter.items():
        class_data = [data for data in data_list if int(data.y[0].item()) == label]
        balanced_data.extend(class_data)  # Include original samples

        # Generate synthetic samples for minority classes
        if count < max_count:
            additional_data = []
            for _ in range(max_count - count):
                sample = random.choice(class_data)   # Random sampling with replacement
                augmented_sample = augment_data(sample.clone())  # Safe tensor cloning
                additional_data.append(augmented_sample)
            balanced_data.extend(additional_data)

    # Final dataset shuffling
    random.shuffle(balanced_data)
    balanced_labels = [int(data.y[0].item()) for data in balanced_data]
    print("Balanced class distribution:", Counter(balanced_labels))

    return balanced_data


if __name__ == "__main__":
    # Data preparation and processing pipeline
    data_list = torch.load('data_list.pt') # Load raw imbalanced dataset
    # Execute class balancing with augmentation
    balanced_data = balance_and_augment(data_list)
    # Shuffle and save processed data
    random.shuffle(balanced_data)
    torch.save(balanced_data, 'data_list_enhance.pt')
    # Verify final class distribution
    balanced_labels = [int(data.y[0].item()) for data in balanced_data]
    print("Final balanced distribution:", Counter(balanced_labels))