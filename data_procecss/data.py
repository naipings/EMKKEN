import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv, MessagePassing, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold
import math
import os
import pickle
import re
import ast
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


def clean_and_convert_embedding(embedding_str: str) -> list:
    """
    Clean and convert string-formatted embeddings into a list of floats.

    Args:
        embedding_str (str): String representation of embedding vector

    Returns:
        list: Cleaned and parsed embedding vector

    Raises:
        ValueError: If string parsing fails due to invalid format
    """
    # Normalize whitespace and remove extraneous characters
    cleaned_str = re.sub(r'\s+', ' ', embedding_str)
    cleaned_str = cleaned_str.strip("[]'")

    try:
        # Safely evaluate string to Python list
        embedding_list = ast.literal_eval(cleaned_str)
        return embedding_list
    except Exception as e:
        print(f"Parsing error: {e}")
        return None


def load_graph_data_with_metadata_embeddings_and_labels(
    root_dir: str,
    citation_dir: str,
    folders: list,
    date_dict: dict
) -> list:
    """
    Load graph data with metadata, embeddings and labels from pickle files.

    Args:
        root_dir (str): Directory containing label data
        citation_dir (str): Base directory for citation graph data
        folders (list): List of domain-specific subdirectories
        date_dict (dict): Mapping of paper IDs to publication years

    Returns:
        list: List of PyTorch Geometric Data objects

    Processing Pipeline:
        1. Load label data from CSV
        2. Iterate through domain folders
        3. Load individual citation graphs
        4. Extract and combine node features
        5. Construct edge indices
        6. Add temporal encoding
        7. Create PyG Data objects

    Note:
        Handles missing/invalid features and embeddings gracefully
    """
    data_list = []
    label_counts = Counter()  # Track class distribution

    # Load label data
    label_file = os.path.join(root_dir, f'class_result.csv')
    label_data = pd.read_csv(label_file, index_col=0)
    valid_paper_ids = label_data.index.tolist()

    for folder in folders:
        citation_path = os.path.join(citation_dir, folder)

        for file_name in os.listdir(citation_path):
            if not file_name.endswith('.pkl'):
                continue

            file_path = os.path.join(citation_path, file_name)
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)

            # Initialize feature containers
            node_features = []
            valid_node_ids = []
            node_labels = []
            node_id_to_idx = {}
            is_center = []
            sci_bert_input = []
            center_id = file_name.split('.')[0]

            # Process each node
            for idx, node_id in enumerate(graph.nodes()):
                # (1) Process metadata features
                if 'features' not in graph.nodes[node_id]:
                    continue
                features = graph.nodes[node_id]['features']
                if isinstance(features[0], np.ndarray) or isinstance(features[0], list):
                    features = np.array(features).flatten()
                    features = features[:2]
                else:
                    features = features[:2]
                # Ensure a feature length of 2 (adjusted according to the dataset)
                if len(features) < 2:
                    features = np.pad(features, (0, 2 - len(features)), 'constant')

                # (2) Process embeddings
                if 'embedding' not in graph.nodes[node_id]:
                    continue
                embedding = graph.nodes[node_id]['embedding']
                embedding = embedding[0]
                if isinstance(embedding, str):
                    embedding = clean_and_convert_embedding(embedding)
                    if embedding is None:
                        continue

                # Combine features and validate
                if node_id in valid_paper_ids and node_id == center_id:
                    label = label_data.loc[node_id].values[0]
                    label_counts[label] += 1
                    node_labels.append(label)
                else:
                    node_labels.append(-1)  # Invalid label

                is_center.append(1 if node_id == center_id else 0)

                if isinstance(embedding, (list, tuple)) and all(isinstance(i, (int, float)) for i in embedding):
                    combined_features = np.concatenate((features, embedding))
                    node_features.append(combined_features)
                    sci_bert_input.append(embedding)
                    valid_node_ids.append(node_id)
                    node_id_to_idx[node_id] = idx

                    # Assign labels
                    if node_id in valid_paper_ids and node_id == center_id:
                        label = label_data.loc[node_id].values[0]
                        label_counts[label] += 1
                        node_labels.append(label)
                    else:
                        node_labels.append(-1)  # Invalid label

                    is_center.append(1 if node_id == center_id else 0)
                else:
                    continue

            if node_features:
                node_features = torch.tensor(node_features, dtype=torch.float)
                converted_data = [item[0] if isinstance(item, np.ndarray) else item for item in node_labels]
                node_labels = torch.tensor(converted_data, dtype=torch.long)
                for idx in is_center:
                    if idx == 1:
                        center_index = len(is_center) - is_center[::-1].index(1) - 1
                is_center = torch.tensor(is_center, dtype=torch.long)

                # Build edge indices
                edges = [
                    (node_id_to_idx[u], node_id_to_idx[v])
                    for u, v in graph.edges()
                    if u in node_id_to_idx and v in node_id_to_idx
                ]

                if edges:
                    edges = np.array(edges).T
                    edge_index = torch.tensor(edges, dtype=torch.long)

                    # Add temporal encoding
                    publish_dates = [date_dict.get(node_id, 0) for node_id in valid_node_ids]
                    publish_dates = torch.tensor(publish_dates, dtype=torch.float)
                    relative_time = calculate_relative_time_encoding(publish_dates, center_index)

                    # Create PyG Data object
                    data = Data(x=node_features, edge_index=edge_index, y=node_labels)
                    data.is_center = is_center
                    data.valid_node_ids = valid_node_ids
                    data.center_id = center_id
                    data.sci_bert_input = sci_bert_input
                    data.date_encoding = relative_time
                    data_list.append(data)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    # Add temporal encoding
                    publish_dates = [date_dict.get(node_id, 0) for node_id in valid_node_ids]
                    publish_dates = torch.tensor(publish_dates, dtype=torch.float)
                    relative_time = calculate_relative_time_encoding(publish_dates, center_index)

                    # Create PyG Data object
                    data = Data(x=node_features, edge_index=edge_index, y=node_labels)
                    data.is_center = is_center
                    data.valid_node_ids = valid_node_ids
                    data.center_id = center_id
                    data.sci_bert_input = sci_bert_input
                    data.date_encoding = relative_time
                    data_list.append(data)
            else:
                continue

    print(f"Class distribution: {label_counts}")
    return data_list


def calculate_relative_time_encoding(time_stamps: torch.Tensor, center_index: int) -> torch.Tensor:
    """
    Compute relative temporal encoding based on center node's publication date.

    Args:
        time_stamps (torch.Tensor): Tensor of publication dates
        center_index (int): Index of center node

    Returns:
        torch.Tensor: Normalized relative time differences

    Note:
        Normalizes time differences to [0, 1] range relative to oldest publication
    """
    num_nodes = len(time_stamps)
    relative_time = torch.zeros(num_nodes, dtype=torch.float)

    # Obtain the release date of the central node
    center_time = time_stamps[center_index]
    # Calculate the relative time difference based on the central node
    for i in range(num_nodes):
        relative_time[i] = center_time - time_stamps[i]
    # Ensure that the encoding of the central node is 0
    relative_time[center_index] = 0.0

    # Normalize to [0, 1] range
    max_value = relative_time.max()
    if max_value > 0:
        relative_time = relative_time / max_value  # Normalize the difference to the range of [0,1]

    return relative_time


def build_date_dict(csv_files: list) -> dict:
    """
    Construct mapping from paper IDs to normalized publication years.

    Args:
        csv_files (list): List of CSV file paths containing publication data

    Returns:
        dict: Mapping of paper IDs to years relative to earliest publication

    Note:
        Normalizes years by subtracting minimum year across all files
    """
    date_dict = {}
    all_years = []
    
    # Collect all publication years
    for file in csv_files:
        date_data = pd.read_csv(file)
        all_years.extend(date_data['Pub_year'].tolist())
    
    # Compute minimum year for normalization
    min_year = min(all_years)
    
    # Build normalized year mapping
    for file in csv_files:
        date_data = pd.read_csv(file)
        date_data['Pub_year'] = date_data['Pub_year'].astype(int) - min_year

        # Update the dictionary with paper_id as the key and relative position encoding as the value
        date_dict.update(dict(zip(date_data['Id'], date_data['Pub_year'])))
    
    return date_dict

if __name__ == "__main__":
    # Data loading pipeline
    root_dir = ''  # Root directory for storing node features and labels
    citation_dir = ''  # Root directory for storing graph data
    folders = ['citation_networks_with_embeddings']  # Domain-specific subdirectories

    # Build temporal data
    csv_files = [
        "processed_abstracts_result.csv",
    ]
    date_dict = build_date_dict(csv_files)

    # Load and process graph data
    data_list = load_graph_data_with_metadata_embeddings_and_labels(root_dir, citation_dir, folders, date_dict)
    print(f"Loaded {len(data_list)} graphs into data_list")

    # Save processed data
    torch.save(data_list, 'data_list.pt')