import pandas as pd
import networkx as nx
import numpy as np
import ast
import os
import pickle
import matplotlib.pyplot as plt

# Data Loading Configuration
CITATION_FILE = 'references.csv'          # Citation relationships data
EMBED_FILE = 'embedding/papers_data_embed.csv'  # Paper embeddings data
METADATA_FILE = 'meta_data.csv'          # Paper metadata
OUTPUT_DIR = 'citation_networks_with_embeddings'  # Output directory for enhanced graphs


def load_and_preprocess_data() -> tuple:
    """
    Load and preprocess required datasets for graph enhancement.

    Returns:
        tuple: Contains three elements:
            - citation_data (pd.DataFrame): Citation relationships
            - paper_embeddings (pd.DataFrame): Paper embedding vectors
            - metadata (pd.DataFrame): Paper metadata

    File Formats Expected:
        - Citation data: paper_id, references_id columns
        - Embeddings: paper_id followed by embedding dimensions
        - Metadata: Paper attributes including ID as 'Id'
    """
    # Load citation relationships
    citation_data = pd.read_csv(CITATION_FILE)

    # Load paper embeddings (numeric feature vectors)
    paper_embeddings = pd.read_csv(EMBED_FILE)

    # Load paper metadata (categorical/textual features)
    metadata = pd.read_csv(METADATA_FILE)

    return citation_data, paper_embeddings, metadata


def create_lookup_dictionaries(embeddings: pd.DataFrame,
                             metadata: pd.DataFrame) -> tuple:
    """
    Create fast lookup structures for paper features.

    Args:
        embeddings (pd.DataFrame): Paper embedding vectors
        metadata (pd.DataFrame): Paper metadata attributes

    Returns:
        tuple: Two dictionaries:
            - paper_embedding_dict: {paper_id: embedding_vector}
            - metadata_dict: {paper_id: metadata_attributes}

    Note:
        Metadata dataframe uses 'Id' column for paper identification
    """
    paper_embedding_dict = dict(zip(embeddings['paper_id'],
                                  embeddings.iloc[:, 1:].values))
    metadata_dict = dict(zip(metadata['Id'],
                           metadata.iloc[:, 1:].values))
    return paper_embedding_dict, metadata_dict


def enhance_citation_graphs(citation_data: pd.DataFrame,
                            embeddings: dict,
                            metadata: dict,
                            output_dir: str) -> list:
    """
    Enhance citation networks with embeddings and metadata.

    Args:
        citation_data (pd.DataFrame): Citation relationships
        embeddings (dict): Paper embedding lookup
        metadata (dict): Paper metadata lookup
        output_dir (str): Directory to save enhanced graphs

    Returns:
        list: Valid paper IDs that were successfully processed

    Processing Steps:
        1. Load existing citation graphs
        2. Add embedding and metadata features to nodes
        3. Clean graphs by removing incomplete nodes
        4. Save enhanced graphs to disk
    """
    valid_paper_ids = []
    processed_count = error_count = 0

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    for paper_id in embeddings['paper_id']:
        try:
            # Load citation graph
            graph_path = f'./citation_networks/{paper_id}.pkl'
            if not os.path.exists(graph_path):
                continue

            with open(graph_path, 'rb') as f:
                citation_graph = pickle.load(f)

            # Add features to center node
            if paper_id in citation_graph.nodes:
                citation_graph.nodes[paper_id]['embedding'] = embeddings[paper_id]
                citation_graph.nodes[paper_id]['features'] = metadata.get(paper_id, None)
                valid_paper_ids.append(paper_id)

            # Process cited papers
            cited_papers = citation_data[citation_data['paper_id'] == paper_id]['references_id']
            cited_papers = parse_reference_list(cited_papers)

            # Add features to cited nodes
            for cited_id in cited_papers:
                if cited_id in citation_graph.nodes:
                    if cited_id in embeddings:
                        citation_graph.nodes[cited_id]['embedding'] = embeddings[cited_id]
                    if cited_id in metadata:
                        citation_graph.nodes[cited_id]['features'] = metadata[cited_id]

            # Clean graph structure
            citation_graph = clean_graph(citation_graph)

            # Save enhanced graph
            output_path = f'./{output_dir}/{paper_id}.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(citation_graph, f)

            processed_count += 1

        except Exception as e:
            print(f"Error processing {paper_id}: {str(e)}")
            error_count += 1

    print(f"Processed {processed_count} graphs, {error_count} errors")
    return valid_paper_ids


def parse_reference_list(reference_series: pd.Series) -> list:
    """
    Parse references from various data formats to clean list.

    Args:
        reference_series (pd.Series): Raw reference data

    Returns:
        list: Cleaned list of paper IDs

    Handles:
        - String representations of lists
        - Missing values
        - Inconsistent data types
    """
    if reference_series.empty:
        return []

    raw_refs = reference_series.iloc[0]

    if isinstance(raw_refs, str):
        try:
            return ast.literal_eval(raw_refs)
        except (ValueError, SyntaxError):
            return []
    elif isinstance(raw_refs, list):
        return raw_refs
    else:
        return []


def clean_graph(graph: nx.Graph) -> nx.Graph:
    """
    Remove incomplete nodes and clean graph structure.

    Args:
        graph (nx.Graph): Input graph with potential missing features

    Returns:
        nx.Graph: Cleaned graph with only complete nodes

    Cleaning Steps:
        1. Remove nodes missing embeddings
        2. Remove nodes missing metadata
        3. Remove residual self-loops
    """
    # Remove nodes without embeddings
    emb_nodes = {n for n, d in graph.nodes(data=True) if 'embedding' in d}
    graph.remove_nodes_from(set(graph.nodes) - emb_nodes)

    # Remove nodes without metadata
    meta_nodes = {n for n, d in graph.nodes(data=True) if 'features' in d}
    graph.remove_nodes_from(set(graph.nodes) - meta_nodes)

    # Clean residual edges
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph


def filter_csv_by_paper_ids(input_path: str,
                            output_path: str,
                            valid_ids: list) -> None:
    """
    Filter CSV file to only include specified paper IDs.

    Args:
        input_path (str): Path to original CSV file
        output_path (str): Path for filtered output
        valid_ids (list): List of paper IDs to retain

    Note:
        First column of CSV is assumed to contain paper IDs
    """
    df = pd.read_csv(input_path)
    original_size = len(df)

    # Filter rows using first column as ID
    filtered_df = df[df.iloc[:, 0].isin(valid_ids)]
    # filtered_df = df[~df.iloc[:, 0].isin(valid_ids)]

    filtered_df.to_csv(output_path, index=False)
    print(f"Filtered {original_size} -> {len(filtered_df)} rows")


# Main execution flow
if __name__ == "__main__":
    # Load and prepare data
    citation_df, embeddings_df, metadata_df = load_and_preprocess_data()
    embed_dict, meta_dict = create_lookup_dictionaries(embeddings_df, metadata_df)

    # Process citation graphs
    valid_ids = enhance_citation_graphs(citation_df, embed_dict, meta_dict, OUTPUT_DIR)

    # Filter metadata CSV
    filter_csv_by_paper_ids('processed_abstracts.csv',
                            'processed_abstracts_result.csv',
                            valid_ids)