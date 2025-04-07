import gzip
import pickle
import os
import networkx as nx
import pandas as pd
import ast
import matplotlib.pyplot as plt


def build_citation_mapping(df: pd.DataFrame) -> dict:
    """
    Construct mapping from papers to their references.

    Args:
        df (pd.DataFrame): DataFrame containing paper IDs and references

    Returns:
        dict: Mapping of paper IDs to lists of references

    Processing:
        1. Iterates through each row in the DataFrame
        2. Parses reference lists using ast.literal_eval
        3. Skips invalid or empty reference entries
    """
    mapping = {}
    for _, row in df.iterrows():
        paper_id = row['paper_id']
        references = row['references_id']

        if pd.isna(references) or references == '[]':
            continue

        try:
            references_list = ast.literal_eval(references)
            mapping[paper_id] = references_list
        except ValueError:
            continue

    return mapping


def analyze_subgraph(subgraph: nx.DiGraph) -> dict:
    """
    Compute network properties for a citation subgraph.

    Args:
        subgraph (nx.DiGraph): Directed citation subgraph

    Returns:
        dict: Dictionary of computed network properties

    Metrics Calculated:
        - Average degree
        - Degree centralities
        - Diameter
        - Density
        - Clustering coefficient
    """
    if subgraph.number_of_nodes() == 0 or subgraph.number_of_edges() == 0:
        return {
            'avg_degree': 0,
            'max_in_degree': 0,
            'min_in_degree': 0,
            'max_out_degree': 0,
            'min_out_degree': 0,
            'diameter': float('inf'),
            'density': 0,
            'clustering_coefficient': 0,
            'in_degree_centrality_avg': 0,
            'out_degree_centrality_avg': 0,
        }

    # Calculate basic properties
    degrees = dict(subgraph.degree())
    in_degrees = dict(subgraph.in_degree())
    out_degrees = dict(subgraph.out_degree())

    # Calculate centralities
    in_degree_centrality = nx.in_degree_centrality(subgraph)
    out_degree_centrality = nx.out_degree_centrality(subgraph)

    return {
        'avg_degree': sum(degrees.values()) / subgraph.number_of_nodes(),
        'max_in_degree': max(in_degrees.values(), default=0),
        'min_in_degree': min(in_degrees.values(), default=0),
        'max_out_degree': max(out_degrees.values(), default=0),
        'min_out_degree': min(out_degrees.values(), default=0),
        'diameter': nx.diameter(subgraph) if nx.is_strongly_connected(subgraph) else float('inf'),
        'density': nx.density(subgraph),
        'clustering_coefficient': nx.average_clustering(subgraph.to_undirected()),
        'in_degree_centrality_avg': sum(in_degree_centrality.values()) / len(in_degree_centrality),
        'out_degree_centrality_avg': sum(out_degree_centrality.values()) / len(out_degree_centrality),
    }


if __name__ == "__main__":
    # Create output directory for citation networks
    output_dir = 'citation_networks'
    os.makedirs(output_dir, exist_ok=True)

    # Load citation data from CSV
    df = pd.read_csv('input.csv', dtype=str)

    # Initialize mapping of papers to their references
    from_to_map = {}
    # Build initial citation mapping
    from_to_map = build_citation_mapping(df)

    # Initialize counters and storage
    subgraph_count = 0
    total_edge_count = 0
    properties = []
    merged_graph = nx.DiGraph()

    # Process each paper's citation network
    for paper_id, references_list in from_to_map.items():
        if not references_list:
            continue

        # Initialize subgraph
        subgraph = nx.DiGraph()

        # Add primary citation edges
        for ref in references_list:
            subgraph.add_edge(paper_id, ref)
            merged_graph.add_edge(paper_id, ref)
            total_edge_count += 1

        # Add secondary citation edges
        for ref in references_list:
            if ref in from_to_map:
                for secondary_ref in from_to_map[ref]:
                    if secondary_ref in references_list:
                        subgraph.add_edge(ref, secondary_ref)
                        merged_graph.add_edge(ref, secondary_ref)
                        total_edge_count  += 1

        # Analyze and store subgraph properties
        subgraph_properties = analyze_subgraph(subgraph)
        subgraph_properties['paper_id'] = paper_id
        properties.append(subgraph_properties)

        # Save subgraph to file
        file_path = os.path.join(output_dir, f'{paper_id}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(subgraph, f)

        subgraph_count += 1

    # Convert properties to DataFrame and save
    properties_df = pd.DataFrame(properties)
    properties_df.describe().to_csv('citation_network_properties.csv', index=False)

    # Analyze merged graph properties
    merged_properties = analyze_subgraph(merged_graph)
    merged_properties = {f'merged_{k}': v for k, v in merged_properties.items()}

    # Save merged graph properties
    merged_properties_df = pd.DataFrame([merged_properties])
    merged_properties_df.to_csv('merged_citation_network_properties.csv', index=False)

    # Print summary statistics
    print(f"Processed {subgraph_count} subgraphs with {total_edge_count} total edges")
    print("Subgraph Properties Summary:")
    print(properties_df.describe())
    print("\nMerged Graph Properties:")
    for key, value in merged_properties.items():
        print(f"{key}: {value}")