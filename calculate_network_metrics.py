import pandas as pd
import networkx as nx
from pathlib import Path
import re
import operator
import argparse

# Reuse the robust parsing and network building functions from your project
# This assumes build_network.py is in the same directory
from build_network import parse_wos_file, build_cocitation_network

# --- Configuration ---
# Set how many of the top papers you want to print to the screen
NUM_TO_DISPLAY = 20
# --- End Configuration ---


def get_all_publications(wos_data_dir):
    """Parses all WoS files from a specific project directory and returns a single list of all publications."""
    print(f"--- Parsing all Web of Science files for project: {wos_data_dir.name} ---")
    
    # File discovery and sorting
    file_pattern_glob = 'savedrecs*.txt'
    all_txt_files_found = list(wos_data_dir.glob(file_pattern_glob))
    if not all_txt_files_found:
        print(f"Error: No files matching '{file_pattern_glob}' found in {wos_data_dir}")
        return []
        
    file_name_pattern_re = re.compile(r"^(savedrecs)(?: \((\d+)\))?\.txt$")
    files_with_num = []
    for f_path in all_txt_files_found:
        match = file_name_pattern_re.match(f_path.name)
        if match:
            num_str = match.group(2)
            num = int(num_str) if num_str else 0
            files_with_num.append((num, f_path))
    files_with_num.sort(key=lambda x: x[0])
    wos_files_to_process = [f_path for num, f_path in files_with_num]

    # Parse all files into one list
    all_publications = []
    for wos_file_path in wos_files_to_process:
        print(f"  Parsing {wos_file_path.name}...")
        pubs_from_file = parse_wos_file(wos_file_path)
        all_publications.extend(pubs_from_file)

    print(f"  Finished parsing. Total publications found: {len(all_publications)}")
    return all_publications


def calculate_and_compile_metrics(graph):
    """
    Calculates key network metrics and returns them as a Pandas DataFrame.
    """
    print("\n--- Calculating Network Metrics ---")
    if not graph or graph.number_of_nodes() == 0:
        print("  Graph is empty. Cannot calculate metrics.")
        return pd.DataFrame()

    # 1. Get Total Citation counts (already stored as 'freq' attribute)
    citation_counts = nx.get_node_attributes(graph, 'freq')
    print("  Calculated Citation Counts.")

    # 2. Calculate Degree Centrality
    degree_centrality = nx.degree_centrality(graph)
    print("  Calculated Degree Centrality.")

    # Note: Betweenness Centrality is commented out as requested due to long processing time.
    # To enable it, uncomment the following lines and the line in the metrics_data dictionary.
    # print("  Calculating Betweenness Centrality (this may take a few moments)...")
    # betweenness_centrality = nx.betweenness_centrality(graph, normalized=True, endpoints=False)
    # print("  Calculated Betweenness Centrality.")

    # 3. Combine all metrics into a single data structure
    metrics_data = []
    for node in graph.nodes():
        metrics_data.append({
            'Paper': node,
            'Citation_Count': citation_counts.get(node, 0),
            'Degree_Centrality': degree_centrality.get(node, 0.0)
            # 'Betweenness_Centrality': betweenness_centrality.get(node, 0.0)
        })
        
    # 4. Convert to a Pandas DataFrame for easy handling
    df = pd.DataFrame(metrics_data)
    
    # Sort the dataframe by the most important metric, e.g., Citation Count
    df_sorted = df.sort_values(by='Citation_Count', ascending=False)
    
    return df_sorted


if __name__ == "__main__":
    # Setup to read a project folder name from the command line
    parser = argparse.ArgumentParser(description="Calculate network metrics for a specific project's co-citation graph.")
    parser.add_argument("project_folder", type=str, help="The name of the project folder inside 'data/wos/' (e.g., 'smart_city' or 'urban_computing')")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    # Construct paths based on the project folder argument
    wos_data_dir = script_dir / 'data' / 'wos' / args.project_folder
    output_dir = script_dir / 'data'
    
    # Make the output filename specific to the project
    output_filename = f"{args.project_folder}_network_metrics.csv"

    print(f"\n--- Starting metric calculation for project: {args.project_folder} ---")

    # 1. Load all publications from the specified project folder
    all_pubs = get_all_publications(wos_data_dir)

    if all_pubs:
        # 2. Build the co-citation network
        # We use low thresholds to build a comprehensive graph for analysis
        graph = build_cocitation_network(all_pubs, 
                                         min_node_citations_threshold=1, 
                                         min_cocitation_strength_threshold=1)

        # 3. Calculate all metrics and get a sorted DataFrame
        metrics_df = calculate_and_compile_metrics(graph)

        if not metrics_df.empty:
            # 4. Display the top N papers in the console
            print(f"\n--- Top {NUM_TO_DISPLAY} Most-Cited Papers with Network Metrics ---")
            # Format floating point numbers for better readability in the console
            pd.options.display.float_format = '{:.6f}'.format
            print(metrics_df.head(NUM_TO_DISPLAY).to_string())

            # 5. Save the full, sorted DataFrame to a project-specific CSV file
            output_filepath = output_dir / output_filename
            try:
                metrics_df.to_csv(output_filepath, index=False)
                print(f"\n--- Success! ---")
                print(f"Full table of network metrics for all {len(metrics_df)} papers saved to:")
                print(f"{output_filepath}")
            except Exception as e:
                print(f"\nError saving CSV file: {e}")

    print("\nScript finished.")
