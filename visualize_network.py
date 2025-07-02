import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import re
import operator

# Reuse the robust parsing and network building functions from your project
from build_network import parse_wos_file, build_cocitation_network

# --- Configuration ---
# Set how many of the top-cited papers you want to include in the visualization.
# 50-100 is a good range for a readable static plot.
TOP_NODES_TO_DRAW = 200
# --- End Configuration ---


def get_all_publications(wos_data_dir):
    """Parses all WoS files and returns a single list of all publications."""
    print("--- Parsing all Web of Science files ---")
    
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

    all_publications = []
    for wos_file_path in wos_files_to_process:
        print(f"  Parsing {wos_file_path.name}...")
        pubs_from_file = parse_wos_file(wos_file_path)
        all_publications.extend(pubs_from_file)

    print(f"  Finished parsing. Total publications found: {len(all_publications)}")
    return all_publications


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    wos_data_dir = script_dir / 'data' / 'wos'

    # 1. Load all publications from the raw files
    all_pubs = get_all_publications(wos_data_dir)

    if all_pubs:
        # 2. Build the full co-citation network
        full_graph = build_cocitation_network(all_pubs, 
                                         min_node_citations_threshold=1, 
                                         min_cocitation_strength_threshold=1)

        if full_graph and full_graph.number_of_nodes() > 0:
            print(f"\n--- Creating visualization for top {TOP_NODES_TO_DRAW} nodes ---")
            
            # 3. Identify the top N most-cited nodes
            node_frequencies = nx.get_node_attributes(full_graph, 'freq')
            sorted_nodes = sorted(node_frequencies.items(), key=operator.itemgetter(1), reverse=True)
            top_nodes = [node for node, freq in sorted_nodes[:TOP_NODES_TO_DRAW]]

            # 4. Create a subgraph containing only these top nodes
            subgraph = full_graph.subgraph(top_nodes)
            print(f"  Subgraph created with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")

            # 5. Prepare for drawing with more detail
            plt.figure(figsize=(20, 20)) # Create a large figure to draw on

            # Create a layout for the nodes
            print("  Calculating layout...")
            pos = nx.spring_layout(subgraph, k=0.8, iterations=50, seed=42)

            # Get node sizes based on citation frequency
            node_sizes = [d['freq'] * 5 for n, d in subgraph.nodes(data=True)]

            # Get edge widths based on co-citation weight
            edge_widths = [d['weight'] * 0.2 for u, v, d in subgraph.edges(data=True)]

            print("  Drawing network...")
            # Draw the nodes and edges
            nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
            nx.draw_networkx_edges(subgraph, pos, width=edge_widths, edge_color='gray', alpha=0.5)

            # Add labels to a few of the most central nodes (optional)
            # Drawing all 200 labels will be unreadable. This is just an example.
            sorted_by_degree = sorted(dict(subgraph.degree()).items(), key=lambda item: item[1], reverse=True)
            top_10_nodes = {node: f"{node.split(',')[0]}, {node.split(',')[1]}" for node, degree in sorted_by_degree[:10]}
            nx.draw_networkx_labels(subgraph, pos, labels=top_10_nodes, font_size=12, font_color='black')
            
            plt.title("Co-citation Network of Top 200 Papers", size=20)
            plt.axis('off') # Hide the axes

            # 6. Save and Show the figure
            output_path = script_dir / 'network_visualization.png'
            print(f"  Saving visualization to {output_path}")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            plt.show() # Also display the plot on screen

    print("\nScript finished.")