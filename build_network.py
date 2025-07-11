import re
import os
import networkx as nx
from collections import defaultdict, Counter
import itertools
from pathlib import Path
import argparse # Added for command-line arguments

# --- Constants for WoS Field Codes ---
FN_FIELD = 'FN'; VR_FIELD = 'VR'; PT_FIELD = 'PT'; AU_FIELD = 'AU'; AF_FIELD = 'AF'
TI_FIELD = 'TI'; SO_FIELD = 'SO'; LA_FIELD = 'LA'; DT_FIELD = 'DT'; DE_FIELD = 'DE'
ID_FIELD = 'ID'; AB_FIELD = 'AB'; C1_FIELD = 'C1'; EM_FIELD = 'EM'; OI_FIELD = 'OI'
CR_FIELD = 'CR'; NR_FIELD = 'NR'; TC_FIELD = 'TC'; PY_FIELD = 'PY'; VL_FIELD = 'VL'
IS_FIELD = 'IS'; BP_FIELD = 'BP'; EP_FIELD = 'EP'; PG_FIELD = 'PG'; UT_FIELD = 'UT'
ER_FIELD = 'ER'; EF_FIELD = 'EF'

# --- Function: parse_wos_file ---
def parse_wos_file(filepath):
    """
    Parses a Web of Science plain text file, correctly handling multi-line fields.
    """
    publications = []
    current_pub = {}
    current_field = None
    line_num = 0

    try:
        # First, try to open with utf-8-sig to handle potential BOM
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        print(f"  Successfully read file {filepath.name} with encoding: utf-8-sig")
    except UnicodeDecodeError:
        try:
            # Fallback to latin-1 if utf-8 fails
            with open(filepath, 'r', encoding='latin-1') as f:
                lines = f.readlines()
            print(f"  Successfully read file {filepath.name} with encoding: latin-1")
        except Exception as e:
            print(f"Error reading file {filepath.name}: {e}")
            return []

    for line in lines:
        line_num += 1
        line = line.strip()
        if not line:
            continue

        # Check if the line starts with a known 2-character field code
        if len(line) > 2 and line[2] == ' ':
            field_code = line[:2]
            content = line[3:].strip()

            if field_code in [FN_FIELD, VR_FIELD, EF_FIELD]:
                continue # Skip file header/footer fields

            if field_code == ER_FIELD:
                if current_pub:
                    publications.append(current_pub)
                current_pub = {}
                current_field = None
                continue

            current_field = field_code
            if field_code in [AU_FIELD, AF_FIELD, C1_FIELD, CR_FIELD]:
                 # These fields can have multiple entries
                if field_code not in current_pub:
                    current_pub[field_code] = []
                current_pub[field_code].append(content)
            else:
                current_pub[field_code] = content
        else:
            # This is a continuation of the previous multi-line field
            if current_field and current_field in current_pub:
                # Append to the last item if it's a list, or to the string itself
                if isinstance(current_pub[current_field], list):
                    current_pub[current_field][-1] += ' ' + line
                else:
                    current_pub[current_field] += ' ' + line
            # else:
                # print(f"Warning: Orphaned line at {filepath.name}:{line_num}: '{line}'")

    if current_pub: # Add the last publication if file doesn't end with ER
        publications.append(current_pub)

    return publications


# --- Function: normalize_cited_ref ---
def normalize_cited_ref(ref_string):
    """
    Normalizes a cited reference string to a consistent format.
    Example Input: 'Caragliu, A, 2011, J URBAN TECHNOL, V18, P65, DOI 10.1080/10630732.2011.601117'
    Example Output: 'CARAGLIU A, 2011, J URBAN TECHNOL'
    """
    parts = [p.strip() for p in ref_string.split(',')]
    if len(parts) < 3:
        return None # Not enough info to normalize

    # Author: Keep first author's last name and first initial
    author_part = parts[0].upper()
    year_part = parts[1]
    journal_part = parts[2].upper()

    # Simple author normalization (e.g., "CARAGLIU A" from "Caragliu, A")
    author_name_parts = author_part.split(' ')
    author_norm = author_name_parts[0]
    if len(author_name_parts) > 1:
        initial = author_name_parts[1]
        if initial:
             author_norm += f" {initial[0]}"

    # Year: Ensure it's a 4-digit number
    year_match = re.search(r'\b(19|20)\d{2}\b', year_part)
    if not year_match:
        return None
    year_norm = year_match.group(0)

    # Journal: Basic cleanup
    journal_norm = journal_part.replace('.', '').strip()

    return f"{author_norm}, {year_norm}, {journal_norm}"


# --- Function: build_cocitation_network ---
def build_cocitation_network(publications, min_node_citations_threshold=1, min_cocitation_strength_threshold=1):
    """
    Builds a co-citation network from a list of publication dictionaries.
    """
    print("\n--- Building Co-citation Network ---")
    
    # Counter for total citations of each reference
    citation_counts = Counter()
    # defaultdict to store co-citation pairs and their counts
    cocitation_counts = defaultdict(int)

    print("Processing publications to count citations and co-citations...")
    for pub in publications:
        if CR_FIELD in pub:
            # Normalize all cited references in the current publication
            cited_refs = [normalize_cited_ref(cr) for cr in pub[CR_FIELD]]
            # Filter out any that failed to normalize
            cited_refs = [ref for ref in cited_refs if ref is not None]

            # Update total citation counts for each unique reference
            for ref in set(cited_refs):
                citation_counts[ref] += 1
            
            # Generate all unique pairs of co-cited references and update their counts
            if len(cited_refs) > 1:
                for ref1, ref2 in itertools.combinations(sorted(set(cited_refs)), 2):
                    cocitation_counts[(ref1, ref2)] += 1

    print("Finished processing publications.")
    print(f"Total unique cited references identified (potential nodes): {len(citation_counts)}")
    print(f"Total unique co-citation links found (potential edge types): {len(cocitation_counts)}")
    print(f"Total co-citation instances (sum of edge weights): {sum(cocitation_counts.values())}")

    # Create the NetworkX graph
    G = nx.Graph()

    # Add nodes to the graph with their total citation frequency as an attribute
    print("Adding nodes to graph...")
    for ref, count in citation_counts.items():
        G.add_node(ref, freq=count)
    print(f"  Initial nodes added to graph: {G.number_of_nodes()}")

    # Add edges to the graph with the co-citation count as the 'weight' attribute
    print("Adding edges to graph (applying co-citation strength filter)...")
    print(f"  Initial potential co-citation link types: {len(cocitation_counts)}")
    edges_added = 0
    for (ref1, ref2), weight in cocitation_counts.items():
        if weight >= min_cocitation_strength_threshold:
            G.add_edge(ref1, ref2, weight=weight)
            edges_added += 1
    print(f"  Edges added after strength filter (weight >= {min_cocitation_strength_threshold}): {edges_added}")
    print(f"  Graph state after edge filtering: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # Filter out nodes that don't meet the minimum total citation threshold
    if min_node_citations_threshold > 1:
        print(f"\nApplying node citation count threshold (freq >= {min_node_citations_threshold})...")
        nodes_to_remove = [node for node, data in G.nodes(data=True) if data.get('freq', 0) < min_node_citations_threshold]
        print(f"  Nodes before citation frequency filter: {G.number_of_nodes()}, Nodes after: {G.number_of_nodes() - len(nodes_to_remove)}")
        G.remove_nodes_from(nodes_to_remove)
        print(f"  Edges remaining after node frequency filter: {G.number_of_edges()}")

    print(f"\nFinal graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


# --- Function: save_graph_to_graphml (Corrected) ---
def save_graph_to_graphml(graph, output_dir, filename):
    """Saves the networkx graph to a GraphML file."""
    # This function now uses the 'filename' variable passed to it.
    output_filepath = output_dir / filename
    try:
        nx.write_graphml(graph, str(output_filepath))
        print(f"\nSuccessfully saved network graph to: {output_filepath}")
    except Exception as e:
        print(f"\nError saving GraphML file: {e}")


# --- Function: save_graph_to_gexf (Corrected) ---
def save_graph_to_gexf(graph, output_dir, filename):
    """Saves the networkx graph to a GEXF file for use in Gephi."""
    # This function also now uses the 'filename' variable.
    output_filepath = output_dir / filename
    try:
        nx.write_gexf(graph, str(output_filepath))
        print(f"Successfully saved GEXF graph for Gephi to: {output_filepath}")
    except Exception as e:
        print(f"\nError saving GEXF file: {e}")


# --- Main execution block (Corrected) ---
def main():
    # Setup to read a project folder name from the command line
    parser = argparse.ArgumentParser(description="Build a co-citation network for a specific project.")
    parser.add_argument("project_folder", type=str, help="The name of the project folder inside 'data/wos/' (e.g., 'smart_city' or 'urban_computing')")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    
    # The path now includes the project folder you specify
    wos_data_dir = script_dir / 'data' / 'wos' / args.project_folder
    output_dir = script_dir / 'data' / 'graphml' / args.project_folder

    print(f"--- Starting process for project: {args.project_folder} ---")

    file_pattern_glob = 'savedrecs*.txt'
    all_txt_files_found = list(wos_data_dir.glob(file_pattern_glob))

    if not all_txt_files_found:
        print(f"\nError: No files matching '{file_pattern_glob}' found in directory: {wos_data_dir}")
        print("Please make sure the project folder name is correct and contains the data files.")
        return

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
        print(f"Parsing {wos_file_path.name}...")
        pubs_from_file = parse_wos_file(wos_file_path)
        all_publications.extend(pubs_from_file)
    
    print("\nFinished parsing all files.")

    # Build the network
    # You can adjust these thresholds as needed
    cocitation_graph = build_cocitation_network(all_publications, 
                                                min_node_citations_threshold=2, 
                                                min_cocitation_strength_threshold=2)

    if cocitation_graph and cocitation_graph.number_of_nodes() > 0:
        # Output filenames are now project-specific
        graphml_filename = f"{args.project_folder}_network.graphml"
        gexf_filename = f"{args.project_folder}_network.gexf"

        save_graph_to_graphml(cocitation_graph, output_dir, graphml_filename)
        save_graph_to_gexf(cocitation_graph, output_dir, gexf_filename)

    print(f"\n--- Process for project '{args.project_folder}' complete. ---")


if __name__ == "__main__":
    main()