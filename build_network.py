# build_network.py

import re
import os # Added for directory operations
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import itertools
import math
from pathlib import Path # Modern way to handle paths

# --- Constants for WoS Field Codes ---
# (Keep the constants definitions as provided before)
FN_FIELD = 'FN'; VR_FIELD = 'VR'; PT_FIELD = 'PT'; AU_FIELD = 'AU'; AF_FIELD = 'AF'
TI_FIELD = 'TI'; SO_FIELD = 'SO'; LA_FIELD = 'LA'; DT_FIELD = 'DT'; DE_FIELD = 'DE'
ID_FIELD = 'ID'; AB_FIELD = 'AB'; C1_FIELD = 'C1'; EM_FIELD = 'EM'; OI_FIELD = 'OI'
CR_FIELD = 'CR'; NR_FIELD = 'NR'; TC_FIELD = 'TC'; PY_FIELD = 'PY'; VL_FIELD = 'VL'
IS_FIELD = 'IS'; BP_FIELD = 'BP'; EP_FIELD = 'EP'; PG_FIELD = 'PG'; UT_FIELD = 'UT'
ER_FIELD = 'ER'; EF_FIELD = 'EF'

# --- Function: parse_wos_file ---
# (Keep the parse_wos_file function exactly as provided before)
def parse_wos_file(filepath):
    """ Parses a Web of Science plain text file. """
    publications = []
    current_pub = {}
    current_field = None
    try:
        encodings_to_try = ['utf-8', 'latin-1']
        file_content = None
        for enc in encodings_to_try:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    file_content = f.readlines()
                print(f"  Successfully read file {Path(filepath).name} with encoding: {enc}")
                break
            except UnicodeDecodeError:
                continue
        if file_content is None:
            print(f"Warning: Could not decode file {filepath} with attempted encodings. Skipping.")
            return []

        for line in file_content:
            line = line.strip()
            if not line: continue
            field_code_match = re.match(r'^([A-Z0-9]{2})\s(.*)$', line)
            if field_code_match:
                current_field = field_code_match.group(1)
                value = field_code_match.group(2).strip()
                if current_field in [AU_FIELD, AF_FIELD, C1_FIELD, CR_FIELD, DE_FIELD, ID_FIELD]:
                    if current_field not in current_pub: current_pub[current_field] = []
                    current_pub[current_field].append(value)
                else: current_pub[current_field] = value
            elif current_field and current_field in current_pub:
                 if isinstance(current_pub[current_field], list):
                     if current_pub[current_field]: current_pub[current_field][-1] += " " + line
                 elif isinstance(current_pub[current_field], str): current_pub[current_field] += " " + line
            if line.startswith(ER_FIELD):
                if current_pub:
                    if UT_FIELD not in current_pub: current_pub[UT_FIELD] = f"MISSING_UT_{len(publications)}"
                    publications.append(current_pub)
                current_pub = {}; current_field = None
        if current_pub and UT_FIELD in current_pub: publications.append(current_pub)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred parsing {filepath}: {e}")
        return []
    return publications

# --- Function: normalize_cited_ref ---
# (Keep the normalize_cited_ref function exactly as provided before)
def normalize_cited_ref(ref_string):
    """ Attempts to parse and normalize a cited reference string. """
    ref_string = ref_string.strip()
    parts = [p.strip() for p in ref_string.split(',')]
    if not parts: return None
    author = parts[0].upper(); author = "ANONYMOUS" if author.startswith('[ANONYMOUS]') or not author else author
    year, source = None, None
    if len(parts) > 1:
        year_match = re.search(r'\b(1[89]\d{2}|20\d{2})\b', parts[1])
        if year_match:
            year = year_match.group(1)
            potential_source = ", ".join(parts[2:]).strip()
            if len(parts) > 2 and not re.match(r'^(V|P|DOI)', parts[2].strip(), re.I): source = parts[2].strip().upper()
            elif potential_source:
                 source_match = re.match(r'^(.*?)(?:,?\s*(?:V\d+|P\d+|DOI\s+|HTTP|WWW))', potential_source, re.I)
                 source = source_match.group(1).strip().upper() if source_match else (parts[2].strip().upper() if len(parts) > 2 else "UNKNOWN_SOURCE")
            else: source = "UNKNOWN_SOURCE"
        else: year, source = "UNKNOWN_YEAR", "UNKNOWN_SOURCE"
    else: year, source = "UNKNOWN_YEAR", "UNKNOWN_SOURCE"
    if source: source = re.sub(r'\s+', ' ', source)
    if author and year and source:
        max_source_len = 50
        return f"{author}, {year}, {source[:max_source_len]}"
    else: return f"{author}, {year or '????'}, {source or '???'}"


# --- Function: build_cocitation_network ---
# (Keep the build_cocitation_network function exactly as provided before)
def build_cocitation_network(publications):
    """ Builds a co-citation network from parsed WoS publications. """
    G = nx.Graph()
    cited_ref_counts = Counter()
    cocitation_links = defaultdict(int)
    node_info = {}
    print(f"Building network from {len(publications)} citing publications...")
    processed_pubs = 0
    for pub in publications:
        citing_pub_id = pub.get(UT_FIELD, 'UnknownUT')
        citing_pub_year = pub.get(PY_FIELD, None)
        cited_refs_raw = pub.get(CR_FIELD, [])
        normalized_refs = []
        for ref_str in cited_refs_raw:
            norm_ref = normalize_cited_ref(ref_str)
            if norm_ref:
                normalized_refs.append(norm_ref)
                cited_ref_counts[norm_ref] += 1
                if norm_ref not in node_info:
                     parts = norm_ref.split(', '); ref_year = parts[1] if len(parts) > 1 and parts[1].isdigit() else None
                     node_info[norm_ref] = {'year': ref_year, 'label': norm_ref}
        for ref1, ref2 in itertools.combinations(normalized_refs, 2):
            edge = tuple(sorted((ref1, ref2))); cocitation_links[edge] += 1
        processed_pubs += 1
        if processed_pubs % 500 == 0: print(f"  Processed {processed_pubs}/{len(publications)} publications...") # Print less often for large datasets
    print(f"Finished processing publications. Found {len(cited_ref_counts)} unique cited references.")
    print(f"Found {len(cocitation_links)} co-citation links.")
    for ref, count in cited_ref_counts.items():
        info = node_info.get(ref, {})
        G.add_node(ref, label=info.get('label', ref), freq=count, year=info.get('year', None))
    for (ref1, ref2), weight in cocitation_links.items():
        if G.has_node(ref1) and G.has_node(ref2): G.add_edge(ref1, ref2, weight=float(weight))
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# --- Function: save_graph_to_graphml ---
# Added function to handle saving with correct attributes and types
def save_graph_to_graphml(graph, filepath):
    """Saves the networkx graph to a GraphML file using lxml."""
    if graph.number_of_nodes() == 0:
        print("Graph is empty, skipping save.")
        return

    print(f"Preparing graph for saving to {filepath}...")
    # --- Ensure data types are correct before saving ---
    for node, data in graph.nodes(data=True):
        # Convert attributes to appropriate types
        data['freq'] = int(data.get('freq', 0))
        data['year'] = str(data.get('year', '')) # Year of the cited reference
        label = data.get('label', node)
        data['name'] = str(label) # Ensure name is string
        parts = label.split(', ')
        data['author'] = str(parts[0]) if len(parts) > 0 else str(label)
        data['so'] = str(parts[2]) if len(parts) > 2 else 'SO' # Ensure string
        # Add placeholders (ensure they are strings as defined in GraphML standard types)
        data['title'] = '...'
        data['vol'] = '0'
        data['page'] = '0'
        data['ut'] = ''

    for u, v, data in graph.edges(data=True):
        data['weight'] = float(data.get('weight', 0.0))
        # Add placeholders (ensure they are strings or appropriate numeric type)
        data['slice'] = int(0) # Use integer type
        data['year'] = '' # Use string type

    # --- Remove the problematic arguments ---
    # These arguments ('node_attr_types', 'edge_attr_types') are not expected by write_graphml_lxml
    # keys_node = { ... } # No longer needed here
    # keys_edge = { ... } # No longer needed here

    try:
        # Ensure the output directory exists
        output_dir = Path(filepath).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write GraphML using lxml (without type arguments)
        # named_key_ids=True ensures readable keys in the graphml file
        nx.write_graphml_lxml(graph, str(filepath), named_key_ids=True, infer_numeric_types=True) # Added infer_numeric_types
        print(f"Graph successfully saved to {filepath}")
    except ImportError:
        print("Error: Saving to GraphML requires the 'lxml' library. Install it using: pip install lxml")
        # Fallback to basic XML writer (less preferred)
        try:
            # Basic writer also doesn't use the _attr_types arguments
            nx.write_graphml_xml(graph, str(filepath), named_key_ids=True)
            print(f"Graph saved using basic XML writer (lxml preferred).")
        except Exception as e_xml:
            print(f"Error saving graph with basic XML writer: {e_xml}")
    except Exception as e:
        print(f"An error occurred saving the graph: {e}")


# --- Main Execution Logic ---
if __name__ == "__main__":
    # Define relative paths based on the script's location
    script_dir = Path(__file__).parent
    wos_data_dir = script_dir / 'data' / 'wos'
    graphml_output_dir = script_dir / 'data' / 'graphml'
    output_filename = 'python_generated_cocitation_network.graphml'
    output_filepath = graphml_output_dir / output_filename

    print(f"Looking for WoS files matching 'savedrecs*.txt' in: {wos_data_dir}")

    all_publications = []
    # Check if the directory exists
    if not wos_data_dir.is_dir():
        print(f"Error: Input directory not found: {wos_data_dir}")
    else:
        # Find all .txt files first
        all_txt_files = list(wos_data_dir.glob('*.txt'))

        # --- New code to filter and sort files ---
        target_files = []
        file_pattern = re.compile(r"savedrecs(?: \((\d+)\))?\.txt") # Regex to match pattern and capture number

        files_with_num = []
        for f_path in all_txt_files:
            match = file_pattern.match(f_path.name)
            if match:
                # Extract number in parenthesis, default to 0 if no parenthesis
                num_str = match.group(1)
                num = int(num_str) if num_str else 0
                files_with_num.append((num, f_path)) # Store as tuple (number, path)

        # Sort files based on the extracted number
        files_with_num.sort(key=lambda x: x[0])

        # Get the sorted list of file paths
        wos_files_to_process = [f_path for num, f_path in files_with_num]
        # --- End of new code ---

        print(f"Found {len(wos_files_to_process)} files matching the pattern to process (sorted):")
        # Optional: print the sorted list for verification
        # for f in wos_files_to_process:
        #     print(f"  - {f.name}")


        if not wos_files_to_process:
            print("No files matching the 'savedrecs*.txt' pattern found. Please check the directory and filenames.")
        else:
            # Loop through the sorted list of files
            for wos_file in wos_files_to_process:
                print(f"Parsing {wos_file.name}...")
                pubs = parse_wos_file(wos_file)
                all_publications.extend(pubs)
                print(f"  Added {len(pubs)} publications from {wos_file.name}. Total: {len(all_publications)}")

            if not all_publications:
                print("No publications were parsed successfully.")
            else:
                # Build the combined network
                G_combined = build_cocitation_network(all_publications)

                # Save the resulting graph
                save_graph_to_graphml(G_combined, output_filepath)
