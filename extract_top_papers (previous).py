# extract_top_papers.py

import re
from pathlib import Path
from collections import defaultdict
import operator
import argparse

from build_network import parse_wos_file, build_cocitation_network

def create_publication_database(wos_data_dir):
    """
    Parses all WoS files from a specific project directory and builds a UT-keyed database,
    a searchable index, and a list of all publications for network building.
    """
    print(f"--- Building Publication Database for Lookup in: {wos_data_dir.name} ---")
    
    # File discovery and sorting
    file_pattern_glob = 'savedrecs*.txt'
    all_txt_files_found = list(wos_data_dir.glob(file_pattern_glob))
    if not all_txt_files_found:
        print(f"Error: No files matching '{file_pattern_glob}' found in {wos_data_dir}")
        return None, None, None
        
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

    # Build the databases
    pub_database_by_ut = {}
    search_index = defaultdict(list)
    all_publications_for_network = []
    
    total_pubs = 0
    for wos_file_path in wos_files_to_process:
        # print(f"  Parsing {wos_file_path.name}...") # Uncomment for more verbose output
        pubs_from_file = parse_wos_file(wos_file_path)
        all_publications_for_network.extend(pubs_from_file)

        for pub in pubs_from_file:
            ut = pub.get('UT')
            if not ut: continue
            
            pub_database_by_ut[ut] = pub
            
            authors = pub.get('AU', [])
            year = pub.get('PY')
            if authors and year:
                first_author_surname = authors[0].split(',')[0].strip().upper()
                lookup_key = (first_author_surname, year)
                search_index[lookup_key].append(ut)
        total_pubs += len(pubs_from_file)

    print(f"  Database built. Parsed {total_pubs} publications.")
    return pub_database_by_ut, search_index, all_publications_for_network

def find_top_cited_papers(graph, num_to_find):
    """
    Finds the top N most frequently cited papers from the network graph.
    """
    print(f"\n--- Identifying Top {num_to_find} Most-Cited Papers from Network ---")
    if not graph or graph.number_of_nodes() == 0:
        print("  Graph is empty. Cannot identify top papers.")
        return []

    node_frequencies = {node: data['freq'] for node, data in graph.nodes(data=True)}
    sorted_nodes = sorted(node_frequencies.items(), key=operator.itemgetter(1), reverse=True)
    
    top_nodes = sorted_nodes[:num_to_find]
    print(f"  Identified top {len(top_nodes)} papers based on citation frequency.")

    return [node for node, freq in top_nodes]


def search_for_records(target_refs, search_index, pub_database_by_ut):
    """
    Takes a list of normalized reference strings and finds their full records.
    """
    print("\n--- Matching Top Papers to Full Records in Database ---")
    found_records = []
    not_found_count = 0
    for target in target_refs:
        parts = target.split(',')
        if len(parts) < 2: continue

        target_author_surname = parts[0].strip().upper().split(' ')[0]
        target_year = parts[1].strip()

        search_key = (target_author_surname, target_year)
        found_uts = search_index.get(search_key)

        if found_uts:
            for ut in found_uts:
                record = pub_database_by_ut.get(ut)
                if record and record not in found_records:
                    found_records.append(record)
        else:
            not_found_count += 1
    
    print(f"  Successfully found full records for {len(found_records)} out of {len(target_refs)} top papers.")
    if not_found_count > 0:
        print(f"  Note: {not_found_count} top cited papers were not found in the source dataset (likely foundational papers from other fields).")
    return found_records

def write_records_to_file(records, output_dir, filename):
    """
    Writes a list of publication records to a new file in WoS format.
    """
    print(f"\n--- Writing Found Records to '{filename}' ---")
    if not records:
        print("  No records to write.")
        return

    # Ensure the output directory exists before writing
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filepath = output_dir / filename
    
    field_order = [
        'FN', 'VR', 'PT', 'AU', 'AF', 'TI', 'SO', 'LA', 'DT', 'DE', 'ID', 'AB',
        'C1', 'EM', 'OI', 'CR', 'NR', 'TC', 'PY', 'VL', 'IS', 'BP', 'EP', 'PG', 'UT'
    ]

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write("FN Clarivate Analytics Web of Science\n")
            f.write("VR 1.0\n")
            for record in records:
                for field in field_order:
                    if field in record:
                        value = record[field]
                        if isinstance(value, list):
                            for item in value:
                                f.write(f"{field} {item}\n")
                        else:
                            f.write(f"{field} {value}\n")
                f.write("ER\n")
            f.write("EF\n")
        print(f"  Successfully wrote {len(records)} records to {output_filepath}")
    except Exception as e:
        print(f"  An error occurred while writing the file: {e}")


if __name__ == "__main__":
    # Setup to read arguments from the command line
    parser = argparse.ArgumentParser(description="Extract full records for the most-cited papers in a project's co-citation network.")
    parser.add_argument("project_folder", type=str, help="The name of the project folder inside 'data/wos/' (e.g., 'smart_city' or 'urban_computing')")
    parser.add_argument("--top_n", type=int, default=500, help="The number of top-cited papers to extract (default: 500).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    
    # Construct paths based on the project folder argument
    wos_data_dir = script_dir / 'data' / 'wos' / args.project_folder
    
    # Create a dedicated, project-specific output directory
    output_dir = script_dir / 'data' / 'export' / args.project_folder
    
    # Make the output filename specific to the project and the number of papers
    output_filename = f"top_{args.top_n}_cited_papers.txt"

    print(f"\n--- Starting extraction for project: {args.project_folder} ---")

    # 1. Build the database of all publications
    db, s_index, all_pubs = create_publication_database(wos_data_dir)

    if all_pubs:
        # 2. Build the co-citation network to get citation counts
        # We use low thresholds here as we just need the node frequencies
        graph = build_cocitation_network(all_pubs, 
                                         min_node_citations_threshold=1, 
                                         min_cocitation_strength_threshold=1)

        # 3. Identify the top N most-cited papers from the network
        top_paper_nodes = find_top_cited_papers(graph, args.top_n)

        # 4. Find the full records for these top papers
        full_records_to_save = search_for_records(top_paper_nodes, s_index, db)

        # 5. Write these records to a new, single txt file
        write_records_to_file(full_records_to_save, output_dir, output_filename)

    print("\nScript finished.")
