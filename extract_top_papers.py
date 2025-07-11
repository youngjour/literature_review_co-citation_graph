# extract_top_papers.py

import re
from pathlib import Path
import argparse

# Reuse the robust parsing function from your project
# This assumes build_network.py is in the same directory
from build_network import parse_wos_file

def get_all_publications(wos_data_dir):
    """
    Parses all WoS files from a specific project directory and returns a single list of all publications.
    """
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

def write_records_to_file(records, output_dir, filename):
    """
    Writes a list of publication records to a new file in WoS format.
    """
    print(f"\n--- Writing Top {len(records)} Records to '{filename}' ---")
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
    parser = argparse.ArgumentParser(description="Extracts the top N most cited source papers from a project's dataset based on the 'TC' field.")
    parser.add_argument("project_folder", type=str, help="The name of the project folder inside 'data/wos/' (e.g., 'smart_city' or 'urban_computing')")
    parser.add_argument("--top_n", type=int, default=200, help="The number of top-cited papers to extract (default: 200).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    
    # Construct paths based on the project folder argument
    wos_data_dir = script_dir / 'data' / 'wos' / args.project_folder
    
    # Create a dedicated, project-specific output directory
    output_dir = script_dir / 'data' / 'extracted_papers' / args.project_folder
    
    # Make the output filename specific to the project and the number of papers
    output_filename = f"top_{args.top_n}_source_papers.txt"

    print(f"\n--- Starting extraction for project: {args.project_folder} ---")

    # 1. Load all publications from the specified project folder
    all_pubs = get_all_publications(wos_data_dir)

    if all_pubs:
        # 2. Sort the publications by their 'TC' (Times Cited) value
        # We use a lambda function to handle missing 'TC' fields and convert to int for sorting
        print(f"\n--- Sorting {len(all_pubs)} source papers by citation count ('TC' field) ---")
        sorted_pubs = sorted(all_pubs, key=lambda p: int(p.get('TC', 0)), reverse=True)
        
        # 3. Get the top N papers from the sorted list
        top_papers = sorted_pubs[:args.top_n]
        print(f"  Selected top {len(top_papers)} papers.")

        # 4. Write these top papers to a new file
        write_records_to_file(top_papers, output_dir, output_filename)

    print("\nScript finished.")
