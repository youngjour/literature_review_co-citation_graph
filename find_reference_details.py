import re
from pathlib import Path
from collections import defaultdict
import argparse 

from build_network import parse_wos_file 

def find_reference_details_by_ut(target_refs, wos_data_dir):
    """
    Searches for detailed information of target references by first building a
    UT-keyed database and a searchable index.
    
    :param target_refs: A list of strings representing the references to find.
                        (e.g., ["Caragliu 2011", "Zanella 2014"])
    :param wos_data_dir: Path to the directory containing wos txt files.
    """
    print(f"--- Building Publication Database for Project in: {wos_data_dir.name} ---")
    
    # File discovery and sorting
    file_pattern_glob = 'savedrecs*.txt'
    all_txt_files_found = list(wos_data_dir.glob(file_pattern_glob))
    if not all_txt_files_found:
        print(f"Error: No files matching '{file_pattern_glob}' found in {wos_data_dir}")
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

    # Build a UT-keyed primary database AND a search index
    pub_database_by_ut = {}
    search_index = defaultdict(list)
    
    total_pubs = 0
    for wos_file_path in wos_files_to_process:
        # print(f"  Parsing {wos_file_path.name}...") # Uncomment for more verbose output
        pubs_from_file = parse_wos_file(wos_file_path)
        for pub in pubs_from_file:
            ut = pub.get('UT')
            if not ut:
                continue
            
            pub_database_by_ut[ut] = pub
            
            authors = pub.get('AU', [])
            year = pub.get('PY')
            if authors and year:
                first_author_surname = authors[0].split(',')[0].strip().upper()
                lookup_key = (first_author_surname, year)
                search_index[lookup_key].append(ut)
        total_pubs += len(pubs_from_file)
        
    print(f"\n--- Database Built. Parsed {total_pubs} publications. ---\n")

    # --- Search for Target References using the two-step method ---
    print("--- Searching for Target References ---")
    for target in target_refs:
        print(f"\n>>> Searching for: '{target}'")
        
        # This regex is slightly more robust to handle author names with dots etc.
        match = re.search(r'([a-zA-Z\s\.\-]+?)\s*(\d{4})', target)
        if not match:
            print("    Could not parse target string. Please use 'Author Year' format.")
            continue
            
        target_author_surname = match.group(1).strip().upper().split(' ')[0]
        target_year = match.group(2).strip()
        
        search_key = (target_author_surname, target_year)
        
        found_uts = search_index.get(search_key)
        
        if found_uts:
            print(f"    SUCCESS: Found {len(found_uts)} potential match(es) in your dataset.")
            for i, ut in enumerate(found_uts):
                record = pub_database_by_ut[ut]
                print(f"\n    --- Record {i+1} (UT: {ut}) ---")
                print(f"    Title: {record.get('TI', 'N/A')}")
                print(f"    Authors: {'; '.join(record.get('AU', []))}")
                print(f"    Journal/Source: {record.get('SO', 'N/A')}")
                print(f"    Year: {record.get('PY', 'N/A')}")
                print(f"    Abstract: {record.get('AB', 'N/A')}")
        else:
            print(f"    INFO: Reference not found in your downloaded dataset.")
            google_scholar_query = f"https://scholar.google.com/scholar?q={target.replace(' ', '+')}"
            print(f"    You can search for it externally: {google_scholar_query}")


if __name__ == "__main__":
    # Setup to read all inputs from the command line
    parser = argparse.ArgumentParser(description="Find details for specific papers within a project's dataset.")
    parser.add_argument("project_folder", type=str, help="The name of the project folder inside 'data/wos/' (e.g., 'smart_city')")
    parser.add_argument("references", nargs='+', type=str, help="A list of references to find, in 'Author Year' format (e.g., \"Caragliu 2011\")")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # The path now includes the project folder you specify on the command line
    wos_data_dir = script_dir / 'data' / 'wos' / args.project_folder
    
    # The list of references now comes from the command line
    references_to_find = args.references
    
    find_reference_details_by_ut(references_to_find, wos_data_dir)

    print("\nScript finished.")
