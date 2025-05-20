# build_network.py

import re
import os # Added for directory operations
# import pandas as pd # Not strictly used in the core logic, can be removed if not needed elsewhere
import networkx as nx
# import matplotlib.pyplot as plt # Not used in this script version
from collections import defaultdict, Counter
import itertools
# import math # Not strictly used
from pathlib import Path # Modern way to handle paths

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
    line_num = 0 # Initialize line number for error reporting

    try:
        encodings_to_try = ['utf-8', 'latin-1', 'utf-8-sig'] # Added utf-8-sig for BOM
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

        for line_num, line_with_newline in enumerate(file_content):
            original_line = line_with_newline # Keep original line to check indentation
            line = line_with_newline.strip() # Work with the stripped version

            if not line: continue # Skip empty lines

            field_code = None
            value = None
            is_continuation = False

            if len(line_with_newline) > 3 and line_with_newline[2] == ' ' and line_with_newline[:2].isalnum() and line_with_newline[:2].isupper():
                 field_code = line_with_newline[:2]
                 value = line_with_newline[3:].strip()
            else:
                 is_continuation = True
                 value = line # Use the stripped line content as value

            if original_line.startswith(ER_FIELD) or line == ER_FIELD:
                if current_pub:
                    if UT_FIELD not in current_pub:
                        current_pub[UT_FIELD] = f"MISSING_UT_{len(publications)}"
                    publications.append(current_pub)
                current_pub = {}
                current_field = None
                continue

            if original_line.startswith(FN_FIELD) or original_line.startswith(VR_FIELD):
                 if len(line_with_newline) > 2 and line_with_newline[2] == ' ': # Check format
                     field_code = line_with_newline[:2]
                     value = line_with_newline[3:].strip()
                     is_continuation = False
                 else:
                      continue

            if not is_continuation and field_code:
                current_field = field_code
                if current_field in [CR_FIELD, AU_FIELD, AF_FIELD, C1_FIELD, DE_FIELD, ID_FIELD]:
                    if current_field not in current_pub:
                        current_pub[current_field] = []
                    current_pub[current_field].append(value)
                elif current_field in [AB_FIELD, TI_FIELD]:
                    current_pub[current_field] = value
                else:
                    current_pub[current_field] = value

            elif is_continuation and current_field:
                # WoS indent is typically 3 spaces for continuation of list items
                # For CR field, an indented line means a NEW reference, not continuation of previous string
                if current_field in [CR_FIELD, AU_FIELD, AF_FIELD, C1_FIELD, DE_FIELD, ID_FIELD]:
                    if original_line.startswith('   ') and current_field in current_pub:
                        current_pub[current_field].append(value) # Add as a NEW list item
                    elif current_field in current_pub and current_pub[current_field]:
                        # If not standard indent, append to the *last* item's string (e.g. wrapped abstract)
                        # This logic might need care for CR if a single CR entry wraps without 3-space indent
                        if current_field not in [CR_FIELD]: # Don't append to last CR item unless sure
                             current_pub[current_field][-1] += " " + value
                        elif current_field == CR_FIELD and not original_line.startswith('   '):
                             # If it's CR and not indented, it's likely a wrapped part of the *previous* CR entry
                             current_pub[current_field][-1] += " " + value
                elif current_field in [AB_FIELD, TI_FIELD]:
                    if current_field in current_pub:
                        current_pub[current_field] += " " + value
        
        if current_pub and UT_FIELD in current_pub : # Add last publication
             publications.append(current_pub)
        elif current_pub and not publications: # Handle case of single publication file without trailing ER
            if UT_FIELD not in current_pub:
                current_pub[UT_FIELD] = f"MISSING_UT_{len(publications)}"
            publications.append(current_pub)


    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred during parsing file {filepath}: {e} near line {line_num + 1}")
        import traceback
        traceback.print_exc()
        return []
    return publications

# --- Function: normalize_cited_ref (Revised) ---
def normalize_cited_ref(ref_string):
    """
    Attempts to parse and normalize a cited reference string.
    Returns a standardized string "AUTHOR, YEAR, SOURCE" or None.
    """
    ref_string = ref_string.strip()
    if not ref_string:
        return None

    # Handle known problematic prefixes like '*[SO AND SO]'
    if ref_string.startswith('*'):
        ref_string = ref_string[1:].strip()
    if ref_string.startswith('[') and ']' in ref_string:
        first_bracket_close = ref_string.find(']')
        # If it's like "[ANONYMOUS]..." or "[SOME ORG]..."
        if first_bracket_close > 0 and first_bracket_close < len(ref_string) -1 :
            # Check if what's inside looks like an org or anonymous
            content_in_bracket = ref_string[1:first_bracket_close]
            if "ANONYMOUS" in content_in_bracket.upper() or len(content_in_bracket.split()) > 2 : # Heuristic for org
                 ref_string = ref_string[first_bracket_close+1:].strip()


    parts = [p.strip() for p in ref_string.split(',')]
    if not parts:
        return None

    # 1. Author Extraction
    author_parts = []
    current_idx = 0
    # Try to gather parts that form the author name, stopping before a clear year or journal-like part
    for i, part in enumerate(parts):
        if i > 2 and len(author_parts) > 0 : # Limit author part search to avoid consuming journal
            break
        is_year = re.fullmatch(r'(1[89]\d{2}|20\d{2})', part)
        # Heuristic: if part is short and all caps (like an initial) or longer and capitalized
        is_likely_author_part = (len(part) > 0 and (len(part) <= 3 and part.isupper()) or \
                                (part.replace('.', '').replace('-', '').isalpha() and not part.islower()))
        
        # Stop if it's a year or if it looks like a journal keyword (e.g., J, PROC, CONF)
        # and we already have some author parts
        is_journal_keyword = part.upper() in ["J", "PROC", "CONF", "SYMP", "ANN", "REV", "B", "INT J", "EUR J"]
        if is_year or (is_journal_keyword and len(author_parts) > 0):
            break
        
        if is_likely_author_part or i == 0 : # Always take the first part as potential author start
            author_parts.append(part.upper())
            current_idx = i + 1
        else: # part doesn't look like author part
            if i==0: # if first part itself is not author like, then it is an issue.
                 author_parts.append(part.upper()) # Take it anyway and hope for the best
                 current_idx = i + 1
            break # Stop consuming for author

    if not author_parts:
        author_str = "ANONYMOUS"
    else:
        author_str = " ".join(author_parts)
        author_str = re.sub(r'\.', '', author_str) # Remove periods
        author_str = re.sub(r'\s+', ' ', author_str).strip()

    if not author_str or author_str.startswith("[ANONYMOUS]"): # Catch if it was explicitly anonymous
        author_str = "ANONYMOUS"

    remaining_parts = parts[current_idx:]

    # 2. Year Extraction
    year_str = "UNKNOWN_YEAR"
    year_idx_in_remaining = -1
    for i, part in enumerate(remaining_parts):
        if re.fullmatch(r'(1[89]\d{2}|20\d{2})', part):
            year_str = part
            year_idx_in_remaining = i
            break
    
    if year_str == "UNKNOWN_YEAR":
        # Fallback: search year in the whole original string if not found conventionally
        year_match_fallback = re.search(r'\b(1[89]\d{2}|20\d{2})\b', ref_string)
        if year_match_fallback:
            year_str = year_match_fallback.group(1)
            # This makes source extraction tricky, so we might have less reliable source
        else:
            return None # Strict: if no year can be found, discard reference

    # 3. Source Extraction
    source_parts_candidate = []
    if year_idx_in_remaining != -1: # Year was found among remaining_parts
        source_parts_candidate = remaining_parts[year_idx_in_remaining+1:]
    elif year_str != "UNKNOWN_YEAR": # Year was found by fallback, source is less certain
        # Try to take parts after the first author-like block and after the globally found year
        # This is heuristic. A simpler take: everything after the identified author block,
        # then try to clean it.
        # For now, if year was fallback, let's try to take all remaining_parts as source candidate
        # This might be too greedy if remaining_parts still contains the year.
        # A better way: find where year_str occurs in remaining_parts (if at all)
        temp_source_parts = []
        year_found_in_rem = False
        for part in remaining_parts:
            if part == year_str and not year_found_in_rem:
                year_found_in_rem = True
                continue # Skip the year itself
            if year_found_in_rem: # Collect parts after the year
                temp_source_parts.append(part)
        source_parts_candidate = temp_source_parts
        if not source_parts_candidate and remaining_parts: # If year was first or not in remaining_parts
             pass # source_parts_candidate remains empty or as is.

    source_str = "UNKNOWN_SOURCE"
    if source_parts_candidate:
        # Try to cut off at Vol, Page, DOI, etc.
        cleaned_source_segments = []
        for part in source_parts_candidate:
            # Common bibliographic markers that usually end the core source title
            if re.match(r'^(V|P|DOI|HTTP|WWW|ISBN|ED|ART|NO|VOL|PAGE|ISS|CHAP|SER|PP|NUM)\b', part, re.IGNORECASE):
                break
            # Heuristic: if a part is a year AND the previous part looked like an author, stop.
            # This is to prevent "SOURCE AUTHOR2, YEAR2" from being part of current source.
            if cleaned_source_segments and re.fullmatch(r'(1[89]\d{2}|20\d{2})', part):
                last_added = cleaned_source_segments[-1]
                # Simple check for author-like: ALL CAPS, no digits, at least one letter.
                if last_added.isupper() and not any(c.isdigit() for c in last_added) and re.search(r'[A-Z]', last_added):
                    # Check if 'last_added' is not a common journal word that can be all caps
                    common_journal_words = {"IEEE", "ACM", "ASCE", "ETRI", "J", "INT"} # Add more if needed
                    if last_added not in common_journal_words and len(last_added.split()) < 4: # Avoid long titles
                        break # Likely start of a new reference
            cleaned_source_segments.append(part)
        
        if cleaned_source_segments:
            source_str = ", ".join(cleaned_source_segments).strip().upper()
            source_str = source_str.strip(', ') # Remove leading/trailing commas/spaces
    
    # Further cleaning of the source string
    if source_str:
        # Remove any remaining DOI string
        source_str = re.sub(r',?\s*DOI\s*:?\s*\S+', '', source_str, flags=re.IGNORECASE).strip()
        source_str = source_str.strip(', ')
        
        # Specific known variations
        if source_str.endswith("-BASEL"):
            source_str = source_str[:-6].strip()
        # Example: "J AM PLANN ASSOC" -> "JOURNAL OF THE AMERICAN PLANNING ASSOCIATION" (can be a map)
        # journal_map = {"J AM PLANN ASSOC": "JOURNAL OF THE AMERICAN PLANNING ASSOCIATION"}
        # if source_str in journal_map: source_str = journal_map[source_str]

    if not source_str or source_str == year_str: # If source became empty or just the year
        source_str = "UNKNOWN_SOURCE"

    return f"{author_str}, {year_str}, {source_str}"


# --- Function: build_cocitation_network ---
def build_cocitation_network(publications):
    """ Builds a co-citation network from parsed WoS publications (with enhanced debugging). """
    G = nx.Graph()
    cited_ref_counts = Counter()
    cocitation_links = defaultdict(int)
    node_info = {} 

    print(f"Building network from {len(publications)} citing publications...")
    processed_pubs = 0
    
    # --- Enhanced Debugging Control ---
    # Set how many initial publications to print detailed debug info for
    DETAILED_DEBUG_COUNT = 5 # Adjust as needed
    # ---

    for pub_idx, pub in enumerate(publications):
        citing_pub_id = pub.get(UT_FIELD, f'UnknownUT_{processed_pubs}')
        # citing_pub_year = pub.get(PY_FIELD, None) # Year the co-citation happened (not used currently for edges)
        cited_refs_raw = pub.get(CR_FIELD, [])

        # --- Determine if detailed debugging should be printed for this publication ---
        print_this_pub_detailed_debug = (pub_idx < DETAILED_DEBUG_COUNT)

        if print_this_pub_detailed_debug:
            print(f"\n--- Detailed Debug for Citing Pub ID: {citing_pub_id} (Index: {pub_idx}) ---")
            print(f"  Raw CRs ({len(cited_refs_raw)}):")
            if not cited_refs_raw:
                print("    This publication has no raw CR entries.")
            for i, r_cr in enumerate(cited_refs_raw):
                print(f"    Raw CR [{i}]: {r_cr}")

        normalized_refs_for_this_pub = []
        for ref_str_idx, ref_str in enumerate(cited_refs_raw):
            norm_ref = normalize_cited_ref(ref_str) 
            
            if print_this_pub_detailed_debug: # Print normalization result for each raw CR
                print(f"    Raw CR [{ref_str_idx}] processed. Normalized: {norm_ref}")

            if norm_ref:
                normalized_refs_for_this_pub.append(norm_ref)
                cited_ref_counts[norm_ref] += 1 # Count total citations for each ref (for node size)
                if norm_ref not in node_info: # Store info for node attributes
                     parts = norm_ref.split(', ') # Expecting "AUTHOR, YEAR, SOURCE"
                     ref_author = parts[0] if len(parts) > 0 else "UNKNOWN_AUTHOR"
                     ref_year = parts[1] if len(parts) > 1 and parts[1].isdigit() else "UNKNOWN_YEAR"
                     ref_source = parts[2] if len(parts) > 2 else "UNKNOWN_SOURCE"
                     node_info[norm_ref] = {'author': ref_author, 'year': ref_year, 'source': ref_source, 'label': norm_ref}

        if print_this_pub_detailed_debug:
            print(f"  Normalized Refs for this pub ({len(normalized_refs_for_this_pub)}):")
            max_norm_to_print_debug = 20
            for i_nr, nr_item in enumerate(normalized_refs_for_this_pub[:max_norm_to_print_debug]):
                 print(f"    Norm Ref [{i_nr}]: {nr_item}")
            if len(normalized_refs_for_this_pub) > max_norm_to_print_debug:
                 print(f"    ... (and {len(normalized_refs_for_this_pub) - max_norm_to_print_debug} more normalized refs)")

        # Generate co-citation pairs (edges)
        pair_count_this_pub = 0
        if len(normalized_refs_for_this_pub) >= 2:
            for ref1, ref2 in itertools.combinations(normalized_refs_for_this_pub, 2):
                edge = tuple(sorted((ref1, ref2))) # Ensure consistent edge ordering
                cocitation_links[edge] += 1
                pair_count_this_pub += 1
            if print_this_pub_detailed_debug:
                 print(f"  Found {pair_count_this_pub} co-citation pairs for this publication.")
        else:
            if print_this_pub_detailed_debug:
                 print(f"  Not enough normalized references (found {len(normalized_refs_for_this_pub)}, need >= 2) to form pairs for this pub.")
        
        processed_pubs += 1
        if processed_pubs % 200 == 0: # Print progress less often
             print(f"  Processed {processed_pubs}/{len(publications)} publications...")

    print(f"\nFinished processing publications.")
    print(f"Total unique cited references identified (potential nodes): {len(cited_ref_counts)}")
    print(f"Total unique co-citation links found (potential edge types): {len(cocitation_links)}")
    total_cocitations = sum(cocitation_links.values())
    print(f"Total co-citation instances (sum of edge weights): {total_cocitations}")

    # Add nodes to the graph
    print("Adding nodes to graph...")
    for ref_key, count in cited_ref_counts.items():
        info = node_info.get(ref_key, {}) # Get the stored author, year, source
        G.add_node(
            ref_key, # The node ID is the normalized string "AUTHOR, YEAR, SOURCE"
            label=info.get('label', ref_key), 
            freq=count,
            author=info.get('author', 'Unknown'),
            year=info.get('year', 'Unknown'),
            source=info.get('source', 'Unknown')
        )

    # Add edges to the graph
    print("Adding edges to graph...")
    edges_added_count = 0
    for (ref1, ref2), weight in cocitation_links.items():
        if G.has_node(ref1) and G.has_node(ref2): # Should always be true if logic is correct
            G.add_edge(ref1, ref2, weight=float(weight))
            edges_added_count +=1
        else:
            print(f"Warning: Skipping edge ({ref1}, {ref2}) because one or both nodes not in graph.")


    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    if G.number_of_edges() == 0 and G.number_of_nodes() > 0:
        print("WARNING: No edges were created. Check normalization and co-citation logic.")
        print("Consider increasing DETAILED_DEBUG_COUNT for more verbose output on problematic CR strings.")
    return G

# --- Function: save_graph_to_graphml ---
def save_graph_to_graphml(graph, filepath):
    """Saves the networkx graph to a GraphML file using lxml."""
    if graph.number_of_nodes() == 0:
        print("Graph is empty, skipping save.")
        return

    print(f"Preparing graph for saving to {filepath}...")
    # Ensure data types are correct before saving 
    # NetworkX's write_graphml_lxml usually infers types, but being explicit can help.
    for node_id, data in graph.nodes(data=True):
        data['freq'] = int(data.get('freq', 0))
        data['year'] = str(data.get('year', '')) 
        data['author'] = str(data.get('author', ''))
        data['source'] = str(data.get('source', ''))
        data['label'] = str(data.get('label', node_id)) # Ensure label is string

        # For Gephi/VOSviewer, sometimes 'name' or 'title' is used for display
        data['name'] = str(data.get('label', node_id)) 
        # Placeholders for other common attributes if not extracted:
        if 'title' not in data: data['title'] = '...' # WoS CR doesn't usually have cited title
        if 'vol' not in data: data['vol'] = ''
        if 'page' not in data: data['page'] = ''
        if 'ut' not in data: data['ut'] = ''


    for u, v, data in graph.edges(data=True):
        data['weight'] = float(data.get('weight', 0.0))
        # Placeholders for common edge attributes if needed
        if 'slice' not in data: data['slice'] = int(0) 
        if 'edge_year' not in data: data['edge_year'] = '' # Year co-citation occurred (if tracked)

    try:
        output_dir = Path(filepath).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Using infer_numeric_types=True can be helpful.
        # named_key_ids=True makes the GraphML more human-readable.
        nx.write_graphml_lxml(graph, str(filepath), named_key_ids=True, infer_numeric_types=True)
        print(f"Graph successfully saved to {filepath}")
    except ImportError:
        print("Error: Saving to GraphML with lxml requires the 'lxml' library. Install it (e.g., pip install lxml).")
        print("Attempting to save with basic XML writer (may have limitations).")
        try:
            nx.write_graphml_xml(graph, str(filepath), named_key_ids=True) # Basic writer
            print(f"Graph saved using basic XML writer.")
        except Exception as e_xml:
            print(f"Error saving graph with basic XML writer: {e_xml}")
    except Exception as e:
        print(f"An error occurred saving the graph: {e}")
        import traceback
        traceback.print_exc()

# --- Main Execution Logic ---
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent # Use resolve() for more robust path
    wos_data_dir = script_dir / 'data' / 'wos'
    graphml_output_dir = script_dir / 'data' / 'graphml'
    
    # --- Configuration: Set your input file pattern and output filename ---
    # Example: process all 'savedrecs*.txt' or a specific file like 'my_data.txt'
    # file_pattern_glob = 'savedrecs*.txt' 
    file_pattern_glob = 'savedrecs*.txt' # Default to original pattern
    # You can also process a single file:
    # wos_files_to_process = [wos_data_dir / "your_single_file.txt"]

    output_filename = 'python_cocitation_network_revised.graphml'
    # --- End Configuration ---

    output_filepath = graphml_output_dir / output_filename

    print(f"Looking for WoS files matching '{file_pattern_glob}' in: {wos_data_dir}")

    all_publications = []
    if not wos_data_dir.is_dir():
        print(f"Error: Input directory not found: {wos_data_dir}")
    else:
        # --- File discovery and sorting (from original script) ---
        all_txt_files = list(wos_data_dir.glob(file_pattern_glob))
        
        target_files = []
        # Regex to match pattern like "savedrecs.txt" or "savedrecs (1).txt"
        # This assumes the base name before parenthesis is 'savedrecs'
        # If your pattern is just '*.txt', the sorting needs to be simpler (e.g. alphanumeric)
        file_name_pattern_re = re.compile(r"^(.*?)(?: \((\d+)\))?\.txt$")
        
        files_with_num = []
        if file_pattern_glob == 'savedrecs*.txt': # Apply numeric sort only for this specific pattern
            for f_path in all_txt_files:
                match = file_name_pattern_re.match(f_path.name)
                if match:
                    base_name = match.group(1) # e.g., "savedrecs"
                    num_str = match.group(2)
                    num = int(num_str) if num_str else 0 # No number in () means 0 for sorting
                    if base_name == "savedrecs": # Ensure it's the correct base filename
                         files_with_num.append((num, f_path))
            
            if files_with_num: # If we found files matching the numbered pattern
                files_with_num.sort(key=lambda x: x[0]) # Sort by the extracted number
                wos_files_to_process = [f_path for num, f_path in files_with_num]
            else: # Fallback to simple sort if no numbered 'savedrecs' files found
                wos_files_to_process = sorted(all_txt_files)
        else: # For other glob patterns, use simple alphanumeric sort
            wos_files_to_process = sorted(all_txt_files)
        # --- End file discovery and sorting ---

        print(f"Found {len(wos_files_to_process)} files to process (sorted):")
        for f_p in wos_files_to_process:
            print(f"  - {f_p.name}")

        if not wos_files_to_process:
            print(f"No files matching the pattern '{file_pattern_glob}' found in {wos_data_dir}.")
        else:
            for wos_file_path in wos_files_to_process:
                print(f"\nParsing {wos_file_path.name}...")
                pubs_from_file = parse_wos_file(wos_file_path)
                all_publications.extend(pubs_from_file)
                print(f"  Added {len(pubs_from_file)} publications from {wos_file_path.name}. Total: {len(all_publications)}")

            if not all_publications:
                print("No publications were parsed successfully from any file.")
            else:
                print(f"\nTotal publications loaded: {len(all_publications)}")
                G_combined = build_cocitation_network(all_publications)
                save_graph_to_graphml(G_combined, output_filepath)

    print("\nScript finished.")
