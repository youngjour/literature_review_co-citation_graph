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

            # Check if the line starts with a 2-character field code followed by a space
            # Use line_with_newline to preserve original spacing for field code detection
            if len(line_with_newline) > 2 and line_with_newline[2] == ' ' and line_with_newline[:2].isalnum() and line_with_newline[:2].isupper():
                 field_code = line_with_newline[:2]
                 value = line_with_newline[3:].strip() # Get value from the rest of the line
            else:
                 # This line does not start with a field code, so it's a continuation or a special line
                 is_continuation = True
                 value = line # Use the stripped line content as value for continuation

            # Handle End of Record (ER)
            # ER might not have the standard 2-char + space format, so check original_line
            if original_line.startswith(ER_FIELD) or line == ER_FIELD:
                if current_pub: # If we have data for a publication
                    # Add UT if missing before saving
                    if UT_FIELD not in current_pub:
                        current_pub[UT_FIELD] = f"MISSING_UT_{len(publications)}"
                    publications.append(current_pub)
                current_pub = {} # Reset for the next publication
                current_field = None # Reset current field context
                continue # Move to the next line after processing ER

            # Handle special start-of-file tags like FN, VR if necessary
            if original_line.startswith(FN_FIELD) or original_line.startswith(VR_FIELD):
                 # These might signal the start, reset context if needed, but usually handled by ER
                 # We can parse them like normal fields if we hit them
                 if len(line_with_newline) > 2 and line_with_newline[2] == ' ': # Check format just in case
                     field_code = line_with_newline[:2]
                     value = line_with_newline[3:].strip()
                     is_continuation = False # Treat as a new field
                 else: # If format is weird (e.g., just "FN"), skip or log?
                      continue


            # Process based on whether it's a new field or continuation
            if not is_continuation and field_code:
                # === It's a new field ===
                current_field = field_code # Update the current field context

                # Fields where each line (starting with code OR indented) is a SEPARATE item
                if current_field in [CR_FIELD, AU_FIELD, AF_FIELD, C1_FIELD, DE_FIELD, ID_FIELD]:
                    if current_field not in current_pub:
                        current_pub[current_field] = []
                    current_pub[current_field].append(value) # Add value as a new list item

                # Fields where continuation lines should be appended to the value (e.g., Abstract)
                elif current_field in [AB_FIELD, TI_FIELD]:
                    current_pub[current_field] = value # Initialize the string value

                # Other single-value fields
                else:
                    current_pub[current_field] = value # Assign the value directly

            elif is_continuation and current_field:
                # === It's a continuation line ===
                # WoS indent is typically 3 spaces for continuation of list items
                # For CR field, an indented line means a NEW reference, not continuation of previous string
                if current_field in [CR_FIELD, AU_FIELD, AF_FIELD, C1_FIELD, DE_FIELD, ID_FIELD]:
                    if original_line.startswith('   ') and current_field in current_pub:
                        # Indented line for list field -> Add as a NEW item
                        current_pub[current_field].append(value)
                    elif current_field in current_pub and current_pub[current_field]:
                        # Not standard indent, or not a list item continuation (e.g. wrapped abstract part)
                        # For CR, if it's not 3-space indented, it's likely part of the *previous* CR string that wrapped.
                        if current_field == CR_FIELD and not original_line.startswith('   '):
                            current_pub[current_field][-1] += " " + value
                        elif current_field not in [CR_FIELD]: # For other list fields, append to last item if not indented
                             current_pub[current_field][-1] += " " + value
                elif current_field in [AB_FIELD, TI_FIELD]: # For string fields like Abstract or Title
                    if current_field in current_pub:
                        current_pub[current_field] += " " + value # Append to string
        
        # Add the last publication if the file doesn't end neatly with ER
        if current_pub and UT_FIELD in current_pub :
             publications.append(current_pub)
        elif current_pub and not publications: # Handle case of single publication file without trailing ER
            if UT_FIELD not in current_pub: # Ensure UT exists
                current_pub[UT_FIELD] = f"MISSING_UT_{len(publications)}"
            publications.append(current_pub)


    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred during parsing file {filepath}: {e} near line {line_num + 1}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return [] # Return empty or partially parsed data
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

    # Handle known problematic prefixes like '*[SO AND SO]' or leading '*'
    if ref_string.startswith('*'):
        ref_string = ref_string[1:].strip()
    # Handle cases like "[ANONYMOUS] ..." or "[AN ORGANISATION] ..."
    if ref_string.startswith('[') and ']' in ref_string:
        first_bracket_close = ref_string.find(']')
        # Check if it's like "[ANONYMOUS]..." or "[SOME ORG]..."
        if first_bracket_close > 0 and first_bracket_close < len(ref_string) -1 :
            content_in_bracket = ref_string[1:first_bracket_close]
            # Heuristic: if content is "ANONYMOUS" or looks like an organization name (multiple words)
            if "ANONYMOUS" in content_in_bracket.upper() or len(content_in_bracket.split()) > 1 :
                 ref_string = ref_string[first_bracket_close+1:].strip() # Use the part after the bracket


    parts = [p.strip() for p in ref_string.split(',')]
    if not parts:
        return None

    # 1. Author Extraction
    author_parts = []
    current_idx = 0 # Index in `parts` to start looking for year/source
    # Try to gather parts that form the author name, stopping before a clear year or journal-like part
    for i, part in enumerate(parts):
        if i > 2 and len(author_parts) > 0 : # Limit author part search to avoid consuming journal too early
            break
        is_year_format = re.fullmatch(r'(1[89]\d{2}|20\d{2})', part) is not None
        
        # Heuristic: if part is short and all caps (like an initial) or longer and capitalized
        is_likely_author_part = (len(part) > 0 and (
                                 (len(part) <= 3 and part.isupper() and part.isalpha()) or \
                                 (part.replace('.', '').replace('-', '').isalpha() and not part.islower())
                                ))
        
        # Stop if it's a year or if it looks like a journal keyword (e.g., J, PROC, CONF)
        # and we already have some author parts
        is_journal_keyword = part.upper() in ["J", "PROC", "CONF", "SYMP", "ANN", "REV", "B", "INT J", "EUR J", "IEEE", "ACM"]
        if is_year_format or (is_journal_keyword and len(author_parts) > 0):
            break # Stop collecting author parts
        
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
        author_str = re.sub(r'\s+', ' ', author_str).strip() # Consolidate spaces

    if not author_str or author_str.startswith("[ANONYMOUS]"): # Catch if it was explicitly anonymous
        author_str = "ANONYMOUS"

    remaining_parts = parts[current_idx:]

    # 2. Year Extraction
    year_str = "UNKNOWN_YEAR"
    year_idx_in_remaining = -1 # Index of the year within remaining_parts
    for i, part in enumerate(remaining_parts):
        if re.fullmatch(r'(1[89]\d{2}|20\d{2})', part):
            year_str = part
            year_idx_in_remaining = i
            break # Found year, stop
    
    if year_str == "UNKNOWN_YEAR":
        # Fallback: search year in the whole original ref_string if not found conventionally
        year_match_fallback = re.search(r'\b(1[89]\d{2}|20\d{2})\b', ref_string)
        if year_match_fallback:
            year_str = year_match_fallback.group(1)
            # If year found by fallback, source extraction is trickier.
            # We'll try to use parts after the author block, then clean.
        else:
            # Strict: if no year can be found, discard reference.
            # Alternatively, you could return with "UNKNOWN_YEAR" but this might create noisy nodes.
            return None 

    # 3. Source Extraction
    source_parts_candidate = []
    if year_idx_in_remaining != -1: # Year was found among remaining_parts (ideal case)
        source_parts_candidate = remaining_parts[year_idx_in_remaining+1:]
    elif year_str != "UNKNOWN_YEAR": # Year was found by fallback
        # Source is everything in remaining_parts, but we need to be careful if the year itself is in remaining_parts
        temp_source_parts = []
        year_found_in_rem_for_source = False
        for part in remaining_parts:
            if part == year_str and not year_found_in_rem_for_source:
                year_found_in_rem_for_source = True # Mark that we've passed the year
                continue # Skip the year itself if it was in remaining_parts
            if year_found_in_rem_for_source or year_idx_in_remaining == -1: # Collect parts after the year, or all if year wasn't in remaining
                temp_source_parts.append(part)
        source_parts_candidate = temp_source_parts
        if not source_parts_candidate and remaining_parts and year_idx_in_remaining == -1:
            # If year was fallback and not in remaining_parts, all remaining_parts are source candidates
             source_parts_candidate = remaining_parts


    source_str = "UNKNOWN_SOURCE"
    if source_parts_candidate:
        # Try to cut off at Vol, Page, DOI, etc.
        cleaned_source_segments = []
        for part in source_parts_candidate:
            # Common bibliographic markers that usually end the core source title
            # Make regex more robust for things like V23, P110, DOI10.1000/..., HTTP://...
            if re.match(r'^(V\d*|P\d*|DOI[\s:]?|HTTP|WWW|ISBN|ED|ARTNO|VOL|PAGE|ISS|CHAP|SER|PP|NUM)\b', part, re.IGNORECASE):
                break # Stop at these markers
            
            # Heuristic: if a part is a year AND the previous part looked like an author, stop.
            # This is to prevent "SOURCE AUTHOR2, YEAR2" from being part of current source.
            if cleaned_source_segments and re.fullmatch(r'(1[89]\d{2}|20\d{2})', part):
                last_added = cleaned_source_segments[-1].strip()
                # Simple check for author-like: ALL CAPS, no digits, at least one letter.
                if last_added.isupper() and not any(c.isdigit() for c in last_added) and re.search(r'[A-Z]', last_added):
                    # Check if 'last_added' is not a common journal word that can be all caps
                    common_journal_words = {"IEEE", "ACM", "ASCE", "ETRI", "J", "INT", "EUR"} # Add more if needed
                    if last_added not in common_journal_words and len(last_added.split()) < 4: # Avoid long titles
                        break # Likely start of a new reference embedded in the source string
            cleaned_source_segments.append(part)
        
        if cleaned_source_segments:
            source_str = ", ".join(cleaned_source_segments).strip().upper()
            source_str = source_str.strip(', ') # Remove leading/trailing commas/spaces
    
    # Further cleaning of the source string
    if source_str:
        # Remove any remaining DOI string more aggressively
        source_str = re.sub(r',?\s*DOI[:\s]*\S+', '', source_str, flags=re.IGNORECASE).strip()
        source_str = re.sub(r'\s*DOI\s*$', '', source_str, flags=re.IGNORECASE).strip() # trailing DOI
        source_str = source_str.strip(', ')
        
        # Specific known variations (add more as needed)
        if source_str.endswith("-BASEL"):
            source_str = source_str[:-6].strip()
        # Example: "J AM PLANN ASSOC" -> "JOURNAL OF THE AMERICAN PLANNING ASSOCIATION" (can be a map)
        # journal_map = {"J AM PLANN ASSOC": "JOURNAL OF THE AMERICAN PLANNING ASSOCIATION", "SUSTAINABILITY-BASEL": "SUSTAINABILITY"}
        # if source_str in journal_map: source_str = journal_map[source_str]

    if not source_str or source_str == year_str: # If source became empty or just the year
        source_str = "UNKNOWN_SOURCE"

    return f"{author_str}, {year_str}, {source_str}"


# --- Function: build_cocitation_network ---
def build_cocitation_network(publications, min_node_citations_threshold=1, min_cocitation_strength_threshold=1):
    """
    Builds a co-citation network from parsed WoS publications.
    Includes options to filter nodes by minimum citation count and
    edges by minimum co-citation strength.
    """
    G = nx.Graph()
    cited_ref_counts = Counter() # Stores frequency of each normalized reference
    cocitation_links = defaultdict(int) # Stores strength of co-citation between pairs
    node_info = {} # Stores attributes (author, year, source) for each node

    print(f"Building network from {len(publications)} citing publications...")
    print(f"Applying: Min Node Citations >= {min_node_citations_threshold}, Min Co-citation Strength >= {min_cocitation_strength_threshold}")
    processed_pubs = 0
    
    # --- Enhanced Debugging Control ---
    # Set how many initial publications to print detailed debug info for
    DETAILED_DEBUG_COUNT = 5 # Adjust as needed (e.g., 0 to disable, 5-10 for debugging)
    # ---

    for pub_idx, pub in enumerate(publications):
        citing_pub_id = pub.get(UT_FIELD, f'UnknownUT_{processed_pubs}')
        # citing_pub_year = pub.get(PY_FIELD, None) # Year the co-citation happened (not used currently for edges)
        cited_refs_raw = pub.get(CR_FIELD, []) # Get list of raw cited reference strings

        # --- Determine if detailed debugging should be printed for this publication ---
        print_this_pub_detailed_debug = (pub_idx < DETAILED_DEBUG_COUNT)

        if print_this_pub_detailed_debug:
            print(f"\n--- Detailed Debug for Citing Pub ID: {citing_pub_id} (Index: {pub_idx}) ---")
            print(f"  Raw CRs ({len(cited_refs_raw)}):")
            if not cited_refs_raw:
                print("    This publication has no raw CR entries.")
            for i, r_cr in enumerate(cited_refs_raw):
                print(f"    Raw CR [{i}]: {r_cr}")

        normalized_refs_for_this_pub = [] # Collect normalized references for *this* citing paper
        for ref_str_idx, ref_str in enumerate(cited_refs_raw):
            norm_ref = normalize_cited_ref(ref_str) # Normalize each raw reference string
            
            if print_this_pub_detailed_debug: # Print normalization result for each raw CR
                print(f"    Raw CR [{ref_str_idx}] processed. Normalized: {norm_ref}")

            if norm_ref: # If normalization was successful
                normalized_refs_for_this_pub.append(norm_ref)
                cited_ref_counts[norm_ref] += 1 # Count total citations for each ref (for node size/filtering)
                if norm_ref not in node_info: # Store info for node attributes if first time seeing this ref
                     parts = norm_ref.split(', ') # Expecting "AUTHOR, YEAR, SOURCE"
                     ref_author = parts[0] if len(parts) > 0 else "UNKNOWN_AUTHOR"
                     ref_year = parts[1] if len(parts) > 1 and parts[1].isdigit() else "UNKNOWN_YEAR"
                     ref_source = ", ".join(parts[2:]) if len(parts) > 2 else "UNKNOWN_SOURCE" # Source can have commas
                     node_info[norm_ref] = {'author': ref_author, 'year': ref_year, 'source': ref_source, 'label': norm_ref}

        if print_this_pub_detailed_debug:
            print(f"  Normalized Refs for this pub ({len(normalized_refs_for_this_pub)}):")
            max_norm_to_print_debug = 20 # Limit how many normalized refs are printed
            for i_nr, nr_item in enumerate(normalized_refs_for_this_pub[:max_norm_to_print_debug]):
                 print(f"    Norm Ref [{i_nr}]: {nr_item}")
            if len(normalized_refs_for_this_pub) > max_norm_to_print_debug:
                 print(f"    ... (and {len(normalized_refs_for_this_pub) - max_norm_to_print_debug} more normalized refs)")

        # Generate co-citation pairs (edges) from the list of normalized refs for this publication
        pair_count_this_pub = 0
        if len(normalized_refs_for_this_pub) >= 2: # Need at least two refs to form a pair
            for ref1, ref2 in itertools.combinations(normalized_refs_for_this_pub, 2):
                edge = tuple(sorted((ref1, ref2))) # Ensure consistent edge ordering (e.g., (A,B) not (B,A))
                cocitation_links[edge] += 1 # Increment co-citation count for this pair
                pair_count_this_pub += 1
            if print_this_pub_detailed_debug:
                 print(f"  Found {pair_count_this_pub} co-citation pairs for this publication.")
        else:
            if print_this_pub_detailed_debug:
                 print(f"  Not enough normalized references (found {len(normalized_refs_for_this_pub)}, need >= 2) to form pairs for this pub.")
        
        processed_pubs += 1
        if processed_pubs % 200 == 0: # Print progress update periodically
             print(f"  Processed {processed_pubs}/{len(publications)} publications...")

    print(f"\nFinished processing publications.")
    print(f"Total unique cited references identified (potential nodes): {len(cited_ref_counts)}")
    print(f"Total unique co-citation links found (potential edge types): {len(cocitation_links)}")
    total_cocitations_instances = sum(cocitation_links.values())
    print(f"Total co-citation instances (sum of edge weights): {total_cocitations_instances}")

    # Add nodes to the graph (initially, all unique cited references found)
    print("Adding nodes to graph...")
    for ref_key, count in cited_ref_counts.items():
        info = node_info.get(ref_key, {}) # Get the stored author, year, source
        G.add_node(
            ref_key, # The node ID is the normalized string "AUTHOR, YEAR, SOURCE"
            label=info.get('label', ref_key), 
            freq=count, # Store total citation frequency of this reference
            author=info.get('author', 'Unknown'),
            year=info.get('year', 'Unknown'),
            source=info.get('source', 'Unknown')
        )
    print(f"  Initial nodes added to graph: {G.number_of_nodes()}")

    # Add edges to the graph, applying the co-citation strength threshold
    print("Adding edges to graph (applying co-citation strength filter)...")
    edges_added_after_strength_filter = 0
    initial_cocitation_link_types = len(cocitation_links)
    for (ref1, ref2), weight in cocitation_links.items():
        if weight >= min_cocitation_strength_threshold: # Apply edge strength filter
            # Nodes should exist if they were in cited_ref_counts
            if G.has_node(ref1) and G.has_node(ref2): 
                G.add_edge(ref1, ref2, weight=float(weight))
                edges_added_after_strength_filter +=1
            # else:
                # This case should be rare if node addition logic is correct
                # print(f"Warning: Skipping edge ({ref1}, {ref2}) because one or both nodes not in graph after initial node addition.")

    print(f"  Initial potential co-citation link types: {initial_cocitation_link_types}")
    print(f"  Edges added after strength filter (weight >= {min_cocitation_strength_threshold}): {edges_added_after_strength_filter}")
    print(f"  Graph state after edge filtering: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")


    # --- Apply Node Citation Threshold Filter ---
    # This filter is applied *after* edges are formed based on co-citation strength.
    # It removes nodes (and their connected edges) if the node's overall citation frequency is too low.
    if min_node_citations_threshold > 1: # Only filter if threshold is meaningful
        print(f"\nApplying node citation count threshold (freq >= {min_node_citations_threshold})...")
        nodes_before_freq_filter = G.number_of_nodes()
        
        # Identify nodes that meet the frequency threshold
        nodes_to_keep_after_freq_filter = [
            node for node, data in G.nodes(data=True) 
            if data.get('freq', 0) >= min_node_citations_threshold
        ]
        
        # Create a new graph containing only the nodes to keep and their induced edges
        G_filtered_by_freq = G.subgraph(nodes_to_keep_after_freq_filter).copy() 
        
        nodes_after_freq_filter = G_filtered_by_freq.number_of_nodes()
        edges_after_freq_filter = G_filtered_by_freq.number_of_edges()
        print(f"  Nodes before citation frequency filter: {nodes_before_freq_filter}, Nodes after: {nodes_after_freq_filter}")
        print(f"  Edges remaining after node frequency filter: {edges_after_freq_filter}")

        # Optionally, remove isolated nodes from the filtered graph (nodes with no edges after all filtering)
        # This can be useful if previous filtering steps left some nodes disconnected.
        # G_final = G_filtered_by_freq.copy() # or work directly on G_filtered_by_freq
        # G_final.remove_nodes_from(list(nx.isolates(G_final)))
        # print(f"  Nodes after removing isolates from final graph: {G_final.number_of_nodes()}")
        # print(f"  Edges after removing isolates from final graph: {G_final.number_of_edges()}")
        # return G_final
        
        print(f"\nFinal graph: {G_filtered_by_freq.number_of_nodes()} nodes, {G_filtered_by_freq.number_of_edges()} edges.")
        if G_filtered_by_freq.number_of_edges() == 0 and G_filtered_by_freq.number_of_nodes() > 0:
            print("WARNING: No edges remain after all filtering. Check your threshold values.")
        return G_filtered_by_freq
    else:
        # If min_node_citations_threshold is 1 (or less), no node frequency filtering is applied beyond initial construction
        print(f"\nFinal graph (no additional node frequency filtering applied): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        if G.number_of_edges() == 0 and G.number_of_nodes() > 0:
             print("WARNING: No edges were created. Check normalization and co-citation logic, and min_cocitation_strength_threshold.")
        return G # Return the graph filtered only by edge strength (if its threshold was > 1)

# --- Function: save_graph_to_graphml ---
def save_graph_to_graphml(graph, filepath):
    """Saves the networkx graph to a GraphML file using lxml."""
    if graph is None or graph.number_of_nodes() == 0: # Check if graph is None
        print("Graph is empty or None, skipping save.")
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
        if 'ut' not in data: data['ut'] = '' # UT of the cited reference is usually not available


    for u, v, data in graph.edges(data=True):
        data['weight'] = float(data.get('weight', 0.0))
        # Placeholders for common edge attributes if needed
        if 'slice' not in data: data['slice'] = int(0) # Example placeholder
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
    file_pattern_glob = 'savedrecs*.txt' # Default to original pattern
    # For example, if you have a single file named 'my_wos_data.txt':
    # file_pattern_glob = 'my_wos_data.txt' 
    # Make sure it's in the 'data/wos/' directory, or adjust path accordingly.
    # wos_files_to_process = [wos_data_dir / "your_single_file.txt"] # Alternative for single file

    output_filename = 'python_cocitation_network_filtered.graphml' # Changed output filename
    # --- End Configuration ---

    output_filepath = graphml_output_dir / output_filename

    # --- Filtering Thresholds ---
    # Set these values to filter the graph. Default is 1 (no effective filtering).
    # Adjust these based on your dataset size and desired level of detail.
    MIN_NODE_CITATIONS = 5    # Example: Node must be cited at least 5 times overall in your dataset
    MIN_COCITATION_STRENGTH = 2 # Example: Edge must have a co-citation weight of at least 2
    # ---
    print(f"Looking for WoS files matching '{file_pattern_glob}' in: {wos_data_dir}")

    all_publications = []
    if not wos_data_dir.is_dir():
        print(f"Error: Input directory not found: {wos_data_dir}")
    else:
        # --- File discovery and sorting ---
        all_txt_files_found = list(wos_data_dir.glob(file_pattern_glob))
        
        wos_files_to_process = []
        # Regex to match pattern like "savedrecs.txt" or "savedrecs (1).txt"
        # This assumes the base name before parenthesis is 'savedrecs'
        file_name_pattern_re = re.compile(r"^(savedrecs)(?: \((\d+)\))?\.txt$") # Specific to 'savedrecs' base
        
        files_with_num = []
        # Apply numeric sort only if the glob pattern suggests multiple 'savedrecs' files
        if '*' in file_pattern_glob or '?' in file_pattern_glob or '[' in file_pattern_glob: # Heuristic for glob
            for f_path in all_txt_files_found:
                match = file_name_pattern_re.match(f_path.name)
                if match:
                    base_name = match.group(1) # e.g., "savedrecs"
                    num_str = match.group(2) # The number in parentheses, if any
                    num = int(num_str) if num_str else 0 # No number in () means 0 for sorting
                    if base_name == "savedrecs": # Ensure it's the correct base filename for this sorting logic
                         files_with_num.append((num, f_path))
            
            if files_with_num: # If we found files matching the numbered 'savedrecs' pattern
                files_with_num.sort(key=lambda x: x[0]) # Sort by the extracted number
                wos_files_to_process = [f_path for num, f_path in files_with_num]
            else: # Fallback to simple alphanumeric sort if no numbered 'savedrecs' files found or different pattern
                wos_files_to_process = sorted(all_txt_files_found)
        else: # For single specific filename (no glob characters), just use it
            wos_files_to_process = all_txt_files_found
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
                # Call the network building function with the specified thresholds
                G_combined = build_cocitation_network(
                    all_publications,
                    min_node_citations_threshold=MIN_NODE_CITATIONS,
                    min_cocitation_strength_threshold=MIN_COCITATION_STRENGTH
                )
                # Save the resulting (potentially filtered) graph
                save_graph_to_graphml(G_combined, output_filepath)

    print("\nScript finished.")
