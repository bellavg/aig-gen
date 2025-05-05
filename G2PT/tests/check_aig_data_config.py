# G2PT/configs/aig_assertions.py
import math
# Import all variables from aig.py (assuming it's in the same directory)
# Note: This pollutes the namespace; passing args is often cleaner, but following user request.
from G2PT.configs.aig import * # Use relative import assuming files are in the same 'configs' directory

#PASSED

# G2PT/tests/check_config_tokenizer_json.py
import os
import sys
import json

# --- Add G2PT root to sys.path ---
# Assuming this script is run from G2PT/tests/ or G2PT/
script_dir = os.path.dirname(os.path.realpath(__file__))
g2pt_root = script_dir if os.path.basename(script_dir) == 'G2PT' else os.path.dirname(script_dir)
if g2pt_root not in sys.path:
    sys.path.insert(0, g2pt_root)
    print(f"Added '{g2pt_root}' to sys.path for importing config.")

# --- Import Config ---
try:
    # Import all variables from the config file
    from G2PT.configs.aig import *
    print("Successfully imported configuration from G2PT.configs.aig")
except ImportError as e:
    print(f"Error importing G2PT.configs.aig: {e}")
    print("Please ensure the script is run from the G2PT project root or G2PT/tests directory,")
    print("or that the G2PT directory is in your PYTHONPATH.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during config import: {e}")
    sys.exit(1)

# --- Define Tokenizer Path ---
# Use the path provided by the user
TOKENIZER_DIR = "/Users/bellavg/aig-gen/G2PT/datasets/aig/tokenizer"
TOKENIZER_JSON_PATH = os.path.join(TOKENIZER_DIR, "tokenizer.json")

def check_vocab_consistency_from_json():
    """
    Compares vocab definitions in config vs. tokenizer.json file content.
    Does NOT load the tokenizer object via transformers library.
    """
    print(f"\n--- Starting Vocabulary Consistency Check (JSON only) ---")
    print(f"Config File : G2PT/configs/aig.py")
    print(f"Tokenizer File: {TOKENIZER_JSON_PATH}")

    errors_found = 0

    # --- Load tokenizer.json ---
    try:
        print(f"\nLoading {TOKENIZER_JSON_PATH} for detailed inspection...")
        if not os.path.exists(TOKENIZER_JSON_PATH):
             raise FileNotFoundError(f"{TOKENIZER_JSON_PATH} not found.")

        with open(TOKENIZER_JSON_PATH, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        # Extract vocab and added tokens directly from the JSON data
        tokenizer_model_vocab = tokenizer_data.get('model', {}).get('vocab', {})
        tokenizer_added_tokens = tokenizer_data.get('added_tokens', [])

        if not tokenizer_model_vocab:
             print("  WARNING: 'model.vocab' field not found or empty in tokenizer.json.")
             # Set to empty dict to avoid errors later, but checks will likely fail
             tokenizer_model_vocab = {}

        # Create a map from token content to ID for added tokens for easier lookup
        tokenizer_added_map = {token['content']: token['id'] for token in tokenizer_added_tokens}
        print(f"tokenizer.json loaded. Model vocab size: {len(tokenizer_model_vocab)}, Added special tokens: {len(tokenizer_added_tokens)}")

    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        errors_found += 1
        return errors_found # Cannot proceed
    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse {TOKENIZER_JSON_PATH}: {e}")
        errors_found += 1
        return errors_found
    except Exception as e:
        print(f"  ERROR: Unexpected error loading {TOKENIZER_JSON_PATH}: {e}")
        errors_found += 1
        return errors_found

    # --- 1. Compare Config FULL_VOCAB with Tokenizer Model Vocab ---
    print("\n1. Comparing Config `FULL_VOCAB` with tokenizer.json Model Vocab...")
    config_full_vocab = FULL_VOCAB # From imported config
    mismatched_ids = []
    missing_in_tokenizer = []
    extra_in_tokenizer = []

    # Check tokens defined in config exist in tokenizer JSON with correct ID
    for token, config_id in config_full_vocab.items():
        tokenizer_id = tokenizer_model_vocab.get(token)
        if tokenizer_id is None:
            missing_in_tokenizer.append(token)
            errors_found += 1
        elif tokenizer_id != config_id:
            mismatched_ids.append(f"'{token}': Config ID {config_id}, Tokenizer JSON ID {tokenizer_id}")
            errors_found += 1

    # Check tokens defined in tokenizer JSON exist in config with correct ID
    for token, tokenizer_id in tokenizer_model_vocab.items():
        config_id = config_full_vocab.get(token)
        if config_id is None:
            extra_in_tokenizer.append(token)
            errors_found += 1
        # Mismatch already checked above

    if not mismatched_ids and not missing_in_tokenizer and not extra_in_tokenizer:
        print("  PASS: Config `FULL_VOCAB` and tokenizer.json model vocab match perfectly.")
    else:
        if mismatched_ids:
            print(f"  ERROR: ID Mismatches found ({len(mismatched_ids)}):")
            for mismatch in mismatched_ids[:10]: print(f"    - {mismatch}") # Print first few
            if len(mismatched_ids) > 10: print("    ...")
        if missing_in_tokenizer:
            print(f"  ERROR: Tokens in Config `FULL_VOCAB` but MISSING from Tokenizer JSON Model Vocab ({len(missing_in_tokenizer)}):")
            for token in missing_in_tokenizer[:10]: print(f"    - '{token}'")
            if len(missing_in_tokenizer) > 10: print("    ...")
        if extra_in_tokenizer:
            print(f"  ERROR: Tokens in Tokenizer JSON Model Vocab but MISSING from Config `FULL_VOCAB` ({len(extra_in_tokenizer)}):")
            for token in extra_in_tokenizer[:10]: print(f"    - '{token}'")
            if len(extra_in_tokenizer) > 10: print("    ...")

    # --- 2. Compare Config SPECIAL_TOKENS with Tokenizer Added Tokens ---
    print("\n2. Comparing Config `SPECIAL_TOKENS` with tokenizer.json Added Tokens...")
    config_special_tokens = SPECIAL_TOKENS # From imported config
    mismatched_special_ids = []
    missing_in_tokenizer_added = []
    extra_in_tokenizer_added = []

    # Check special tokens defined in config exist in tokenizer added tokens
    for token, config_id in config_special_tokens.items():
        tokenizer_id = tokenizer_added_map.get(token)
        if tokenizer_id is None:
            missing_in_tokenizer_added.append(token)
            errors_found += 1
        elif tokenizer_id != config_id:
            mismatched_special_ids.append(f"'{token}': Config ID {config_id}, Tokenizer Added ID {tokenizer_id}")
            errors_found += 1
        # Optional: Check if 'special' flag is True in tokenizer.json for this token
        found_token_data = next((t for t in tokenizer_added_tokens if t['content'] == token), None)
        if found_token_data and not found_token_data.get('special', False):
             print(f"  WARNING: Token '{token}' found in added_tokens but 'special' flag is not True.")

    # Check added tokens defined in tokenizer exist in config special tokens
    for token, tokenizer_id in tokenizer_added_map.items():
        config_id = config_special_tokens.get(token)
        if config_id is None:
            # Check if it's just the default UNK token from the model section
            unk_token_from_model = tokenizer_data.get('model', {}).get('unk_token')
            if token == unk_token_from_model and config_id is None:
                 print(f"  INFO: Tokenizer added token '{token}' seems to be the model's UNK token, which is expected not to be in Config `SPECIAL_TOKENS`.")
            else:
                extra_in_tokenizer_added.append(token)
                errors_found += 1
        # Mismatch already checked above

    if not mismatched_special_ids and not missing_in_tokenizer_added and not extra_in_tokenizer_added:
        print("  PASS: Config `SPECIAL_TOKENS` and tokenizer.json added tokens match.")
    else:
        if mismatched_special_ids:
            print(f"  ERROR: Special Token ID Mismatches found ({len(mismatched_special_ids)}):")
            for mismatch in mismatched_special_ids: print(f"    - {mismatch}")
        if missing_in_tokenizer_added:
            print(f"  ERROR: Tokens in Config `SPECIAL_TOKENS` but MISSING from Tokenizer Added Tokens ({len(missing_in_tokenizer_added)}):")
            for token in missing_in_tokenizer_added: print(f"    - '{token}'")
        if extra_in_tokenizer_added:
            print(f"  ERROR: Tokens in Tokenizer Added Tokens but MISSING from Config `SPECIAL_TOKENS` ({len(extra_in_tokenizer_added)}):")
            for token in extra_in_tokenizer_added: print(f"    - '{token}'")

    # --- 3. Compare Overall Vocab Size ---
    print("\n3. Comparing Overall Vocab Size...")
    config_vocab_size = vocab_size # From imported config (expected size)
    print(f"  Config `vocab_size` (derived): {config_vocab_size}")

    # Calculate expected size based *only* on tokenizer.json content
    all_tokenizer_ids = set(tokenizer_model_vocab.values()) | set(tokenizer_added_map.values())
    if not all_tokenizer_ids:
         highest_tokenizer_id = -1
         print("  WARNING: No token IDs found in tokenizer.json model vocab or added tokens.")
    else:
         highest_tokenizer_id = max(all_tokenizer_ids)

    expected_tokenizer_size_from_json = highest_tokenizer_id + 1
    print(f"  Highest token ID found in tokenizer.json (model + added): {highest_tokenizer_id}")
    print(f"  Expected Tokenizer Size (based on highest ID + 1): {expected_tokenizer_size_from_json}")

    if config_vocab_size == expected_tokenizer_size_from_json:
        print("  PASS: Config `vocab_size` matches expected size based on highest ID in tokenizer.json.")
    else:
        print(f"  ERROR: Config `vocab_size` ({config_vocab_size}) does NOT match expected size ({expected_tokenizer_size_from_json}) based on highest ID in tokenizer.json.")
        errors_found += 1

    return errors_found

# --- Run the Check ---
if __name__ == "__main__":
    num_errors = check_vocab_consistency_from_json()
    print(f"\n--- Check Complete ---")
    if num_errors == 0:
        print("✅ All vocabulary consistency checks passed!")
    else:
        print(f"❌ Found {num_errors} vocabulary consistency errors.")
        sys.exit(1) # Exit with error code if discrepancies found

def check_aig_config():
    """
    Runs consistency checks on the imported AIG configuration variables.
    Relies on 'from .aig import *' having been executed.
    Uses uppercase constants imported from aig.py.
    Raises AssertionError if any check fails.
    """
    # --- Basic Type/Value Checks ---
    assert isinstance(MAX_NODE_COUNT, int) and MAX_NODE_COUNT > 0, "MAX_NODE_COUNT must be a positive integer."
    assert isinstance(MIN_PI_COUNT, int) and MIN_PI_COUNT >= 0, "MIN_PI_COUNT must be a non-negative integer."
    assert isinstance(MAX_PI_COUNT, int) and MAX_PI_COUNT >= MIN_PI_COUNT, "MAX_PI_COUNT must be >= MIN_PI_COUNT."
    assert isinstance(MIN_PO_COUNT, int) and MIN_PO_COUNT >= 1, "MIN_PO_COUNT must be at least 1."
    assert isinstance(MAX_PO_COUNT, int) and MAX_PO_COUNT >= MIN_PO_COUNT, "MAX_PO_COUNT must be >= MIN_PO_COUNT."
    assert isinstance(MIN_AND_COUNT, int) and MIN_AND_COUNT >= 0, "MIN_AND_COUNT must be non-negative." # Adjust if >= 1 needed
    assert isinstance(block_size, int) and block_size > 0, "block_size must be a positive integer."
    assert isinstance(PAD_VALUE, int), "PAD_VALUE (for loss) must be an integer."

    # --- Feature Count and Vocab Size Checks ---
    # Check derived feature counts against derived vocabs and base key lists
    assert NUM_NODE_FEATURES == len(NODE_TYPE_VOCAB), \
        "Derived NUM_NODE_FEATURES should match the size of derived NODE_TYPE_VOCAB."
    assert NUM_EDGE_FEATURES == len(EDGE_TYPE_VOCAB), \
        "Derived NUM_EDGE_FEATURES should match the size of derived EDGE_TYPE_VOCAB."
    # *** Use uppercase constant names imported via * ***
    assert NUM_NODE_FEATURES == len(NODE_TYPE_KEYS), \
        "NUM_NODE_FEATURES should match the number of base node type keys defined (NODE_TYPE_KEYS)."
    assert NUM_EDGE_FEATURES == len(EDGE_TYPE_KEYS), \
        "NUM_EDGE_FEATURES should match the number of base edge type keys defined (EDGE_TYPE_KEYS)."

    # --- Offset and Value Checks (ensure derived correctly) ---
    # Check that the derived offsets match the minimum values in the specific vocabs
    if NODE_TYPE_VOCAB:
        assert NODE_VOCAB_OFFSET == min(NODE_TYPE_VOCAB.values()), \
            "Derived NODE_VOCAB_OFFSET should be the minimum value in NODE_TYPE_VOCAB."
    if EDGE_TYPE_VOCAB:
        assert EDGE_VOCAB_OFFSET == min(EDGE_TYPE_VOCAB.values()), \
            "Derived EDGE_VOCAB_OFFSET should be the minimum value in EDGE_TYPE_VOCAB."

    # Check for sequential values within NODE_TYPE_VOCAB and EDGE_TYPE_VOCAB specifically
    if NODE_TYPE_VOCAB:
        node_values = sorted(NODE_TYPE_VOCAB.values())
        assert node_values == list(range(min(node_values), max(node_values) + 1)), \
            "NODE_TYPE_VOCAB values should be sequential."
    if EDGE_TYPE_VOCAB:
        edge_values = sorted(EDGE_TYPE_VOCAB.values())
        assert edge_values == list(range(min(edge_values), max(edge_values) + 1)), \
            "EDGE_TYPE_VOCAB values should be sequential."


    # --- Encoding Consistency ---
    # Check encoding dimensions against feature counts
    # *** Use uppercase constant names imported via * ***
    if NODE_TYPE_ENCODING and NODE_TYPE_KEYS:
         first_node_key = NODE_TYPE_KEYS[0]
         assert first_node_key in NODE_TYPE_ENCODING, f"First node key '{first_node_key}' not found in NODE_TYPE_ENCODING."
         assert NUM_NODE_FEATURES == len(NODE_TYPE_ENCODING[first_node_key]), \
             f"Length of node encodings ({len(NODE_TYPE_ENCODING[first_node_key])}) must match NUM_NODE_FEATURES ({NUM_NODE_FEATURES})."
    if EDGE_LABEL_ENCODING and EDGE_TYPE_KEYS:
        first_edge_key = EDGE_TYPE_KEYS[0]
        assert first_edge_key in EDGE_LABEL_ENCODING, f"First edge key '{first_edge_key}' not found in EDGE_LABEL_ENCODING."
        assert NUM_EDGE_FEATURES == len(EDGE_LABEL_ENCODING[first_edge_key]), \
             f"Length of edge encodings ({len(EDGE_LABEL_ENCODING[first_edge_key])}) must match NUM_EDGE_FEATURES ({NUM_EDGE_FEATURES})."
    # Check if encoding keys match derived vocab keys (important for lookup consistency)
    assert set(NODE_TYPE_ENCODING.keys()) == set(NODE_TYPE_VOCAB.keys()), \
        "Keys in NODE_TYPE_ENCODING must match keys in derived NODE_TYPE_VOCAB."
    assert set(EDGE_LABEL_ENCODING.keys()) == set(EDGE_TYPE_VOCAB.keys()), \
        "Keys in EDGE_LABEL_ENCODING must match keys in derived EDGE_TYPE_VOCAB."


    # --- Feature Index Mapping Checks ---
    # Check that the index mappings cover the correct range [0, NUM_FEATURES-1]
    assert set(NODE_FEATURE_INDEX_TO_VOCAB.keys()) == set(range(NUM_NODE_FEATURES)), \
        "Indices (keys) in NODE_FEATURE_INDEX_TO_VOCAB should range from 0 to NUM_NODE_FEATURES - 1."
    assert set(EDGE_FEATURE_INDEX_TO_VOCAB.keys()) == set(range(NUM_EDGE_FEATURES)), \
        "Indices (keys) in EDGE_FEATURE_INDEX_TO_VOCAB should range from 0 to NUM_EDGE_FEATURES - 1."
    # Check if values map correctly based on the assumed order from NODE_TYPE_KEYS/EDGE_TYPE_KEYS
    # *** Use uppercase constant names imported via * ***
    if NODE_TYPE_KEYS:
        assert NODE_FEATURE_INDEX_TO_VOCAB[0] == NODE_TYPE_VOCAB[NODE_TYPE_KEYS[0]], "NODE_FEATURE_INDEX_TO_VOCAB[0] mismatch."
        assert NODE_FEATURE_INDEX_TO_VOCAB[NUM_NODE_FEATURES-1] == NODE_TYPE_VOCAB[NODE_TYPE_KEYS[-1]], "NODE_FEATURE_INDEX_TO_VOCAB last element mismatch."
    if EDGE_TYPE_KEYS:
        assert EDGE_FEATURE_INDEX_TO_VOCAB[0] == EDGE_TYPE_VOCAB[EDGE_TYPE_KEYS[0]], "EDGE_FEATURE_INDEX_TO_VOCAB[0] mismatch."
        assert EDGE_FEATURE_INDEX_TO_VOCAB[NUM_EDGE_FEATURES-1] == EDGE_TYPE_VOCAB[EDGE_TYPE_KEYS[-1]], "EDGE_FEATURE_INDEX_TO_VOCAB last element mismatch."

    # --- Valid Types Set Checks Removed ---
    # VALID_AIG_NODE_TYPES / VALID_AIG_EDGE_TYPES no longer defined in config.

    # --- AIG Constraint Sanity Checks ---
    # Check if MAX_NODE_COUNT is reasonably larger than minimums combined
    assert MAX_NODE_COUNT >= (MIN_PI_COUNT + MIN_PO_COUNT + MIN_AND_COUNT), \
         "MAX_NODE_COUNT must be >= sum of minimum PI, PO, and AND counts."

    # --- FULL_VOCAB Assertions (assuming generated correctly) ---
    assert isinstance(FULL_VOCAB, dict), "FULL_VOCAB must be a dictionary."
    if not FULL_VOCAB:
        print("Warning: FULL_VOCAB is empty, skipping related checks.")
    else:
        actual_full_vocab_size = len(FULL_VOCAB)
        # Use MAX_FULL_VOCAB_ID constant imported via *
        max_full_vocab_value = MAX_FULL_VOCAB_ID

        # Check value uniqueness and range within FULL_VOCAB itself
        assert len(set(FULL_VOCAB.values())) == actual_full_vocab_size, "All values in generated FULL_VOCAB must be unique."
        assert min(FULL_VOCAB.values()) == 0, "Generated FULL_VOCAB values should start from 0."
        # Check if the values form a continuous sequence from 0 to N-1
        assert sorted(FULL_VOCAB.values()) == list(range(actual_full_vocab_size)), \
            "Generated FULL_VOCAB values should be sequential integers starting from 0."

        # --- Check derived contents based on primary constants ---
        # Check inclusion of NODE and EDGE types
        for k, v in NODE_TYPE_VOCAB.items():
            assert k in FULL_VOCAB and FULL_VOCAB[k] == v, f"Node type '{k}' with value {v} not found or mismatched in FULL_VOCAB."
        for k, v in EDGE_TYPE_VOCAB.items():
            assert k in FULL_VOCAB and FULL_VOCAB[k] == v, f"Edge type '{k}' with value {v} not found or mismatched in FULL_VOCAB."

        # IDX tokens: Check count and bounds
        expected_idx_count = MAX_NODE_COUNT
        actual_idx_count = sum(1 for k in FULL_VOCAB if k.startswith("IDX_"))
        assert actual_idx_count == expected_idx_count, f"Expected {expected_idx_count} IDX tokens, found {actual_idx_count}."
        assert f"IDX_{MAX_NODE_COUNT - 1}" in FULL_VOCAB, "Highest expected IDX token (IDX_{MAX_NODE_COUNT - 1}) not found."
        assert f"IDX_{MAX_NODE_COUNT}" not in FULL_VOCAB, "IDX token for MAX_NODE_COUNT should not exist (0-based indexing)."

        # PI/PO Count tokens: Check count and bounds
        expected_pi_count_tokens = MAX_PI_COUNT - MIN_PI_COUNT + 1
        actual_pi_count = sum(1 for k in FULL_VOCAB if k.startswith("PI_COUNT_"))
        assert actual_pi_count == expected_pi_count_tokens, f"Mismatch in PI_COUNT token count. Expected {expected_pi_count_tokens}, found {actual_pi_count}."
        assert f"PI_COUNT_{MAX_PI_COUNT}" in FULL_VOCAB, "Highest expected PI_COUNT token not found."
        assert f"PI_COUNT_{MIN_PI_COUNT}" in FULL_VOCAB, "Lowest expected PI_COUNT token not found."


        expected_po_count_tokens = MAX_PO_COUNT - MIN_PO_COUNT + 1
        actual_po_count = sum(1 for k in FULL_VOCAB if k.startswith("PO_COUNT_"))
        assert actual_po_count == expected_po_count_tokens, f"Mismatch in PO_COUNT token count. Expected {expected_po_count_tokens}, found {actual_po_count}."
        assert f"PO_COUNT_{MAX_PO_COUNT}" in FULL_VOCAB, "Highest expected PO_COUNT token not found."
        assert f"PO_COUNT_{MIN_PO_COUNT}" in FULL_VOCAB, "Lowest expected PO_COUNT token not found."

    # --- Special Token Assertions ---
    assert isinstance(SPECIAL_TOKENS, dict), "SPECIAL_TOKENS must be a dictionary."
    special_token_ids = set(SPECIAL_TOKENS.values())
    assert len(special_token_ids) == len(SPECIAL_TOKENS), "Special token IDs must be unique within SPECIAL_TOKENS."

    # Check that special token IDs are distinct from the main vocabulary
    if FULL_VOCAB and SPECIAL_TOKENS:
        # Use MAX_FULL_VOCAB_ID constant imported via *
        assert min(special_token_ids) > MAX_FULL_VOCAB_ID, \
            f"Special token IDs (min: {min(special_token_ids)}) must be greater than the max value in FULL_VOCAB ({MAX_FULL_VOCAB_ID})."
        # Ensure the specific PAD_TOKEN_ID from aig.py matches the value in the SPECIAL_TOKENS dict
        assert PAD_TOKEN_ID == SPECIAL_TOKENS.get("[PAD]", None), \
             "PAD_TOKEN_ID derived in config differs from '[PAD]' value in SPECIAL_TOKENS dict."
        # Check all special tokens are outside the main vocab range
        assert not (special_token_ids & set(FULL_VOCAB.values())), \
            "Special token IDs must not overlap with values in FULL_VOCAB."

    # Check PAD_VALUE (for loss) vs PAD_TOKEN_ID (for sequence padding)
    assert isinstance(PAD_VALUE, int) and PAD_VALUE < 0, \
        "PAD_VALUE (for loss masking) should be a distinct negative integer (e.g., -100)."
    assert PAD_VALUE not in FULL_VOCAB.values(), "PAD_VALUE (loss) must not be in FULL_VOCAB values."
    assert PAD_VALUE not in special_token_ids, "PAD_VALUE (loss) must not be one of the special token IDs."
    # Crucially, ensure the loss padding value is different from the sequence padding token ID
    assert PAD_VALUE != PAD_TOKEN_ID, \
        f"PAD_VALUE ({PAD_VALUE}) must be different from PAD_TOKEN_ID ({PAD_TOKEN_ID})."

    # --- Overall vocab_size check ---
    # Calculate the expected size based on the highest ID found + 1
    # Use OVERALL_MAX_ID constant imported via *
    expected_vocab_size = OVERALL_MAX_ID + 1
    assert vocab_size == expected_vocab_size, \
        f"Derived vocab_size ({vocab_size}) must be equal to the highest defined token ID + 1 ({expected_vocab_size})."

    # Block size check - basic heuristic
    # A more precise check would require knowing the exact sequence format
    min_tokens_per_node = 2 # E.g., node type + node index
    estimated_min_seq_len = MAX_NODE_COUNT * min_tokens_per_node + 10 # Add buffer for structure/edge tokens
    assert block_size >= estimated_min_seq_len, \
        f"block_size ({block_size}) might be too small. Estimated minimum length for MAX_NODE_COUNT={MAX_NODE_COUNT} is roughly {estimated_min_seq_len}."

# --- Optional: Allow running checks directly ---
if __name__ == "__main__":
    try:
        check_aig_config()
        print("AIG configuration consistency checks passed.")
    except AssertionError as e:
        print(f"AIG configuration consistency check FAILED:\n{e}")
    except ImportError:
        print("Could not import 'aig' config. Make sure aig.py is in the same directory or accessible via PYTHONPATH.")
    except NameError as e:
        print(f"NameError during checks: {e}. Ensure 'from .aig import *' imports all necessary uppercase constants.")

