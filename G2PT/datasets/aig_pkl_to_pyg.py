# G2PT/datasets/aig_pkl_to_pyg.py
# MODIFIED: To run ONLY Stage 2, assuming Stage 1 (creating temp files) is complete.
#           Loads existing temporary PyG files (temp_pyg_part_*.pt)
#           and saves the final split files (train.pt, train2.pt, val.pt).
# Stage 1 (PKL conversion) is SKIPPED.
# Uses configurations from G2PT/configs/aig.py

import os
import pickle
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm
import warnings
import sys
import gc # Import garbage collector interface
# import tempfile # No longer needed - directory should exist
# import shutil # No longer needed - directory should persist

# --- Configuration ---
# Import AIG specific configurations
try:
    # Try direct import first
    import G2PT.configs.aig as aig_cfg
    print("Successfully imported G2PT.configs.aig")
except ImportError:
    print("Could not import G2PT.configs.aig directly. Adjusting sys.path.")
    # Attempt to adjust path assuming standard project structure
    # Assumes this script is in G2PT/datasets/
    script_dir = os.path.dirname(os.path.realpath(__file__))
    g2pt_root = os.path.dirname(script_dir) # G2PT/
    if g2pt_root not in sys.path:
        sys.path.append(g2pt_root)
    try:
        import G2PT.configs.aig as aig_cfg
        print(f"Successfully imported G2PT.configs.aig after path adjustment.")
    except ImportError as e:
        print(f"Error importing G2PT.configs.aig even after path adjustment: {e}")
        print(f"Current sys.path: {sys.path}")
        print("Ensure the G2PT root directory is accessible via the Python path or installed.")
        sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during config import: {e}")
     sys.exit(1)


# --- Script-Specific Settings ---

# --- Define file indices for each FINAL output file ---
# These determine which temp files go into which final output.
TRAIN1_FILE_INDICES = list(range(1, 5)) # Parts 1-4 for train.pt
TRAIN2_FILE_INDICES = [5]                 # Part 5 for train2.pt
VAL_FILE_INDICES = [6]                  # Part 6 for val.pt
NUM_EXPECTED_PARTS = 6 # Total number of parts expected

# --- *** USER ACTION REQUIRED *** ---
# Set this variable to the *exact* path of the directory containing
# the existing temp_pyg_part_*.pt files from the previous run.
# Example: EXISTING_TEMP_DIR = "/Users/bellavg/aig-gen/G2PT/datasets/aig/processed/pyg_temp_parts_abc123"
EXISTING_TEMP_DIR = "/Users/bellavg/aig-gen/G2PT/datasets/aig/processed/pyg_temp_oghozzx_" # <-- SET THIS PATH
# ---

# Output Base Directory (Derived from aig_cfg) - Where raw/ will be created
try:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    g2pt_root = os.path.dirname(script_dir) # G2PT/
    OUTPUT_PYG_DIR = os.path.abspath(os.path.join(g2pt_root, os.path.normpath(aig_cfg.data_dir)))
    output_raw_dir = os.path.join(OUTPUT_PYG_DIR, 'raw')
    # Processed dir is where the EXISTING_TEMP_DIR should be located
    output_processed_dir = os.path.join(OUTPUT_PYG_DIR, 'processed')
except AttributeError:
     print("Error: Missing 'data_dir' attribute in imported G2PT.configs.aig.")
     sys.exit(1)
except Exception as e:
     print(f"Error determining output directory: {e}")
     sys.exit(1)

# --- Main Conversion Logic ---
if __name__ == '__main__':
    print(f"--- AIG PyG Conversion - Stage 2 ONLY (Combine Existing Temp Files) ---")
    print(f"--- Creates raw/train.pt (Parts 1-4), raw/train2.pt (Part 5), raw/val.pt (Part 6) ---")
    print(f"Using Configuration from G2PT.configs.aig")
    print(f"Expecting Temporary Files In: {EXISTING_TEMP_DIR}")
    print(f"Final Output Raw Dir : {output_raw_dir}")
    print(f"Train Set 1 (train.pt): Parts {TRAIN1_FILE_INDICES}")
    print(f"Train Set 2 (train2.pt): Part {TRAIN2_FILE_INDICES}")
    print(f"Validation Set (val.pt): Part {VAL_FILE_INDICES}")

    # --- Validate Existing Temp Dir ---
    if not os.path.isdir(EXISTING_TEMP_DIR):
        print(f"Error: Specified temporary directory does not exist: {EXISTING_TEMP_DIR}")
        print("Please ensure the path is correct and the temporary files from Stage 1 exist.")
        sys.exit(1)
    else:
        print(f"Found existing temporary directory: {EXISTING_TEMP_DIR}")

    # --- Create Final Output Directory ---
    try:
        os.makedirs(output_raw_dir, exist_ok=True)
        print(f"Ensured final output directory exists: {output_raw_dir}")
    except OSError as e:
        print(f"Error creating final output directory {output_raw_dir}: {e}")
        sys.exit(1)

    # --- Map part number to expected temporary file path ---
    temp_pyg_file_paths = {
        part_num: os.path.join(EXISTING_TEMP_DIR, f"temp_pyg_part_{part_num}.pt")
        for part_num in range(1, NUM_EXPECTED_PARTS + 1)
    }

    # ==============================================================
    # Stage 1: SKIPPED
    # ==============================================================
    print(f"\n--- Stage 1 (Converting PKL to Temp Files) SKIPPED ---")


    # ==================================================================
    # Stage 2: Load existing temporary files and save final split files
    # ==================================================================
    print(f"\n--- Stage 2: Loading existing temporary PyG files and saving final splits ---")
    graphs_loaded_count = 0

    # --- Process and Save train.pt (Parts 1-4) ---
    print("\nProcessing Train Set 1 (Parts 1-4)...")
    train1_data_list = []
    for part_num in TRAIN1_FILE_INDICES:
        temp_file_path = temp_pyg_file_paths.get(part_num)
        if temp_file_path and os.path.exists(temp_file_path):
            print(f"  -> Loading Part {part_num} from: {os.path.basename(temp_file_path)}")
            try:
                # Load the list of Data objects. Use weights_only=False as these are Data objects.
                current_part_data = torch.load(temp_file_path, weights_only=False)
                if isinstance(current_part_data, list):
                    loaded_count = len(current_part_data)
                    print(f"     Loaded {loaded_count} graphs.")
                    train1_data_list.extend(current_part_data)
                    graphs_loaded_count += loaded_count
                    del current_part_data # Free memory
                    gc.collect()
                else: print(f"     Warning: Expected list in {temp_file_path}, got {type(current_part_data)}. Skipping.")
            except Exception as e: print(f"     Error loading {temp_file_path}: {e}. Skipping.")
        else: print(f"  -> Skipping Part {part_num}: File not found at '{temp_file_path}'.")

    train1_save_path = os.path.join(output_raw_dir, 'train.pt')
    if train1_data_list:
        print(f"Saving Train Set 1 ({len(train1_data_list)} graphs) to {train1_save_path}...")
        try:
            torch.save(train1_data_list, train1_save_path, _use_new_zipfile_serialization=False)
            print("Save complete.")
        except Exception as e: print(f"Error saving {train1_save_path}: {e}")
        del train1_data_list # Free memory
        gc.collect()
    else: print("Train Set 1 is empty. Skipping save.")

    # --- Process and Save train2.pt (Part 5) ---
    print("\nProcessing Train Set 2 (Part 5)...")
    train2_data_list = []
    part_num = TRAIN2_FILE_INDICES[0] # Should be 5
    temp_file_path = temp_pyg_file_paths.get(part_num)
    if temp_file_path and os.path.exists(temp_file_path):
        print(f"  -> Loading Part {part_num} from: {os.path.basename(temp_file_path)}")
        try:
            current_part_data = torch.load(temp_file_path, weights_only=False)
            if isinstance(current_part_data, list):
                loaded_count = len(current_part_data)
                print(f"     Loaded {loaded_count} graphs.")
                train2_data_list.extend(current_part_data)
                graphs_loaded_count += loaded_count
                del current_part_data; gc.collect()
            else: print(f"     Warning: Expected list in {temp_file_path}, got {type(current_part_data)}. Skipping.")
        except Exception as e: print(f"     Error loading {temp_file_path}: {e}. Skipping.")
    else: print(f"  -> Skipping Part {part_num}: File not found at '{temp_file_path}'.")

    train2_save_path = os.path.join(output_raw_dir, 'train2.pt')
    if train2_data_list:
        print(f"Saving Train Set 2 ({len(train2_data_list)} graphs) to {train2_save_path}...")
        try:
            torch.save(train2_data_list, train2_save_path, _use_new_zipfile_serialization=False)
            print("Save complete.")
        except Exception as e: print(f"Error saving {train2_save_path}: {e}")
        del train2_data_list; gc.collect()
    else: print("Train Set 2 is empty. Skipping save.")

    # --- Process and Save val.pt (Part 6) ---
    print("\nProcessing Validation Set (Part 6)...")
    val_data_list = []
    part_num = VAL_FILE_INDICES[0] # Should be 6
    temp_file_path = temp_pyg_file_paths.get(part_num)
    if temp_file_path and os.path.exists(temp_file_path):
        print(f"  -> Loading Part {part_num} from: {os.path.basename(temp_file_path)}")
        try:
            current_part_data = torch.load(temp_file_path, weights_only=False)
            if isinstance(current_part_data, list):
                loaded_count = len(current_part_data)
                print(f"     Loaded {loaded_count} graphs.")
                val_data_list.extend(current_part_data)
                graphs_loaded_count += loaded_count
                del current_part_data; gc.collect()
            else: print(f"     Warning: Expected list in {temp_file_path}, got {type(current_part_data)}. Skipping.")
        except Exception as e: print(f"     Error loading {temp_file_path}: {e}. Skipping.")
    else: print(f"  -> Skipping Part {part_num}: File not found at '{temp_file_path}'.")

    val_save_path = os.path.join(output_raw_dir, 'val.pt')
    if val_data_list:
        print(f"Saving Validation Set ({len(val_data_list)} graphs) to {val_save_path}...")
        try:
            torch.save(val_data_list, val_save_path, _use_new_zipfile_serialization=False)
            print("Save complete.")
        except Exception as e: print(f"Error saving {val_save_path}: {e}")
        del val_data_list; gc.collect()
    else: print("Validation Set is empty. Skipping save.")


    # --- Print Final Summary Statistics ---
    print("\n--- Overall Summary (Stage 2 Only) ---")
    print(f"Total PyG graphs loaded from temp files: {graphs_loaded_count}")
    # Final counts depend on successful saving
    print("-" * 40)

    # --- Clean Up Temporary Files ---
    # print(f"\nCleaning up temporary directory: {EXISTING_TEMP_DIR}") # No longer cleaning up temp dir
    # try:
    #     shutil.rmtree(EXISTING_TEMP_DIR)
    #     print(f"Successfully removed temporary directory.")
    # except Exception as e:
    #     print(f"Error removing temporary directory {EXISTING_TEMP_DIR}: {e}")
    #     print("You may need to remove it manually.")
    print(f"\n--- Temporary directory NOT removed ---")
    print(f"Kept temporary files in: {EXISTING_TEMP_DIR}")


    print("\n--- Script finished (Stage 2 Only). ---")
    print(f"Final PyG dataset files saved in: {output_raw_dir}")
    print("Files created: train.pt, train2.pt, val.pt (if corresponding parts were processed successfully)")
    print("Next step: Modify AIGPygDataset/DataModule to handle train.pt and train2.pt as the training set.")

