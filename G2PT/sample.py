"""
Sample from a trained model - AIG Generation Only Version
(Modified for Conditional Generation Alignment)
"""
import os
from contextlib import nullcontext
import torch
import numpy as np
from model import GPTConfig, GPT
from transformers import AutoTokenizer
# Ensure seq_to_nxgraph is available (it creates DiGraphs)
# Adjust import path if datasets_utils.py is not in the same directory
try:
    from datasets_utils import seq_to_nxgraph
except ImportError:
    print("Warning: Could not import 'seq_to_nxgraph' from 'datasets_utils'. Ensure the file is accessible.")
    # Define a dummy function or exit if essential
    def seq_to_nxgraph(seq_str, parsing_mode='strict'):
        print("Error: seq_to_nxgraph not imported!")
        return None
    # exit(1) # Uncomment to exit if seq_to_nxgraph is critical

import argparse
import pickle
import itertools
import networkx as nx # Keep nx import if seq_to_nxgraph returns nx objects

def parse_args():
    parser = argparse.ArgumentParser(description='Sample from a trained model for multiple PI/PO combinations')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory containing model checkpoint (e.g., results/aig-large3-topo)')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to tokenizer (e.g., tokenizers/aig)')
    # --- Modified PI/PO arguments ---
    parser.add_argument('--num_pis', type=int, nargs='+', required=True,
                        help='List of primary input counts for generated AIGs (e.g., --num_pis 2 3 4)')
    parser.add_argument('--num_pos', type=int, nargs='+', required=True,
                        help='List of primary output counts for generated AIGs (e.g., --num_pos 1 2)')
    # --- Modified num_samples argument ---
    parser.add_argument('--num_samples_per_combo', type=int, default=100, # Renamed and adjusted default
                        help='Number of samples to generate *for each* PI/PO combination')
    # --- End of modified arguments ---
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for generation within each combination')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (1.0 = standard)')
    parser.add_argument('--output_filename', type=str, default='generated_aigs_multi_cond.pkl', # Adjusted default name
                        help='Name for the output pickle file containing all graphs')
    parser.add_argument('--parsing_mode', type=str, default='strict', choices=['strict', 'robust'],
                        help='Edge sequence parsing mode: strict (fail on non-triplet length) or robust (skip malformed parts)')
    parser.add_argument('--max_length', type=int, default=None, help='Override max generation length (defaults to tokenizer.model_max_length)')


    return parser.parse_args()

def setup_device(seed):
    # (Setup remains the same as your original code)
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         device = 'mps'
    else:
         device = 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dtype = 'float32'
    print(f"Using dtype: {dtype}")
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    return device, ctx

def load_model(out_dir, device):
    # (Loading remains the same as your original code)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    print(f"Loading checkpoint from: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        print("Ensure --out_dir points to the correct directory containing ckpt.pt")
        exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)
    try:
        model_args = checkpoint['model_args']
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    except KeyError:
        print("Error: 'model_args' not found in checkpoint. Cannot recreate model.")
        exit(1)
    except Exception as e:
        print(f"Error recreating model from checkpoint args: {e}")
        exit(1)
    try:
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    except KeyError:
        print("Error: 'model' state dict not found in checkpoint.")
        exit(1)
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        exit(1)
    try:
        hf_model = model.to_hf()
    except Exception as e:
        print(f"Warning: Failed to convert model to Hugging Face format: {e}. Using original model.")
        hf_model = model
    hf_model.eval()
    hf_model.to(device)
    print(f"Model loaded successfully with {sum(p.numel() for p in hf_model.parameters())/1e6:.2f}M parameters.")
    return hf_model

# --- MODIFIED generate_sequences function ---
def generate_sequences(model, tokenizer, batch_size, num_samples, device,
                       pi_token_id: int, po_token_id: int, # Pass token IDs directly
                       temperature=1.0, max_length=None):

    # Create the initial prompt tensor (shape: [batch_size, 2])
    # Contains [[PI_ID, PO_ID], [PI_ID, PO_ID], ...]
    start_ids = torch.tensor([[pi_token_id, po_token_id]], dtype=torch.long, device=device)
    input_ids = start_ids.repeat(batch_size, 1) # Repeat for the batch

    # Create the corresponding attention mask (all 1s for the prompt)
    attention_mask = torch.ones_like(input_ids)

    print(f"Using initial prompt token IDs: [{pi_token_id}, {po_token_id}]")

    generated_sequences = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    print(f"Starting generation for {num_samples} samples in {num_batches} batches (batch size: {batch_size})...")

    # --- Get EOS token ID ---
    eos_token_id = tokenizer.eos_token_id
    eos_token_str = "<eog>" # Common end-of-graph token in this project
    if eos_token_id is None:
        try:
             eos_token_id = tokenizer.convert_tokens_to_ids(eos_token_str)
             print(f"Using '{eos_token_str}' as EOS token (ID: {eos_token_id}).")
        except KeyError:
             print(f"Warning: EOS token '{eos_token_str}' not found in tokenizer vocab. Generation might not terminate correctly.")
             eos_token_id = tokenizer.pad_token_id # Use pad as a less ideal fallback
    else:
         print(f"Using tokenizer's default EOS token (ID: {eos_token_id}). String: '{tokenizer.decode(eos_token_id)}'")


    # Determine max generation length
    effective_max_length = max_length if max_length is not None else tokenizer.model_max_length
    if effective_max_length is None:
         print("Warning: Could not determine max_length. Defaulting to 1024.")
         effective_max_length = 1024 # Set a reasonable default if tokenizer doesn't have it
    else:
         print(f"Max generation length set to: {effective_max_length}")


    for i in range(num_batches):
        print(f"Generating batch {i+1}/{num_batches}...")
        # Determine the number of samples needed for this batch
        current_batch_size = min(batch_size, num_samples - len(generated_sequences))
        if current_batch_size != input_ids.shape[0]:
             # Adjust batch size for the last batch if necessary
             batch_input_ids = input_ids[:current_batch_size]
             batch_attention_mask = attention_mask[:current_batch_size]
        else:
             batch_input_ids = input_ids
             batch_attention_mask = attention_mask

        with torch.no_grad(): # Ensure no gradients are calculated
            ids = model.generate(
                input_ids=batch_input_ids,         # Use the 2-token prompt
                attention_mask=batch_attention_mask, # Pass corresponding mask
                max_length=effective_max_length,   # Use determined max length
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id, # Use determined EOS token ID
                do_sample=True,
                temperature=temperature,
                # Add other generation parameters if needed (e.g., top_k, top_p)
            )
        # Decode, skipping special tokens *except* for structure tokens if needed for parsing
        # seq_to_nxgraph needs the structure tokens (<boc>, <eoc>, etc.)
        seq_strs = tokenizer.batch_decode(ids, skip_special_tokens=False)
        generated_sequences.extend(seq_strs)
        print(f"Batch {i+1} generated.")

    return generated_sequences[:num_samples]
# --- END OF MODIFIED FUNCTION ---


if __name__ == '__main__':
    args = parse_args()
    print("--- Starting AIG Sampling Script for Multiple Combinations (Aligned Conditional) ---")
    print(f"Arguments: {args}")

    # Ensure the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    device, ctx = setup_device(args.seed)

    # Load Tokenizer and Model
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        # Manually set pad token if not set (common issue)
        if tokenizer.pad_token is None:
             print("Tokenizer missing pad token, setting to UNK.")
             # Check if UNK exists, otherwise maybe EOS?
             unk_token = "[UNK]"
             eos_token = "<eog>" # Or tokenizer.eos_token if defined
             if unk_token in tokenizer.vocab:
                 tokenizer.pad_token = unk_token
             elif eos_token in tokenizer.vocab:
                 print("Using EOS as PAD token fallback.")
                 tokenizer.pad_token = eos_token
             else:
                 print("ERROR: Cannot set PAD token automatically. Please check tokenizer.")
                 exit(1)
             tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
             print(f"Using PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit(1)

    model = load_model(args.out_dir, device) # Loads the ckpt.pt from out_dir

    # --- Create PI/PO combinations ---
    pi_po_combinations = list(itertools.product(args.num_pis, args.num_pos))
    print(f"Found {len(pi_po_combinations)} PI/PO combinations to generate for:")
    print(pi_po_combinations)

    all_generated_graphs = [] # List to store graphs from all combinations
    total_sequences_generated = 0
    total_conversion_errors = 0

    # --- Loop over each combination ---
    for combo_idx, (num_pi, num_po) in enumerate(pi_po_combinations):
        print(f"\n--- Generating for Combination {combo_idx+1}/{len(pi_po_combinations)}: PI={num_pi}, PO={num_po} ---")

        # --- Construct the conditional tokens and get their IDs ---
        pi_token_str = f"PI_COUNT_{num_pi}"
        po_token_str = f"PO_COUNT_{num_po}"

        try:
             pi_token_id = tokenizer.convert_tokens_to_ids(pi_token_str)
             po_token_id = tokenizer.convert_tokens_to_ids(po_token_str)
             print(f"Found Token IDs: {pi_token_str}={pi_token_id}, {po_token_str}={po_token_id}")
        except KeyError as e:
            print(f"Warning: Token '{e}' not found in tokenizer vocab for PI={num_pi}, PO={num_po}. Skipping this combination.")
            continue # Skip to the next combination

        # --- Sequence Generation for this combination ---
        with ctx: # Use autocast context if on GPU
            generated_sequences_combo = generate_sequences(
                model,
                tokenizer,
                args.batch_size,
                args.num_samples_per_combo, # Use the per-combo count
                device,
                pi_token_id=pi_token_id,     # Pass PI token ID
                po_token_id=po_token_id,     # Pass PO token ID
                temperature=args.temperature,
                max_length=args.max_length   # Pass max_length override
            )
        print(f"Finished generating {len(generated_sequences_combo)} sequences for PI={num_pi}, PO={num_po}.")
        total_sequences_generated += len(generated_sequences_combo)

        # --- Convert Sequences to AIG DiGraphs for this combination ---
        print(f"Converting sequences for PI={num_pi}, PO={num_po}...")
        num_processed_combo = 0
        num_errors_combo = 0
        for i, seq_str in enumerate(generated_sequences_combo):
             # **Crucial Check:** Ensure the sequence starts correctly after generation
             # The model should have generated <boc> after the initial prompt tokens.
             # We need to decode the *full* sequence from model.generate()
             # and then potentially pass only the <boc>...<eog> part to seq_to_nxgraph if needed,
             # OR ensure seq_to_nxgraph can handle the prepended tokens (it should ignore them).
             # Based on seq_to_nxgraph implementation, it looks for <boc>, <eoc> etc.
             # so passing the full sequence string `seq_str` should be fine.

            try:
                # Make sure seq_to_nxgraph is properly imported or defined
                if seq_to_nxgraph is None:
                    print("Error: seq_to_nxgraph not available. Cannot convert sequences.")
                    num_errors_combo = len(generated_sequences_combo) # Mark all as errors
                    break # Exit inner loop

                graph = seq_to_nxgraph(seq_str, parsing_mode=args.parsing_mode)

                if isinstance(graph, nx.DiGraph): # Changed from nx.Graph to nx.DiGraph
                    # Optionally add PI/PO info to the graph object itself
                    # (This info is useful for later analysis/debugging)
                    graph.graph['target_num_pis'] = num_pi
                    graph.graph['target_num_pos'] = num_po
                    graph.graph['generated_sequence_sample'] = seq_str[:200] # Store prefix for debugging

                    all_generated_graphs.append(graph) # Add to the main list
                    num_processed_combo += 1
                else:
                    print(f"Warning: seq_to_nxgraph did not return a NetworkX DiGraph for sequence {i} (PI={num_pi}, PO={num_po}). Got {type(graph)}.")
                    print(f"Sequence sample: {seq_str[:150]}...") # Print sample on conversion failure
                    num_errors_combo += 1
            except Exception as e:
                print(f"Error processing sequence {i} (PI={num_pi}, PO={num_po}) to AIG: {e}")
                print(f"Sequence sample: {seq_str[:150]}...") # Print sample on error
                num_errors_combo += 1
        total_conversion_errors += num_errors_combo
        print(f"Successfully converted {num_processed_combo}/{len(generated_sequences_combo)} sequences for this combination.")

    # --- End of combination loop ---


    # --- Final Reporting ---
    print("\n" + "="*30)
    print("--- Overall AIG Generation Summary ---")
    print(f"Combinations processed    : {len(pi_po_combinations)}")
    print(f"Total sequences generated : {total_sequences_generated}")
    print(f"Total graphs converted    : {len(all_generated_graphs)}")
    print(f"Total conversion errors : {total_conversion_errors}")
    print("="*30 + "\n")

    # --- Saving Combined Results ---
    # Create output dir based on input dir name if needed
    save_dir = os.path.dirname(args.output_filename)
    if save_dir and not os.path.exists(os.path.join(args.out_dir, save_dir)):
         os.makedirs(os.path.join(args.out_dir, save_dir), exist_ok=True)

    output_file_path = os.path.join(args.out_dir, args.output_filename)
    print(f"Saving {len(all_generated_graphs)} generated AIG DiGraphs (from all combinations) to {output_file_path}")
    try:
        with open(output_file_path, 'wb') as f:
            # Save the list containing graphs from all combinations
            pickle.dump(all_generated_graphs, f)
    except Exception as e:
        print(f"Error saving combined pickle file: {e}")

    print("\n--- AIG Sampling Script Finished ---")