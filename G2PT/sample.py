"""
Sample from a trained model - AIG Generation Only Version
"""
import os
from contextlib import nullcontext
import torch
import numpy as np
from model import GPTConfig, GPT
from transformers import AutoTokenizer
# Ensure seq_to_nxgraph is available (it creates DiGraphs)
from datasets_utils import seq_to_nxgraph
import argparse
import pickle
import itertools
import networkx as nx # Keep nx import if seq_to_nxgraph returns nx objects

def parse_args():
    parser = argparse.ArgumentParser(description='Sample from a trained model for multiple PI/PO combinations')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory containing model checkpoint (e.g., results/aig-small-topo)')
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
    parser.add_argument('--output_filename', type=str, default='generated_aigs_multi.pkl', # Adjusted default name
                        help='Name for the output pickle file containing all graphs')
    parser.add_argument('--parsing_mode', type=str, default='strict', choices=['strict', 'robust'],
                        help='Edge sequence parsing mode: strict (fail on non-triplet length) or robust (skip malformed parts)')

    return parser.parse_args()

def setup_device(seed):
    # Automatically detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         device = 'mps'
    else:
         device = 'cpu'
    print(f"Using device: {device}")

    # Determine appropriate dtype based on chosen device
    if device == 'cuda':
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        # Enable TF32 for CUDA acceleration if available
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

    # Load model configuration from checkpoint
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

    # Load model state dict
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

    # Convert to Hugging Face format if needed (original script did this, might impact generation slightly if removed)
    # Keeping it for consistency with original sample.py
    try:
        hf_model = model.to_hf()
    except Exception as e:
        print(f"Warning: Failed to convert model to Hugging Face format: {e}. Using original model.")
        hf_model = model # Fallback to original model object

    hf_model.eval()
    hf_model.to(device)
    print(f"Model loaded successfully with {sum(p.numel() for p in hf_model.parameters())/1e6:.2f}M parameters.")
    return hf_model

def generate_sequences(model, tokenizer, batch_size, num_samples, device, prefix=None, temperature=1.0):
    if prefix is None:
        # Standard start token for graph generation
        start_token = "<boc>"
        print(f"Using start token: {start_token}")
        inputs = tokenizer([start_token]*batch_size, return_tensors="pt")
    else:
        print(f"Using custom prefix: {prefix}")
        inputs = tokenizer([prefix]*batch_size, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    generated_sequences = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    print(f"Starting generation for {num_samples} samples in {num_batches} batches (batch size: {batch_size})...")

    # --- Get EOS token ID ---
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        # Fallback or specific token if EOS is not standard
        eos_token = "<eog>" # Common end-of-graph token in this project
        try:
             eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
             print(f"Using '{eos_token}' as EOS token (ID: {eos_token_id}).")
        except KeyError:
             print(f"Warning: EOS token '{eos_token}' not found in tokenizer vocab. Generation might not terminate correctly.")
             eos_token_id = tokenizer.pad_token_id # Use pad as a less ideal fallback

    for i in range(num_batches):
        print(f"Generating batch {i+1}/{num_batches}...")
        with torch.no_grad(): # Ensure no gradients are calculated
            ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=tokenizer.model_max_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id, # Use determined EOS token ID
                do_sample=True,
                temperature=temperature,
                # Add other generation parameters if needed (e.g., top_k, top_p)
            )
        seq_strs = tokenizer.batch_decode(ids, skip_special_tokens=False) # Keep special tokens for parsing
        generated_sequences.extend(seq_strs)
        print(f"Batch {i+1} generated.")

    return generated_sequences[:num_samples]


if __name__ == '__main__':
    args = parse_args()
    print("--- Starting AIG Sampling Script for Multiple Combinations ---")
    print(f"Arguments: {args}")

    # Ensure the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    device, ctx = setup_device(args.seed)

    # Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args.out_dir, device) # Loads the ckpt.pt from out_dir

    # --- Create PI/PO combinations ---
    pi_po_combinations = list(itertools.product(args.num_pis, args.num_pos))
    print(f"Found {len(pi_po_combinations)} PI/PO combinations to generate for:")
    print(pi_po_combinations)

    all_generated_graphs = [] # List to store graphs from all combinations
    total_sequences_generated = 0
    total_conversion_errors = 0
    start_token_str = "<boc>" # Start of the actual graph context

    # --- Loop over each combination ---
    for combo_idx, (num_pi, num_po) in enumerate(pi_po_combinations):
        print(f"\n--- Generating for Combination {combo_idx+1}/{len(pi_po_combinations)}: PI={num_pi}, PO={num_po} ---")

        # --- Construct the conditional prefix for this combination ---
        pi_token_str = f"PI_COUNT_{num_pi}"
        po_token_str = f"PO_COUNT_{num_po}"

        # Validate PI/PO tokens exist in tokenizer vocabulary
        if pi_token_str not in tokenizer.vocab:
            print(f"Warning: Token '{pi_token_str}' for PI={num_pi} not found in vocab. Skipping this combination.")
            continue # Skip to the next combination
        if po_token_str not in tokenizer.vocab:
            print(f"Warning: Token '{po_token_str}' for PO={num_po} not found in vocab. Skipping this combination.")
            continue # Skip to the next combination

        # Combine tokens to form the prefix for generation
        prefix = f"{pi_token_str} {po_token_str} {start_token_str}"
        print(f"Using generation prefix: '{prefix}'")

        # --- Sequence Generation for this combination ---
        with ctx: # Use autocast context if on GPU
            generated_sequences_combo = generate_sequences(
                model,
                tokenizer,
                args.batch_size,
                args.num_samples_per_combo, # Use the per-combo count
                device,
                prefix=prefix, # Pass the constructed prefix for this combo
                temperature=args.temperature,
            )
        print(f"Finished generating {len(generated_sequences_combo)} sequences for PI={num_pi}, PO={num_po}.")
        total_sequences_generated += len(generated_sequences_combo)

        # --- Convert Sequences to AIG DiGraphs for this combination ---
        print(f"Converting sequences for PI={num_pi}, PO={num_po}...")
        num_processed_combo = 0
        num_errors_combo = 0
        for i, seq_str in enumerate(generated_sequences_combo):
            try:
                graph = seq_to_nxgraph(seq_str, parsing_mode=args.parsing_mode)
                if isinstance(graph, nx.Graph):
                    # Optionally add PI/PO info to the graph object itself
                    graph.graph['num_pis'] = num_pi
                    graph.graph['num_pos'] = num_po
                    all_generated_graphs.append(graph) # Add to the main list
                    num_processed_combo += 1
                else:
                    print(f"Warning: seq_to_nxgraph did not return a NetworkX graph for sequence {i} (PI={num_pi}, PO={num_po}). Got {type(graph)}.")
                    num_errors_combo += 1
            except Exception as e:
                print(f"Error processing sequence {i} (PI={num_pi}, PO={num_po}) to AIG: {e}\nSequence sample: {seq_str[:150]}...")
                num_errors_combo += 1
        total_conversion_errors += num_errors_combo
        print(f"Converted {num_processed_combo}/{len(generated_sequences_combo)} sequences for this combination.")

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
    output_file_path = os.path.join(args.out_dir, args.output_filename)
    print(f"Saving {len(all_generated_graphs)} generated AIG DiGraphs (from all combinations) to {output_file_path}")
    try:
        with open(output_file_path, 'wb') as f:
            # Save the list containing graphs from all combinations
            pickle.dump(all_generated_graphs, f)
    except Exception as e:
        print(f"Error saving combined pickle file: {e}")

    print("\n--- AIG Sampling Script Finished ---")