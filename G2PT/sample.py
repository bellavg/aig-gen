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
import networkx as nx # Keep nx import if seq_to_nxgraph returns nx objects

def parse_args():
    parser = argparse.ArgumentParser(description='Sample from a trained model')
    parser.add_argument('--out_dir', type=str, required=True, # Make required
                        help='Directory containing model checkpoint (e.g., results/aig-small-topo)')
    parser.add_argument('--tokenizer_path', type=str, required=True, # Make required
                        help='Path to tokenizer (e.g., tokenizers/aig)')
    parser.add_argument('--batch_size', type=int, default=256, # Adjusted default
                        help='Batch size for generation')
    parser.add_argument('--num_samples', type=int, default=1000, # Adjusted default
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (1.0 = standard)')
    parser.add_argument('--output_filename', type=str, default='generated_aigs.pkl',
                        help='Name for the output pickle file')
    parser.add_argument('--input_checkpoint', type=str, required=True,
                        help='Name/path for the input model checkpoint')
    parser.add_argument('--parsing_mode', type=str, default='robust', choices=['strict', 'robust'],
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

def load_model(ckpt_path, device):
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
    print("--- Starting AIG Sampling Script ---")
    print(f"Arguments: {args}")

    # Ensure the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    device, ctx = setup_device(args.seed)

    # Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args.input_checkpoint, device) # Loads the ckpt.pt from out_dir

    # --- Sequence Generation ---
    prefix = None # AIG usually starts from <boc>
    with ctx: # Use autocast context if on GPU
        generated_sequences = generate_sequences(
            model,
            tokenizer,
            args.batch_size,
            args.num_samples,
            device,
            prefix=prefix,
            temperature=args.temperature,
        )
    print(f"Finished generating {len(generated_sequences)} sequences.")

    # --- Convert Sequences to AIG DiGraphs ---
    print("Converting generated sequences to AIG DiGraphs...")
    generated_graphs = []
    num_processed = 0
    num_errors = 0

    for i, seq_str in enumerate(generated_sequences):
        try:
            # Convert sequence to graph using the function from datasets_utils
            graph = seq_to_nxgraph(seq_str, parsing_mode=args.parsing_mode)  #  # Should return nx.DiGraph
            # Basic check: Ensure it's a NetworkX graph object
            if isinstance(graph, nx.Graph): # Check base class (DiGraph inherits from Graph)
                generated_graphs.append(graph)
                num_processed += 1
            else:
                print(f"Warning: seq_to_nxgraph did not return a NetworkX graph for sequence {i}. Got {type(graph)}.")
                num_errors += 1
        except Exception as e:
            # Catch errors during seq_to_nxgraph conversion
            print(f"Error processing sequence {i} to AIG: {e}\nSequence sample: {seq_str[:150]}...")
            num_errors += 1

    # --- Reporting ---
    print("\n--- AIG Generation Summary ---")
    print(f"Total sequences generated   : {len(generated_sequences)}")
    print(f"Successfully converted    : {num_processed}")
    print(f"Errors during conversion  : {num_errors}")
    print("------------------------------")

    # --- Saving Results ---
    output_file_path = os.path.join(args.out_dir,"multinomial_sampling_"+args.output_filename)
    print(f"Saving {len(generated_graphs)} generated AIG DiGraphs to {output_file_path}")
    try:
        with open(output_file_path, 'wb') as f:
            pickle.dump(generated_graphs, f)
    except Exception as e:
        print(f"Error saving pickle file: {e}")

    print("\n--- AIG Sampling Script Finished ---")