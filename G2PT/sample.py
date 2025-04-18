"""
Sample from a trained model
"""
import os
from contextlib import nullcontext
import torch
import numpy as np
from model import GPTConfig, GPT
from transformers import AutoTokenizer
from datasets_utils import  seq_to_nxgraph
import argparse
from contextlib import nullcontext
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Sample from a trained model')
    parser.add_argument('--out_dir', type=str, default='results/moses-small-bfs',
                        help='Directory containing model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizers/moses',
                        help='Path to tokenizer') 
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for generation')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    return parser.parse_args()

def setup_device(seed):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    return device, ctx

def load_model(out_dir, device):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    hf_model = model.to_hf()
    hf_model.eval()
    hf_model.to(device)
    
    return hf_model

def generate_sequences(model, tokenizer, batch_size, num_samples, device, prefix=None, temperature=1.0):
    if prefix is None:
        inputs = tokenizer(['<boc>']*batch_size, return_tensors="pt")
    else:
        inputs = tokenizer([prefix]*batch_size, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    generated_sequences = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    for _ in range(num_batches):
        ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=tokenizer.model_max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
        )
        seq_strs = tokenizer.batch_decode(ids)
        generated_sequences.extend(seq_strs)
        
    return generated_sequences[:num_samples]


if __name__ == '__main__':
    args = parse_args()
    device, ctx = setup_device(args.seed)

    # Load the correct AIG tokenizer and the trained AIG model
    # Ensure args.tokenizer_path points to 'tokenizers/aig/'
    # Ensure args.out_dir points to your trained AIG model checkpoint directory
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args.out_dir, device)

    # Determine dataset type based on tokenizer path
    dataset_type = 'unknown'
    if 'aig' in args.tokenizer_path:
        dataset_type = 'aig'
    elif any(name in args.tokenizer_path for name in ['guacamol', 'qm9', 'moses']):
        dataset_type = 'molecular'
    elif any(name in args.tokenizer_path for name in ['planar', 'sbm', 'tree', 'lobster']):
        dataset_type = 'general_graph'

    # --- Sequence Generation (Common) ---
    prefix = None # AIG usually starts from <boc>
    temperature = 1.0 # Standard temperature unless specific tuning needed
    # Add specific prefix/temperature logic here if needed for AIG, similar to 'planar' example

    with ctx:
        generated_sequences = generate_sequences(
            model,
            tokenizer,
            args.batch_size,
            args.num_samples,
            device,
            prefix=prefix,
            temperature=temperature,
        )

    # --- Post-processing based on dataset type ---
    if dataset_type == 'molecular':
        # --- Keep existing molecular processing logic ---
        # save smiles
        smiles = []
        # Import molecule processing functions if needed
        from datasets_utils import seq_to_mol, get_smiles, seq_to_molecule_with_partial_charges
        for seq_str in generated_sequences:
            try:
                if 'guacamol' in args.tokenizer_path:
                    mol = seq_to_molecule_with_partial_charges(seq_str)
                else:
                    mol = seq_to_mol(seq_str)
                smile = get_smiles(mol)
                if smile:
                    smiles.append(smile)
                else:
                    smiles.append(None) # Keep track of failed conversions
            except Exception as e:
                print(f"Error processing sequence to SMILES: {e}")
                smiles.append(None) # Add None for errors
        smiles_out = [str(s) if s else "" for s in smiles] # Represent None as empty string
        output_file = os.path.join(args.out_dir, 'generated_smiles.txt')
        print(f"Saving {len(smiles_out)} generated SMILES (including empty strings for errors) to {output_file}")
        with open(output_file, 'w') as f:
            f.write('\n'.join(smiles_out))

    elif dataset_type == 'general_graph':
         # --- Keep existing general graph processing logic ---
        nx_graphs = []
        # Import graph processing function if needed
        from datasets_utils import seq_to_nxgraph # Ensure this uses the *old* version if needed for these datasets
        for seq_str in generated_sequences:
            try:
                # Use the appropriate seq_to_nxgraph for undirected graphs if necessary
                # This assumes the seq_to_nxgraph in datasets_utils handles these cases correctly
                graph = seq_to_nxgraph(seq_str) # Make sure this call reconstructs undirected nx.Graph
                nx_graphs.append(graph)
            except Exception as e:
                print(f"Error processing sequence to graph: {e}")
                # Optionally append None or skip
        output_file = os.path.join(args.out_dir, 'generated_graphs.pkl')
        print(f"Saving {len(nx_graphs)} generated graphs to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(nx_graphs, f)

    # --- ADDED AIG Handling ---
    elif dataset_type == 'aig':
        print("Processing generated sequences as AIGs...")
        aig_graphs = []
        # Ensure seq_to_nxgraph is imported (it should be already)
        # from datasets_utils import seq_to_nxgraph # Make sure this uses the *updated* DiGraph version

        num_processed = 0
        num_errors = 0
        for seq_str in generated_sequences:
            try:
                # Call the UPDATED seq_to_nxgraph which creates nx.DiGraph
                graph = seq_to_nxgraph(seq_str)
                # Optional: Add validation checks specific to AIGs if needed
                aig_graphs.append(graph)
                num_processed += 1
            except Exception as e:
                print(f"Error processing sequence to AIG: {e}\nSequence sample: {seq_str[:100]}...")
                num_errors += 1
                # Decide if you want to store None for errors or just skip
                # aig_graphs.append(None) # Option to store None

        output_file = os.path.join(args.out_dir, 'generated_aigs.pkl')
        print(f"Processed {num_processed} sequences, encountered {num_errors} errors.")
        print(f"Saving {len(aig_graphs)} generated AIG DiGraphs to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(aig_graphs, f)

    else:
         print(f"Warning: Unknown dataset type derived from tokenizer path '{args.tokenizer_path}'. No specific post-processing applied.")
         # Optionally save the raw sequences
         output_file = os.path.join(args.out_dir, 'generated_sequences.txt')
         print(f"Saving raw generated sequences to {output_file}")
         with open(output_file, 'w') as f:
            f.write('\n'.join(generated_sequences))


