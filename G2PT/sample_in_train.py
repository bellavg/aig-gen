"""
Sample from a trained model
"""
import os
from contextlib import nullcontext
import torch
import numpy as np
from model import GPTConfig, GPT
import networkx as nx
from transformers import AutoTokenizer
from datasets_utils import seq_to_nxgraph
import argparse
from contextlib import nullcontext



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
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    hf_model = model.to_hf()
    hf_model.eval()
    hf_model.to(device)

    return hf_model


def generate_sequences(model, tokenizer, batch_size, num_samples, device, prefix=None, temperature=0.8):
    if prefix is None:
        inputs = tokenizer(['<boc>'] * batch_size, return_tensors="pt")
    else:
        inputs = tokenizer([prefix] * batch_size, return_tensors="pt")
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


def get_graphs(model, tokenizer, seed, temperature,
               batch_size, num_samples, prefix=None, out_dir=None, save=False, parsing_mode='robust'):
    device, ctx = setup_device(seed)
    prefix = prefix
    temperature = temperature
    hf_model = model.to_hf()
    hf_model.eval()
    hf_model.to(device)
    with ctx:
        generated_sequences = generate_sequences(
            hf_model,
            tokenizer,
            batch_size,
            num_samples,
            device,
            prefix=prefix,
            temperature=temperature,
        )

    nx_graphs = []
    num_processed = 0
    num_errors = 0

    for i, seq_str in enumerate(generated_sequences):
        try:
            # Convert sequence to graph using the function from datasets_utils
            graph = seq_to_nxgraph(seq_str, parsing_mode=parsing_mode)  # # Should return nx.DiGraph
            # Basic check: Ensure it's a NetworkX graph object
            if isinstance(graph, nx.Graph):  # Check base class (DiGraph inherits from Graph)
                nx_graphs.append(graph)
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

    if save and out_dir is not None:
        import pickle
        open(f'{out_dir}/generated_graphs.pkl', 'wb').write(pickle.dumps(nx_graphs))

    return nx_graphs