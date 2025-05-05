# G2PT/tests/test_dataset_utils.py
import unittest
import os
import sys
import random
import tempfile
import shutil
import json
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from functools import partial
import warnings # Import warnings

# --- Add G2PT root to path to import config and modules ---
script_dir = os.path.dirname(os.path.realpath(__file__)) # G2PT/tests
g2pt_root = os.path.dirname(script_dir) # G2PT/
if g2pt_root not in sys.path:
    sys.path.append(g2pt_root)
    print(f"Appended '{g2pt_root}' to sys.path") # Log path addition

# --- Simplified Imports ---
import G2PT.configs.aig as aig_cfg
from G2PT.datasets_utils import (
    pre_tokenize_function,
    custom_randomized_topological_sort,
    to_seq_aig_topo,
    seq_to_nxgraph,
    NumpyBinDataset,
    get_datasets
)
from G2PT.evaluate_aigs import calculate_structural_aig_metrics
from G2PT.tests.graph_utils import pyg_data_to_nx

print("Successfully imported G2PT modules.")
# --- End Simplified Imports ---


# --- Helper function create_test_pyg_data_with_vocab_ids (No changes needed) ---
def create_test_pyg_data_with_vocab_ids(num_nodes, num_edges, seed=None):
    if seed is not None: torch.manual_seed(seed)
    node_vocab_ids = torch.randint(aig_cfg.NODE_VOCAB_OFFSET, aig_cfg.NODE_VOCAB_OFFSET + aig_cfg.NUM_NODE_FEATURES, (num_nodes,), dtype=torch.long)
    if num_edges > 0 and num_nodes > 1:
        edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long); edge_index = edge_index[:, edge_index[0] != edge_index[1]]; num_edges = edge_index.shape[1]
    else: edge_index = torch.empty((2, 0), dtype=torch.long); num_edges = 0
    if num_edges > 0: edge_attr_vocab_ids = torch.randint(aig_cfg.EDGE_VOCAB_OFFSET, aig_cfg.EDGE_VOCAB_OFFSET + aig_cfg.NUM_EDGE_FEATURES, (num_edges,), dtype=torch.long)
    else: edge_attr_vocab_ids = torch.empty((0,), dtype=torch.long)
    return {'x': node_vocab_ids, 'edge_index': edge_index, 'edge_attr': edge_attr_vocab_ids}

# --- Test Class 1: TestDataConversionPipeline (No changes needed) ---
class TestDataConversionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atom_type_list = list(aig_cfg.NODE_TYPE_KEYS); cls.bond_type_list = list(aig_cfg.EDGE_TYPE_KEYS)
        print("\n--- Starting Data Conversion Pipeline Tests ---")
    def _create_simple_aig_dict(self):
        return {'x': torch.tensor([aig_cfg.NODE_TYPE_VOCAB["NODE_PI"], aig_cfg.NODE_TYPE_VOCAB["NODE_PI"], aig_cfg.NODE_TYPE_VOCAB["NODE_AND"], aig_cfg.NODE_TYPE_VOCAB["NODE_PO"]], dtype=torch.long),
                'edge_index': torch.tensor([[0, 1, 2], [2, 2, 3]], dtype=torch.long),
                'edge_attr': torch.tensor([aig_cfg.EDGE_TYPE_VOCAB["EDGE_REG"], aig_cfg.EDGE_TYPE_VOCAB["EDGE_REG"], aig_cfg.EDGE_TYPE_VOCAB["EDGE_INV"]], dtype=torch.long)}
    def test_round_trip_simple_graph_no_aug(self):
        print("\nTesting simple graph round-trip (no augmentation)...")
        pyg_dict = self._create_simple_aig_dict()
        seq_dict = to_seq_aig_topo(pyg_dict, self.atom_type_list, self.bond_type_list, aug_seed=None); sequence = seq_dict['text'][0]
        print(f"  Generated Sequence (Seed=None): {sequence}"); self.assertIsInstance(sequence, str); self.assertGreater(len(sequence), 20); self.assertTrue(sequence.startswith("<boc>") and sequence.endswith("<eog>"))
        reconstructed_graph = seq_to_nxgraph(sequence, parsing_mode='strict'); self.assertIsInstance(reconstructed_graph, nx.DiGraph)
        print(f"  Reconstructed Graph: Nodes={reconstructed_graph.number_of_nodes()}, Edges={reconstructed_graph.number_of_edges()}")
        validity_metrics = calculate_structural_aig_metrics(reconstructed_graph)
        self.assertTrue(validity_metrics.get('is_structurally_valid', False), f"Reconstructed graph failed validity checks: {validity_metrics.get('constraints_failed', 'Unknown')}")
        original_pyg_data_obj = Data(x=pyg_dict['x'], edge_index=pyg_dict['edge_index'], edge_attr=pyg_dict['edge_attr']); original_nx_graph = pyg_data_to_nx(original_pyg_data_obj)
        self.assertIsNotNone(original_nx_graph, "Failed to convert original PyG data to NX for comparison.")
        nm = nx.isomorphism.categorical_node_match('type', default='UNKNOWN'); em = nx.isomorphism.categorical_edge_match('type', default='UNKNOWN_EDGE') # Use edge match now
        is_iso = nx.is_isomorphic(original_nx_graph, reconstructed_graph, node_match=nm, edge_match=em) # Check with edge attributes
        if not is_iso: print("Original NX Graph Nodes:", original_nx_graph.nodes(data=True)); print("Reconstructed NX Graph Nodes:", reconstructed_graph.nodes(data=True)); print("Original NX Graph Edges:", original_nx_graph.edges(data=True)); print("Reconstructed NX Graph Edges:", reconstructed_graph.edges(data=True))
        self.assertTrue(is_iso, "Reconstructed graph is NOT isomorphic to the original structure (including edge types)."); print("  Round-trip validity and isomorphism checks passed.")

    def test_augmentation_impact_and_validity(self):
        """Test sequence variation with seeds and validity of reconstructions."""
        print("\nTesting augmentation impact and validity...")
        # Use a graph where randomization is likely
        pyg_dict = create_test_pyg_data_with_vocab_ids(num_nodes=6, num_edges=6, seed=42)
        # Example structure (adjust if needed for more variability)
        pyg_dict['x'] = torch.tensor([
            aig_cfg.NODE_TYPE_VOCAB["NODE_PI"], aig_cfg.NODE_TYPE_VOCAB["NODE_PI"],  # 0, 1
            aig_cfg.NODE_TYPE_VOCAB["NODE_AND"], aig_cfg.NODE_TYPE_VOCAB["NODE_AND"],  # 2, 3
            aig_cfg.NODE_TYPE_VOCAB["NODE_AND"],  # 4
            aig_cfg.NODE_TYPE_VOCAB["NODE_PO"]  # 5
        ], dtype=torch.long)
        pyg_dict['edge_index'] = torch.tensor([[0, 0, 1, 1, 2, 3, 4], [2, 3, 3, 2, 4, 4, 5]],
                                              dtype=torch.long)  # 0->2, 1->3, 2->4, 3->4, 4->5
        pyg_dict['edge_attr'] = torch.randint(aig_cfg.EDGE_VOCAB_OFFSET,
                                              aig_cfg.EDGE_VOCAB_OFFSET + aig_cfg.NUM_EDGE_FEATURES, (7,),
                                              dtype=torch.long)

        # <<< ADDED: Validate the original input graph first >>>
        original_pyg_data_obj = Data(x=pyg_dict['x'], edge_index=pyg_dict['edge_index'],
                                     edge_attr=pyg_dict['edge_attr'])
        original_nx_graph = pyg_data_to_nx(original_pyg_data_obj)
        self.assertIsNotNone(original_nx_graph, "Failed to convert original PyG data to NX for validation.")
        original_validity_metrics = calculate_structural_aig_metrics(original_nx_graph)
        self.assertTrue(original_validity_metrics.get('is_structurally_valid', False),
                        f"The ORIGINAL input graph for this test is invalid: {original_validity_metrics.get('constraints_failed', 'Unknown')}")
        print("  Original input graph passed validity check.")
        # <<< END ADDED VALIDATION >>>

        num_augmentations_to_test = 5;
        generated_sequences = set();
        reconstructed_graphs = []
        for aug_seed in range(num_augmentations_to_test):
            seq_dict = to_seq_aig_topo(pyg_dict, self.atom_type_list, self.bond_type_list, aug_seed=aug_seed);
            sequence = seq_dict['text'][0];
            generated_sequences.add(sequence)
            recon_graph = seq_to_nxgraph(sequence, parsing_mode='robust');
            self.assertIsInstance(recon_graph, nx.DiGraph, f"seq_to_nxgraph failed for seed {aug_seed}")
            validity_metrics = calculate_structural_aig_metrics(recon_graph)

            # <<< Enhanced Debug Print on Failure >>>
            if not validity_metrics.get('is_structurally_valid', False):
                print(f"\nDEBUG: Failing sequence for seed {aug_seed}:\n{sequence}\n")
                print(
                    f"DEBUG: Reconstructed graph nodes ({recon_graph.number_of_nodes()}): {recon_graph.nodes(data=True)}")
                print(
                    f"DEBUG: Reconstructed graph edges ({recon_graph.number_of_edges()}): {recon_graph.edges(data=True)}")
                print(
                    f"DEBUG: Reconstructed graph AND node in-degrees: { {n: recon_graph.in_degree(n) for n, d in recon_graph.nodes(data=True) if d.get('type') == 'NODE_AND'} }")
                print(f"DEBUG: Validity metrics: {validity_metrics}\n")
                # Compare with original graph
                print(
                    f"DEBUG: Original graph nodes ({original_nx_graph.number_of_nodes()}): {original_nx_graph.nodes(data=True)}")
                print(
                    f"DEBUG: Original graph edges ({original_nx_graph.number_of_edges()}): {original_nx_graph.edges(data=True)}")
                print(
                    f"DEBUG: Original graph AND node in-degrees: { {n: original_nx_graph.in_degree(n) for n, d in original_nx_graph.nodes(data=True) if d.get('type') == 'NODE_AND'} }")
            # <<< END Enhanced Debug Print >>>

            self.assertTrue(validity_metrics.get('is_structurally_valid', False),
                            f"Reconstructed graph for seed {aug_seed} failed validity: {validity_metrics.get('constraints_failed', 'Unknown')}")
            reconstructed_graphs.append(recon_graph)

        print(
            f"  Generated {len(generated_sequences)} unique sequences from {num_augmentations_to_test} augmentations.")
        if num_augmentations_to_test > 1: self.assertGreater(len(generated_sequences), 1,
                                                             "Augmentation did not produce different sequences.")
        if len(reconstructed_graphs) >= 2:
            nm = nx.isomorphism.categorical_node_match('type', default='UNKNOWN');
            em = nx.isomorphism.categorical_edge_match('type', default='UNKNOWN_EDGE')
            self.assertTrue(
                nx.is_isomorphic(reconstructed_graphs[0], reconstructed_graphs[1], node_match=nm, edge_match=em),
                "Reconstructed graphs from different augmentations are not isomorphic.")
            print("  Isomorphism check between augmented reconstructions passed.")

    def test_seq_to_nxgraph_malformed(self):
        """Test seq_to_nxgraph handling of malformed sequences."""
        print("\nTesting seq_to_nxgraph robustness...")
        malformed1 = "<sepc> NODE_PI IDX_0 <sepc> NODE_AND IDX_1 <eoc> <bog> <sepg> IDX_0 IDX_1 EDGE_REG <eog>"
        G1_strict = seq_to_nxgraph(malformed1, parsing_mode='strict'); G1_robust = seq_to_nxgraph(malformed1, parsing_mode='robust')
        self.assertEqual(G1_strict.number_of_nodes(), 0, "Strict parsing should fail on missing <boc>")
        self.assertEqual(G1_robust.number_of_nodes(), 0, "Robust parsing should fail on missing <boc>")
        malformed2 = "<boc> <sepc> NODE_PI IDX_0 <eoc> <bog> <sepg> IDX_0 IDX_1 <eog>"
        G2_strict = seq_to_nxgraph(malformed2, parsing_mode='strict')
        self.assertEqual(G2_strict.number_of_nodes(), 1, "Strict parsing: Node should be parsed")
        self.assertEqual(G2_strict.number_of_edges(), 0, "Strict parsing should fail on malformed edge triplet")
        G2_robust = seq_to_nxgraph(malformed2, parsing_mode='robust')
        self.assertEqual(G2_robust.number_of_nodes(), 1, "Robust parsing: Node should be parsed")
        self.assertEqual(G2_robust.number_of_edges(), 0, "Robust parsing should skip malformed edge triplet")
        malformed3 = "<boc> <sepc> NODE_X IDX_0 <eoc> <bog> <sepg> IDX_0 IDX_0 EDGE_Y <eog>"
        G3_strict = seq_to_nxgraph(malformed3, parsing_mode='strict')
        self.assertEqual(G3_strict.nodes[0]['type'], 'NODE_X', "seq_to_nxgraph should preserve matched unknown node patterns")
        self.assertTrue(G3_strict.has_edge(0, 0), "Edge (0,0) should exist")
        self.assertEqual(G3_strict.edges[0, 0]['type'], 'UNKNOWN_EDGE', "seq_to_nxgraph should map unknown edge patterns to UNKNOWN_EDGE")

# --- Test Class 2: TestCustomTopoSortFunction (No changes needed) ---
class TestCustomTopoSortFunction(unittest.TestCase):
    def setUp(self): print("\n--- Starting Custom Topological Sort Tests ---")
    def test_sort_validity(self):
        G = nx.DiGraph([(0, 2), (1, 2)]); valid_sorts_set = {tuple(s) for s in nx.all_topological_sorts(G)}; self.assertEqual(len(valid_sorts_set), 2)
        for seed in range(10):
            rng = random.Random(seed); custom_sort_result = custom_randomized_topological_sort(G, rng)
            self.assertIsInstance(custom_sort_result, list); self.assertIn(tuple(custom_sort_result), valid_sorts_set)
    def test_sort_variety(self):
        G = nx.DiGraph([(0, 2), (1, 2)]); valid_sorts_set = {tuple(s) for s in nx.all_topological_sorts(G)}; self.assertEqual(len(valid_sorts_set), 2)
        generated_sorts_set = set()
        for seed in range(20): rng = random.Random(seed); generated_sorts_set.add(tuple(custom_randomized_topological_sort(G, rng)))
        self.assertGreater(len(generated_sorts_set), 1); self.assertTrue(generated_sorts_set.issubset(valid_sorts_set)); self.assertEqual(generated_sorts_set, valid_sorts_set)
    def test_single_sort_graph(self):
        G = nx.DiGraph([(0, 1), (1, 2), (2, 3)]); expected_sort = [0, 1, 2, 3]
        valid_sorts_list = list(nx.all_topological_sorts(G)); self.assertEqual(len(valid_sorts_list), 1); self.assertEqual(valid_sorts_list[0], expected_sort)
        for seed in range(5): rng = random.Random(seed); custom_sort_result = custom_randomized_topological_sort(G, rng); self.assertEqual(custom_sort_result, expected_sort)
    def test_cycle_detection(self):
        G = nx.DiGraph([(0, 1), (1, 2), (2, 0)]); rng = random.Random(42)
        with self.assertRaises(nx.NetworkXUnfeasible): custom_randomized_topological_sort(G, rng)
        with self.assertRaises(nx.NetworkXUnfeasible): list(nx.all_topological_sorts(G))

# --- Test Class 3: TestDatasetLoading ---
class TestDatasetLoading(unittest.TestCase):

    tokenizer = None
    real_data_dir = None # Path to the specific dataset dir (e.g., .../datasets/aig)
    train_shape = None
    eval_shape = None
    data_meta = None # Store loaded metadata
    real_data_exists = False # Flag to check if real data is accessible

    @classmethod
    def setUpClass(cls):
        """Load REAL tokenizer and REAL dataset metadata."""
        print("\n--- Starting Dataset Loading Tests (Using Real Data) ---")

        # --- Load REAL Tokenizer ---
        real_tokenizer_path = os.path.abspath(os.path.join(g2pt_root, aig_cfg.tokenizer_path))
        print(f"Attempting to load REAL tokenizer from: {real_tokenizer_path}")
        if not os.path.isdir(real_tokenizer_path):
             print(f"ERROR: Real tokenizer directory not found at {real_tokenizer_path}")
             cls.tokenizer = None
        else:
            try:
                cls.tokenizer = AutoTokenizer.from_pretrained(real_tokenizer_path, use_fast=True, legacy=False)
                if cls.tokenizer.unk_token is None or cls.tokenizer.unk_token_id is None:
                     print(f"ERROR: Loaded tokenizer from {real_tokenizer_path} is missing a functional UNK token.")
                     if '[UNK]' in cls.tokenizer.vocab:
                          cls.tokenizer.unk_token = '[UNK]'; cls.tokenizer.unk_token_id = cls.tokenizer.vocab['[UNK]']
                          print("Manually set UNK token based on vocab.")
                     else: print("Could not find '[UNK]' in vocab to manually set."); cls.tokenizer = None
                else: print(f"Real tokenizer loaded successfully. UNK token: '{cls.tokenizer.unk_token}' (ID: {cls.tokenizer.unk_token_id})")
            except Exception as e: print(f"Failed to load REAL tokenizer from {real_tokenizer_path}: {e}"); cls.tokenizer = None

        # --- Load REAL Metadata ---
        try:
            # <<< FIX: Construct path correctly >>>
            # Construct path to the base dataset dir (e.g., G2PT/datasets/)
            base_data_dir = os.path.abspath(os.path.join(g2pt_root, os.path.dirname(aig_cfg.data_dir)))
            # Construct path to the specific dataset dir (e.g., G2PT/datasets/aig/)
            cls.real_data_dir = os.path.join(base_data_dir)
            meta_path = os.path.join(cls.real_data_dir, 'data_meta.json')
            # --- End Fix ---
            print(f"Attempting to load REAL metadata from: {meta_path}")

            if not os.path.exists(meta_path):
                 raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            with open(meta_path, 'r') as f:
                cls.data_meta = json.load(f)

            train_key = 'train_shape'
            eval_key = 'eval_shape' if 'eval_shape' in cls.data_meta else 'val_shape'
            if train_key not in cls.data_meta or eval_key not in cls.data_meta:
                 raise KeyError("Missing train_shape or eval/val_shape in metadata.")

            cls.train_shape = {k: tuple(v) for k, v in cls.data_meta[train_key].items()}
            cls.eval_shape = {k: tuple(v) for k, v in cls.data_meta[eval_key].items()}
            cls.eval_split_name = 'eval' if 'eval_shape' in cls.data_meta else 'val'

            train_bin_path = os.path.join(cls.real_data_dir, 'train')
            eval_bin_path = os.path.join(cls.real_data_dir, cls.eval_split_name)
            if not os.path.isdir(train_bin_path) or not os.path.isdir(eval_bin_path):
                 raise FileNotFoundError("Train or Eval/Val binary data directory not found.")

            cls.real_data_exists = True
            print(f"Real metadata loaded successfully from {meta_path}.")
            print(f"Train shape: {cls.train_shape}")
            print(f"{cls.eval_split_name.capitalize()} shape: {cls.eval_shape}")

        except FileNotFoundError as e:
            print(f"ERROR loading real data/metadata: {e}")
            print(f"Looked in: {cls.real_data_dir}")
            print("Ensure your actual dataset (including data_meta.json, train/, eval/ or val/) exists at the path specified in G2PT/configs/aig.py (data_dir).")
            cls.real_data_exists = False
        except (KeyError, Exception) as e:
            print(f"ERROR parsing real metadata: {e}")
            cls.real_data_exists = False

    @classmethod
    def tearDownClass(cls):
        """No temp directory to clean up."""
        print("\nFinished Dataset Loading Tests.")
        pass


    def test_numpy_bin_dataset_len_and_getitem(self):
        """Test NumpyBinDataset length and item retrieval using REAL data."""
        num_augmentations = 3
        split = 'train'
        split_path = os.path.join(self.real_data_dir, split)
        shape = self.train_shape
        num_original_data = shape['xs'][0]
        process_fn = partial(pre_tokenize_function, tokenizer=self.tokenizer, order_function=to_seq_aig_topo, atom_type=aig_cfg.NODE_TYPE_KEYS, bond_type=aig_cfg.EDGE_TYPE_KEYS)
        try:
            dataset = NumpyBinDataset(split_path, num_original_data, aig_cfg.NUM_NODE_FEATURES, aig_cfg.NUM_EDGE_FEATURES, shape=shape, process_fn=process_fn, num_augmentations=num_augmentations)
        except Exception as e: self.fail(f"Failed to initialize NumpyBinDataset with real data path '{split_path}': {e}")
        self.assertEqual(len(dataset), num_original_data * num_augmentations)
        try:
            item0 = dataset[0]; self.assertIsInstance(item0, dict); self.assertIn('input_ids', item0); self.assertIsInstance(item0['input_ids'], torch.Tensor); self.assertIn('attention_mask', item0); self.assertIsInstance(item0['attention_mask'], torch.Tensor); self.assertIn('labels', item0); self.assertIsInstance(item0['labels'], torch.Tensor)
            decoded_text0 = self.tokenizer.decode(item0['input_ids'], skip_special_tokens=False); self.assertTrue(decoded_text0.startswith("<boc>"))
            if num_augmentations > 1 and num_original_data > 0:
                item1 = dataset[1]; self.assertIsInstance(item1, dict); decoded_text1 = self.tokenizer.decode(item1['input_ids'], skip_special_tokens=False); self.assertTrue(decoded_text1.startswith("<boc>"))
                self.assertNotEqual(decoded_text0, decoded_text1, "Augmentations did not produce different sequences for the same graph.")
            if num_original_data > 1:
                item_next_graph = dataset[num_augmentations]; self.assertIsInstance(item_next_graph, dict); decoded_text_next = self.tokenizer.decode(item_next_graph['input_ids'], skip_special_tokens=False); self.assertTrue(decoded_text_next.startswith("<boc>"))
        except Exception as e: self.fail(f"Error during dataset __getitem__ or tokenization with real data: {e}")
        with self.assertRaises(IndexError): _ = dataset[len(dataset)]


    def test_get_datasets(self):
        """Test the get_datasets function using REAL data paths."""
        num_augmentations = 2
        # No need to modify aig_cfg.data_dir, get_datasets should use it directly
        print(f"test_get_datasets: Using aig_cfg.data_dir = {aig_cfg.data_dir}")
        try:
            train_ds, eval_ds = get_datasets(dataset_name='aig', tokenizer=self.tokenizer, order='topo', num_augmentations=num_augmentations)
            self.assertIsInstance(train_ds, NumpyBinDataset); self.assertIsInstance(eval_ds, NumpyBinDataset)
            self.assertEqual(train_ds.num_augmentations, num_augmentations); self.assertEqual(eval_ds.num_augmentations, 1)
            self.assertEqual(len(train_ds), self.train_shape['xs'][0] * num_augmentations)
            self.assertEqual(len(eval_ds), self.eval_shape['xs'][0] * 1)
            self.assertTrue(callable(train_ds.process_fn)); self.assertTrue(callable(eval_ds.process_fn))
        except Exception as e: self.fail(f"get_datasets failed when using real data paths: {e}")

if __name__ == '__main__':
    unittest.main()
