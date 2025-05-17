#!/usr/bin/env python3
import os
import os.path as osp
import warnings
from typing import List, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

# --- Configuration Constants ---
# These are now imported from your aig_config.py file.
# Ensure aig_config.py is in your Python path or the same directory.
try:
    from aig_config import (
        MAX_NODE_COUNT,
        NUM_EXPLICIT_NODE_FEATURES,
        PADDING_NODE_CHANNEL,
        NUM_NODE_ATTRIBUTES,  # Total node feature dimensions (explicit + padding)
        NUM_EXPLICIT_EDGE_FEATURES,
        NO_EDGE_CHANNEL,  # Index of the no-edge channel
        NUM_EDGE_ATTRIBUTES  # Total adjacency/edge channels (explicit + no-edge)
    )

    print("Successfully imported AIG configuration constants.")
except ImportError as e:
    warnings.warn(f"Could not import AIG configuration constants from aig_config.py: {e}. "
                  "Please ensure it's available and correctly configured. "
                  "Using placeholder values for script structure, but this will likely fail at runtime if aig_config is missing.")
    # Placeholders to allow script structure to be valid, but processing will likely fail
    MAX_NODE_COUNT = 64
    NUM_EXPLICIT_NODE_FEATURES = 4
    PADDING_NODE_CHANNEL = 4
    NUM_NODE_ATTRIBUTES = 5
    NUM_EXPLICIT_EDGE_FEATURES = 2
    NO_EDGE_CHANNEL = 2
    NUM_EDGE_ATTRIBUTES = 3


# --- Transform for Padding and Adjacency Matrix Reformatting ---
class AIGTransformAndPad(BaseTransform):
    def __init__(self, max_nodes: int,
                 num_explicit_node_features: int, padding_node_channel_idx: int, total_node_feature_dim: int,
                 num_explicit_edge_features: int, no_edge_channel_idx: int, total_adj_channels: int):
        """
        Initializes the transformation class.

        Args:
            max_nodes: MAX_NODE_COUNT from aig_config.
            num_explicit_node_features: NUM_EXPLICIT_NODE_FEATURES from aig_config.
            padding_node_channel_idx: PADDING_NODE_CHANNEL from aig_config.
            total_node_feature_dim: NUM_NODE_ATTRIBUTES from aig_config.
            num_explicit_edge_features: NUM_EXPLICIT_EDGE_FEATURES from aig_config.
            no_edge_channel_idx: NO_EDGE_CHANNEL from aig_config.
            total_adj_channels: NUM_EDGE_ATTRIBUTES from aig_config.
        """
        self.max_nodes = max_nodes
        self.num_explicit_node_features = num_explicit_node_features
        self.padding_node_channel_idx = padding_node_channel_idx
        self.total_node_feature_dim = total_node_feature_dim
        self.num_explicit_edge_features = num_explicit_edge_features
        self.no_edge_channel_idx = no_edge_channel_idx
        self.total_adj_channels = total_adj_channels

    def __call__(self, data: Data) -> Data:
        """
        Applies padding and adjacency transformation to a single Data object.
        Assumes input data.x has shape [num_actual_nodes, num_explicit_node_features]
        and data.edge_attr has shape [num_edges, num_explicit_edge_features].
        """
        graph_id_for_error = getattr(data, 'graph_name_original', 'Unknown')
        num_actual_nodes = data.num_nodes.item()

        # 1. Pad Node Features (data.x)
        # Expected new_x shape: [max_nodes, total_node_feature_dim]
        new_x = torch.zeros((self.max_nodes, self.total_node_feature_dim), dtype=torch.float)

        # Initialize all to padding type by default
        if self.padding_node_channel_idx < self.total_node_feature_dim:
            new_x[:, self.padding_node_channel_idx] = 1.0
        else:
            warnings.warn(f"Graph {graph_id_for_error}: padding_node_channel_idx ({self.padding_node_channel_idx}) "
                          f"is out of bounds for total_node_feature_dim ({self.total_node_feature_dim}). Padding type not set.")

        if num_actual_nodes > 0:
            if not hasattr(data, 'x') or data.x is None:
                warnings.warn(
                    f"Graph {graph_id_for_error}: data.x is missing or None, but num_actual_nodes is {num_actual_nodes}. Cannot populate node features.")
            else:
                if data.x.shape[0] != num_actual_nodes:
                    warnings.warn(
                        f"Graph {graph_id_for_error}: data.x row count ({data.x.shape[0]}) does not match data.num_nodes ({num_actual_nodes}).")

                if data.x.shape[1] != self.num_explicit_node_features:
                    warnings.warn(
                        f"Graph {graph_id_for_error}: Input data.x feature dimension ({data.x.shape[1]}) does not match "
                        f"configured num_explicit_node_features ({self.num_explicit_node_features}). Slicing data.x.")

                # Fill in actual node features
                # Slice data.x to ensure it matches num_explicit_node_features if it's wider for some reason
                # and ensure we don't try to read more nodes than available in data.x
                nodes_to_copy = min(num_actual_nodes, data.x.shape[0])
                features_to_copy = min(self.num_explicit_node_features, data.x.shape[1])

                new_x[:nodes_to_copy, :features_to_copy] = data.x[:nodes_to_copy, :features_to_copy]

                # Ensure actual nodes are not marked as padding type
                if self.padding_node_channel_idx < self.total_node_feature_dim:
                    new_x[:nodes_to_copy, self.padding_node_channel_idx] = 0.0

        # 2. Create Padded Adjacency Matrix (new_adj)
        # Expected new_adj shape: [total_adj_channels, max_nodes, max_nodes]
        new_adj = torch.zeros((self.total_adj_channels, self.max_nodes, self.max_nodes), dtype=torch.float)

        # Initialize "no edge" channel (NO_EDGE_CHANNEL)
        if self.no_edge_channel_idx < self.total_adj_channels:
            for i in range(self.max_nodes):
                for j in range(self.max_nodes):
                    if i != j:  # No self-loops for "no edge" type
                        new_adj[self.no_edge_channel_idx, i, j] = 1.0
        else:
            warnings.warn(f"Graph {graph_id_for_error}: no_edge_channel_idx ({self.no_edge_channel_idx}) "
                          f"is out of bounds for total_adj_channels ({self.total_adj_channels}). 'No edge' channel not set.")

        # Diagonals of all channels (including no_edge_channel) remain 0.

        # Populate explicit edge channels using data.edge_index and data.edge_attr
        if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
            if not hasattr(data, 'edge_attr') or data.edge_attr is None:
                warnings.warn(
                    f"Graph {graph_id_for_error} has edge_index but no edge_attr. "
                    "Cannot determine explicit edge types for the new adjacency matrix.")
            elif data.edge_attr.shape[0] != data.edge_index.shape[1]:
                warnings.warn(
                    f"Graph {graph_id_for_error}: data.edge_attr row count ({data.edge_attr.shape[0]}) "
                    f"does not match data.edge_index column count ({data.edge_index.shape[1]}). Edge attributes may be misaligned."
                )
            elif data.edge_attr.shape[1] != self.num_explicit_edge_features:
                warnings.warn(
                    f"Graph {graph_id_for_error} edge_attr feature dimension ({data.edge_attr.shape[1]}) "
                    f"does not match configured num_explicit_edge_features ({self.num_explicit_edge_features}). "
                    "Cannot reliably populate new_adj from edge_attr.")
            else:
                for k in range(data.edge_index.size(1)):  # Iterate over each directed edge
                    u, v = data.edge_index[0, k].item(), data.edge_index[1, k].item()

                    if not (0 <= u < num_actual_nodes and 0 <= v < num_actual_nodes):
                        warnings.warn(
                            f"Graph {graph_id_for_error}: Edge index ({u},{v}) "
                            f"is out of bounds for actual nodes ({num_actual_nodes}). Skipping this edge for new_adj.")
                        continue

                    edge_features_one_hot = data.edge_attr[k]

                    try:
                        explicit_channel_idx = torch.argmax(edge_features_one_hot).item()
                        if not (0 <= explicit_channel_idx < self.num_explicit_edge_features):
                            warnings.warn(
                                f"Graph {graph_id_for_error}: Invalid explicit channel index {explicit_channel_idx} "
                                f"derived from edge_attr for edge ({u},{v}). Skipping.")
                            continue

                        if not (
                                0 <= u < self.max_nodes and 0 <= v < self.max_nodes):  # Check against max_nodes for new_adj
                            warnings.warn(
                                f"Graph {graph_id_for_error}: Edge index ({u},{v}) "
                                f"is out of bounds for new_adj (max_nodes: {self.max_nodes}). This should not happen if u,v < num_actual_nodes <= max_nodes. Skipping."
                            )
                            continue

                        new_adj[explicit_channel_idx, u, v] = 1.0
                        if self.no_edge_channel_idx < self.total_adj_channels:
                            new_adj[self.no_edge_channel_idx, u, v] = 0.0
                    except Exception as e:
                        warnings.warn(
                            f"Graph {graph_id_for_error}: Error processing edge_attr for edge ({u},{v}): {e}. Skipping edge for new_adj.")
                        continue

        # Create the transformed Data object
        transformed_data = Data()
        transformed_data.x = new_x
        transformed_data.adj = new_adj
        transformed_data.num_nodes = data.num_nodes  # Preserve actual number of nodes

        # --- ADDING ASSERTIONS HERE ---
        expected_x_shape = (self.max_nodes, self.total_node_feature_dim)
        assert transformed_data.x.shape == expected_x_shape, \
            f"Graph {graph_id_for_error}: transformed_data.x shape is {transformed_data.x.shape}, expected {expected_x_shape}"

        expected_adj_shape = (self.total_adj_channels, self.max_nodes, self.max_nodes)
        assert transformed_data.adj.shape == expected_adj_shape, \
            f"Graph {graph_id_for_error}: transformed_data.adj shape is {transformed_data.adj.shape}, expected {expected_adj_shape}"
        # --- END OF ASSERTIONS ---

        # Pass through any other attributes from the original Data object
        for key in data.keys():
            if key not in ['x', 'adj', 'num_nodes', 'edge_index', 'edge_attr']:
                transformed_data[key] = data[key]

        return transformed_data


# --- InMemoryDataset for Padded and Formatted AIGs ---
class AIGPaddedInMemoryDataset(InMemoryDataset):
    def __init__(self, root: str,
                 split: str,
                 raw_pt_input_dir: str,
                 raw_pt_input_filename: str = "aig_undirected.pt",
                 transform: Optional[callable] = None,
                 pre_transform: Optional[callable] = None,
                 pre_filter: Optional[callable] = None,
                 processed_file_prefix: str = "padded_aig_"):

        self.split = split
        self.raw_pt_input_path = osp.join(raw_pt_input_dir, raw_pt_input_filename)
        self.processed_file_prefix = processed_file_prefix
        self.graph_identifiers: List[str] = []

        super().__init__(root, transform, pre_transform, pre_filter)

        try:
            loaded_content = torch.load(self.processed_paths[0], weights_only=False)
            if isinstance(loaded_content, tuple) and len(loaded_content) == 3:
                self.data, self.slices, self.graph_identifiers = loaded_content
                print(f"Successfully loaded processed & padded '{self.split}' AIG data from: {self.processed_paths[0]}")
            elif isinstance(loaded_content, tuple) and len(loaded_content) == 2:
                self.data, self.slices = loaded_content
                num_graphs = 0
                if self.slices and list(self.slices.keys()):  # Check if slices is not empty
                    slice_key = list(self.slices.keys())[0]
                    if self.slices[slice_key].numel() > 0:  # Check if the slice tensor is not empty
                        num_graphs = self.slices[slice_key].size(0) - 1
                self.graph_identifiers = [f"graph_{i}" for i in range(num_graphs)]
                warnings.warn(
                    f"Loaded older format (data, slices) from {self.processed_paths[0]}; graph identifiers are dummies.")
            else:
                raise TypeError(f"Loaded content from {self.processed_paths[0]} is not a recognized tuple format.")

        except FileNotFoundError:
            print(f"Padded processed file not found at {self.processed_paths[0]}. "
                  "This is normal if processing for the first time. Dataset will attempt to process.")
        except Exception as e:
            warnings.warn(
                f"Could not load data from {self.processed_paths[0]} with weights_only=False: {e}. "
                "If this is the first run, processing will be attempted. Otherwise, the file might be corrupted or incompatible.")

    @property
    def raw_file_names(self) -> List[str]:
        return [osp.basename(self.raw_pt_input_path)]

    @property
    def raw_dir(self) -> str:
        return osp.dirname(self.raw_pt_input_path)

    @property
    def processed_file_names(self) -> str:
        return f'{self.processed_file_prefix}{self.split}.pt'

    def download(self):
        if not osp.exists(self.raw_pt_input_path):
            raise FileNotFoundError(f"Input .pt file not found: {self.raw_pt_input_path}. "
                                    "This dataset expects it to be pre-existing in the specified `raw_pt_input_dir`.")

    def process(self):
        print(
            f"Processing for '{self.split}' split: Loading from '{self.raw_pt_input_path}' and applying transformations...")

        if not osp.exists(self.raw_pt_input_path):
            raise FileNotFoundError(f"Cannot process: Raw input file {self.raw_pt_input_path} not found.")

        try:
            unpadded_data_list: List[Data] = torch.load(self.raw_pt_input_path, weights_only=False)
            if not isinstance(unpadded_data_list, list):
                raise TypeError(f"Expected a list of Data objects from {self.raw_pt_input_path}, "
                                f"but got {type(unpadded_data_list)}.")
        except Exception as e:
            raise RuntimeError(f"Error loading unpadded data from {self.raw_pt_input_path}: {e}")

        transformer = AIGTransformAndPad(
            max_nodes=MAX_NODE_COUNT,
            num_explicit_node_features=NUM_EXPLICIT_NODE_FEATURES,
            padding_node_channel_idx=PADDING_NODE_CHANNEL,
            total_node_feature_dim=NUM_NODE_ATTRIBUTES,
            num_explicit_edge_features=NUM_EXPLICIT_EDGE_FEATURES,
            no_edge_channel_idx=NO_EDGE_CHANNEL,
            total_adj_channels=NUM_EDGE_ATTRIBUTES
        )

        transformed_data_list: List[Data] = []
        current_graph_identifiers: List[str] = []

        for i, data_obj in enumerate(tqdm(unpadded_data_list, desc="Applying padding and transform")):
            if not isinstance(data_obj, Data):
                warnings.warn(f"Item {i} in {self.raw_pt_input_path} is not a PyG Data object. Skipping.")
                continue

            graph_id_for_error_process = getattr(data_obj, 'graph_name_original', f"{self.split}_raw_graph_{i}")

            if not hasattr(data_obj, 'num_nodes') or data_obj.num_nodes is None:
                warnings.warn(f"Data object {graph_id_for_error_process} "
                              "is missing 'num_nodes' attribute. Attempting to infer from data.x.shape[0].")
                if hasattr(data_obj, 'x') and data_obj.x is not None:
                    data_obj.num_nodes = torch.tensor(data_obj.x.shape[0], dtype=torch.long)
                else:
                    warnings.warn(f"Cannot infer num_nodes for data object {graph_id_for_error_process}. Skipping.")
                    continue

            # Ensure num_nodes is not greater than max_nodes before transformation
            if data_obj.num_nodes.item() > MAX_NODE_COUNT:
                warnings.warn(f"Graph {graph_id_for_error_process} has {data_obj.num_nodes.item()} nodes, "
                              f"which exceeds MAX_NODE_COUNT ({MAX_NODE_COUNT}). Skipping this graph.")
                continue

            transformed_obj = transformer(data_obj)
            if transformed_obj:
                transformed_data_list.append(transformed_obj)
                identifier = getattr(data_obj, 'graph_name_original', f"{self.split}_graph_{i}")
                current_graph_identifiers.append(identifier)
                if hasattr(transformed_obj, 'graph_name_original') and not transformed_obj.graph_name_original:
                    transformed_obj.graph_name_original = identifier

        if self.pre_filter is not None:
            filtered_indices = [i for i, data in enumerate(transformed_data_list) if self.pre_filter(data)]
            transformed_data_list = [transformed_data_list[i] for i in filtered_indices]
            current_graph_identifiers = [current_graph_identifiers[i] for i in filtered_indices]
            print(f"Applied pre_filter, {len(transformed_data_list)} graphs remaining for {self.split}.")

        if self.pre_transform is not None:
            print(f"Applying pre_transform for {self.split}...")
            transformed_data_list = [self.pre_transform(data) for data in
                                     tqdm(transformed_data_list, desc="Pre-transforming (on padded data)")]

        if not transformed_data_list:
            warnings.warn(
                f"No data to save for split '{self.split}' after transformation. Saving empty dataset structure.")

        data, slices = self.collate(transformed_data_list)
        torch.save((data, slices, current_graph_identifiers), self.processed_paths[0])
        print(
            f"Finished processing for '{self.split}'. Saved {len(transformed_data_list)} graphs to {self.processed_paths[0]}.")

    def get(self, idx: int) -> Data:
        data = super().get(idx)
        if hasattr(self, 'graph_identifiers') and self.graph_identifiers and idx < len(self.graph_identifiers):
            data.graph_name = self.graph_identifiers[idx]
        else:
            data.graph_name = f"{self.split}_graph_idx_{idx}"
        return data

    def len(self) -> int:
        if not hasattr(self, 'slices') or not self.slices: return 0
        if not self.slices: return 0
        slice_key = next(iter(self.slices.keys()), None)
        return self.slices[slice_key].size(0) - 1 if slice_key and self.slices[slice_key].numel() > 0 else 0
