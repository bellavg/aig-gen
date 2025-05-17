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
        num_actual_nodes = data.num_nodes.item()

        # 1. Pad Node Features (data.x)
        # New x shape: [max_nodes, total_node_feature_dim]
        new_x = torch.zeros((self.max_nodes, self.total_node_feature_dim), dtype=torch.float)

        # Initialize all to padding type by default
        if self.padding_node_channel_idx < self.total_node_feature_dim:
            new_x[:, self.padding_node_channel_idx] = 1.0

        if num_actual_nodes > 0:
            if data.x.shape[0] != num_actual_nodes:
                warnings.warn(
                    f"data.x row count ({data.x.shape[0]}) does not match data.num_nodes ({num_actual_nodes}). Graph: {getattr(data, 'graph_name_original', 'Unknown')}")

            if data.x.shape[1] != self.num_explicit_node_features:
                warnings.warn(f"Input data.x feature dimension ({data.x.shape[1]}) does not match "
                              f"configured num_explicit_node_features ({self.num_explicit_node_features}). Graph: {getattr(data, 'graph_name_original', 'Unknown')}")
                # Handle mismatch if possible, or this might lead to errors
                # For now, assume it will be sliced or cause an error if not matching

            # Fill in actual node features
            # Slice data.x to ensure it matches num_explicit_node_features if it's wider for some reason
            new_x[:num_actual_nodes, :self.num_explicit_node_features] = data.x[:num_actual_nodes,
                                                                         :self.num_explicit_node_features]
            # Ensure actual nodes are not marked as padding type
            if self.padding_node_channel_idx < self.total_node_feature_dim:
                new_x[:num_actual_nodes, self.padding_node_channel_idx] = 0.0

        # 2. Create Padded Adjacency Matrix (new_adj)
        # Shape: [total_adj_channels, max_nodes, max_nodes]
        new_adj = torch.zeros((self.total_adj_channels, self.max_nodes, self.max_nodes), dtype=torch.float)

        # Initialize "no edge" channel (NO_EDGE_CHANNEL)
        if self.no_edge_channel_idx < self.total_adj_channels:
            for i in range(self.max_nodes):
                for j in range(self.max_nodes):
                    if i != j:  # No self-loops for "no edge" type
                        new_adj[self.no_edge_channel_idx, i, j] = 1.0
        # Diagonals of all channels (including no_edge_channel) remain 0.

        # Populate explicit edge channels using data.edge_index and data.edge_attr
        # data.edge_index: [2, num_undirected_actual_edges]
        # data.edge_attr: [num_undirected_actual_edges, num_explicit_edge_features] (one-hot)
        if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
            if not hasattr(data, 'edge_attr') or data.edge_attr is None:
                warnings.warn(
                    f"Graph {getattr(data, 'graph_name_original', 'Unknown')} has edge_index but no edge_attr. "
                    "Cannot determine explicit edge types for the new adjacency matrix.")
            elif data.edge_attr.shape[1] != self.num_explicit_edge_features:
                warnings.warn(
                    f"Graph {getattr(data, 'graph_name_original', 'Unknown')} edge_attr feature dimension ({data.edge_attr.shape[1]}) "
                    f"does not match configured num_explicit_edge_features ({self.num_explicit_edge_features}). "
                    "Cannot reliably populate new_adj from edge_attr.")
            else:
                for k in range(data.edge_index.size(1)):  # Iterate over each directed edge in the undirected pair
                    u, v = data.edge_index[0, k].item(), data.edge_index[1, k].item()

                    # Ensure u and v are within the bounds of actual nodes for this graph
                    if u >= num_actual_nodes or v >= num_actual_nodes:
                        warnings.warn(
                            f"Edge index ({u},{v}) in graph {getattr(data, 'graph_name_original', 'Unknown')} "
                            f"is out of bounds for actual nodes ({num_actual_nodes}). Skipping this edge for new_adj.")
                        continue

                    edge_features_one_hot = data.edge_attr[k]  # Shape: [num_explicit_edge_features]

                    try:
                        # Determine the channel for this explicit edge type
                        explicit_channel_idx = torch.argmax(edge_features_one_hot).item()
                        if not (0 <= explicit_channel_idx < self.num_explicit_edge_features):
                            warnings.warn(
                                f"Invalid explicit channel index {explicit_channel_idx} derived from edge_attr for edge ({u},{v}). Skipping.")
                            continue

                        # Set the explicit edge channel
                        new_adj[explicit_channel_idx, u, v] = 1.0
                        # new_adj is made symmetric by processing both (u,v) and (v,u) from edge_index

                        # Mark this edge as NOT a "no edge"
                        if self.no_edge_channel_idx < self.total_adj_channels:
                            new_adj[self.no_edge_channel_idx, u, v] = 0.0
                    except Exception as e:
                        warnings.warn(
                            f"Error processing edge_attr for edge ({u},{v}) in graph {getattr(data, 'graph_name_original', 'Unknown')}: {e}. Skipping edge for new_adj.")
                        continue

        # Create the transformed Data object
        transformed_data = Data()
        transformed_data.x = new_x
        transformed_data.adj = new_adj
        transformed_data.num_nodes = data.num_nodes  # Preserve actual number of nodes

        # Pass through original edge_index and edge_attr if they exist and are needed by some models
        if hasattr(data, 'edge_index'): transformed_data.edge_index = data.edge_index
        if hasattr(data, 'edge_attr'): transformed_data.edge_attr = data.edge_attr

        # Pass through any other attributes from the original Data object
        for key in data.keys:
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
                 processed_file_prefix: str = "padded_aig_"):  # Added "aig" to prefix

        self.split = split
        self.raw_pt_input_path = osp.join(raw_pt_input_dir, raw_pt_input_filename)
        self.processed_file_prefix = processed_file_prefix
        self.graph_identifiers: List[str] = []

        super().__init__(root, transform, pre_transform, pre_filter)

        try:
            # InMemoryDataset's __init__ calls self.load() if processed files exist.
            # self.load() attempts to load self.processed_paths[0]
            self.data, self.slices, self.graph_identifiers = torch.load(self.processed_paths[0])
            print(f"Successfully loaded processed & padded '{self.split}' AIG data from: {self.processed_paths[0]}")
        except FileNotFoundError:
            print(f"Padded processed file not found at {self.processed_paths[0]}. "
                  "This is normal if processing for the first time. Dataset will attempt to process.")
        except Exception as e:
            warnings.warn(
                f"Could not load full data tuple (data, slices, identifiers) from {self.processed_paths[0]}: {e}. "
                "Attempting to load just data and slices if processing was already done.")
            try:
                self.data, self.slices = torch.load(self.processed_paths[0])
                num_graphs = self.slices[list(self.slices.keys())[0]].size(
                    0) - 1 if self.slices and self.slices.keys() else 0
                self.graph_identifiers = [f"graph_{i}" for i in range(num_graphs)]  # Dummy identifiers
                warnings.warn("Loaded only data and slices; graph identifiers are dummies.")
            except Exception as e_simple:
                # This might happen if the file is truly corrupted or doesn't exist and process() hasn't run
                warnings.warn(f"Failed to load even basic data/slices from {self.processed_paths[0]}: {e_simple}. "
                              "If this is the first run, processing will be attempted.")

    @property
    def raw_file_names(self) -> List[str]:
        # The "raw" file for this dataset is the pre-existing .pt file (e.g., aig_undirected.pt)
        return [osp.basename(self.raw_pt_input_path)]

    @property
    def raw_dir(self) -> str:
        # The directory where the input .pt file is located
        return osp.dirname(self.raw_pt_input_path)

    @property
    def processed_file_names(self) -> str:
        # Name of the file this dataset will create (e.g., padded_aig_train.pt)
        return f'{self.processed_file_prefix}{self.split}.pt'

    def download(self):
        if not osp.exists(self.raw_pt_input_path):
            raise FileNotFoundError(f"Input .pt file not found: {self.raw_pt_input_path}. "
                                    "This dataset expects it to be pre-existing in the specified `raw_pt_input_dir`.")

    def process(self):
        print(
            f"Processing for '{self.split}' split: Loading from '{self.raw_pt_input_path}' and applying transformations...")

        if not osp.exists(self.raw_pt_input_path):  # Double check before loading
            raise FileNotFoundError(f"Cannot process: Raw input file {self.raw_pt_input_path} not found.")

        try:
            unpadded_data_list: List[Data] = torch.load(self.raw_pt_input_path)
            if not isinstance(unpadded_data_list, list):
                raise TypeError(f"Expected a list of Data objects from {self.raw_pt_input_path}, "
                                f"but got {type(unpadded_data_list)}.")
        except Exception as e:
            raise RuntimeError(f"Error loading unpadded data from {self.raw_pt_input_path}: {e}")

        # Instantiate the transformer using imported config values
        transformer = AIGTransformAndPad(
            max_nodes=MAX_NODE_COUNT,
            num_explicit_node_types=NUM_EXPLICIT_NODE_FEATURES,
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

            # Important: Ensure data_obj.num_nodes is present and correct before transforming
            if not hasattr(data_obj, 'num_nodes') or data_obj.num_nodes is None:
                warnings.warn(f"Data object {i} (name: {getattr(data_obj, 'graph_name_original', 'Unknown')}) "
                              "is missing 'num_nodes' attribute. Attempting to infer from data.x.shape[0].")
                if hasattr(data_obj, 'x') and data_obj.x is not None:
                    data_obj.num_nodes = torch.tensor(data_obj.x.shape[0], dtype=torch.long)
                else:
                    warnings.warn(f"Cannot infer num_nodes for data object {i}. Skipping.")
                    continue

            transformed_obj = transformer(data_obj)
            if transformed_obj:  # Transformer might return None if input data is problematic
                transformed_data_list.append(transformed_obj)
                identifier = getattr(data_obj, 'graph_name_original', f"{self.split}_graph_{i}")
                current_graph_identifiers.append(identifier)
                # Ensure the transformed object also carries the identifier if needed later by get()
                if hasattr(transformed_obj, 'graph_name_original') and not transformed_obj.graph_name_original:
                    transformed_obj.graph_name_original = identifier

        if self.pre_filter is not None:
            # Apply pre_filter to the transformed data
            # Ensure identifiers list is filtered along with data_list
            filtered_indices = [i for i, data in enumerate(transformed_data_list) if self.pre_filter(data)]
            transformed_data_list = [transformed_data_list[i] for i in filtered_indices]
            current_graph_identifiers = [current_graph_identifiers[i] for i in filtered_indices]
            print(f"Applied pre_filter, {len(transformed_data_list)} graphs remaining for {self.split}.")

        if self.pre_transform is not None:
            # Apply pre_transform to the already transformed (padded) data
            print(f"Applying pre_transform for {self.split}...")
            transformed_data_list = [self.pre_transform(data) for data in
                                     tqdm(transformed_data_list, desc="Pre-transforming (on padded data)")]

        if not transformed_data_list:
            warnings.warn(
                f"No data to save for split '{self.split}' after transformation. Saving empty dataset structure.")

        # Collate data_list into PyG's internal storage format
        # This creates self.data and self.slices for InMemoryDataset
        data, slices = self.collate(transformed_data_list)

        # Save the processed (padded, collated) data and identifiers
        # The path is self.processed_paths[0]
        torch.save((data, slices, current_graph_identifiers), self.processed_paths[0])
        print(
            f"Finished processing for '{self.split}'. Saved {len(transformed_data_list)} graphs to {self.processed_paths[0]}.")
        # self.graph_identifiers is loaded in __init__ from the saved file,
        # so no need to set it here directly after saving. It will be set on next load.

    def get(self, idx: int) -> Data:
        data = super().get(idx)
        if hasattr(self, 'graph_identifiers') and self.graph_identifiers and idx < len(self.graph_identifiers):
            data.graph_name = self.graph_identifiers[idx]
        else:
            data.graph_name = f"{self.split}_graph_idx_{idx}"
        return data

    def len(self) -> int:
        if not hasattr(self, 'slices') or not self.slices: return 0
        # Check if slices dictionary is empty
        if not self.slices: return 0
        slice_key = next(iter(self.slices.keys()), None)
        return self.slices[slice_key].size(0) - 1 if slice_key and self.slices[slice_key].numel() > 0 else 0

