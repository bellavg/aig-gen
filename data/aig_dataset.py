import os
import torch
import warnings
import os.path as osp
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.separate import separate


# Add this function to GraphDF/aig_dataset.py
import random
from collections import deque
import networkx as nx
import numpy as np
import torch
import warnings

def custom_randomized_topological_sort(G, random_generator):
    """
    Performs a topological sort, randomizing the order of nodes
    that have the same in-degree at each step.
    Uses the provided random_generator instance.
    Raises NetworkXUnfeasible if a cycle is detected.
    """
    if not G.is_directed():
        raise nx.NetworkXError("Topological sort not defined for undirected graphs.")

    in_degree_map = {node: degree for node, degree in G.in_degree()}
    # Nodes with in-degree 0 are the starting points
    zero_in_degree_nodes = [node for node, degree in in_degree_map.items() if degree == 0]

    # Shuffle the initial zero-degree nodes if augmentation is desired
    if len(zero_in_degree_nodes) > 1:
        random_generator.shuffle(zero_in_degree_nodes) # Use the passed generator

    queue = deque(zero_in_degree_nodes)
    result_order = []

    while queue:
        u = queue.popleft()
        result_order.append(u)

        # Find successors whose in-degree will become zero
        newly_zero_in_degree = []
        # Sort successors for deterministic iteration before potential shuffle
        for v in sorted(list(G.successors(u))):
            in_degree_map[v] -= 1
            if in_degree_map[v] == 0:
                newly_zero_in_degree.append(v)
            elif in_degree_map[v] < 0:
                # This indicates an issue in the graph structure or algorithm
                raise RuntimeError(f"In-degree became negative for node {v} during topological sort.")

        # Shuffle the newly zero-in-degree nodes for augmentation
        if len(newly_zero_in_degree) > 1:
            random_generator.shuffle(newly_zero_in_degree) # Use the passed generator

        # Add newly discovered zero-in-degree nodes to the queue
        for node in newly_zero_in_degree:
            queue.append(node)

    # Check if all nodes were included (if not, there's a cycle)
    if len(result_order) != G.number_of_nodes():
        # Raise the specific error NetworkX uses for cycles in topological sort
        raise nx.NetworkXUnfeasible(f"Graph contains a cycle. Topological sort cannot proceed.")

    return result_order



class AIGDatasetLoader(InMemoryDataset):
    """
    Loads pre-processed AIG data saved in the InMemoryDataset format
    (collated data and slices tuple). Assumes processing was done externally.
    """
    def __init__(self, root, name='aig', dataset_type="train", transform=None, pre_transform=None, pre_filter=None):

        self.name = name
        self.dataset_type = dataset_type

        # Call super() - it will look for processed files
        super(AIGDatasetLoader, self).__init__(root, transform, pre_transform, pre_filter)

        # Load the pre-processed data
        processed_path = self.processed_paths[0]
        if not osp.exists(processed_path):
             # If super() didn't trigger processing (e.g., because raw files were missing)
             # or if the file just isn't there.
             raise FileNotFoundError(f"Processed file not found at {processed_path}. "
                                     "Please ensure the processing script ran successfully and created the file.")

        try:
            # ***** MODIFIED LINE *****
            # Explicitly set weights_only=False to load PyG Data objects
            # Ensure you trust the source of the .pt file!
            loaded_data = torch.load(processed_path, weights_only=False)
            # ***** END MODIFICATION *****

            # Load the tuple (data, slices) saved by the processing script
            # Handle potential loading of the older tuple format without all_aig_ids
            if len(loaded_data) == 3:
                 self.data, self.slices, _ = loaded_data # Discard loaded IDs if any
            elif len(loaded_data) == 2:
                 self.data, self.slices = loaded_data
            else:
                 raise ValueError(f"Loaded tuple from {processed_path} has unexpected length {len(loaded_data)}")

            print(f"Successfully loaded pre-processed data from {processed_path}")

        except Exception as e:
            # Catch potential unpickling errors even with weights_only=False
            warnings.warn(f"Could not load processed file {processed_path}: {e}")
            self.data, self.slices = None, None


    @property
    def raw_file_names(self):
        # Raw files are not needed by this loader class
        return []

    @property
    def processed_dir(self):
        # Defines where to look for the pre-processed file
        return osp.join(self.root, self.name, 'processed', self.dataset_type)

    @property
    def processed_file_names(self):
        # The expected name of the pre-processed file
        return ['data.pt']

    def download(self):
        # No download needed, data should be pre-processed
        pass

    def process(self):
        # Processing is done externally by the dedicated script
        print(f"Dataset '{self.name}' type '{self.dataset_type}' expects pre-processed file at {self.processed_paths[0]}.")
        print("Run the separate processing script if this file does not exist.")
        if not osp.exists(self.processed_paths[0]):
            raise FileNotFoundError(f"Pre-processed file not found: {self.processed_paths[0]}. Run the processing script.")
        pass # Do not process here

    # get() and len() are standard for InMemoryDataset
    def get(self, idx: int) -> Data:
        if self.data is None or self.slices is None:
             raise RuntimeError(f"Dataset not loaded. Processed file might be missing or failed to load.")

        data = separate(
            cls=self.data.__class__, #type: ignore
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )
        # Add an identifier (optional)
        data.aig_id = f"{self.name}_{self.dataset_type}_graph_{idx}"

        # Ensure num_atom is scalar tensor
        if hasattr(data, 'num_atom') and isinstance(data.num_atom, torch.Tensor) and data.num_atom.ndim > 0 :
             data.num_atom = data.num_atom[0]
        elif not hasattr(data, 'num_atom'):
             # This case indicates an issue with the saved .pt file, as _process_graph should add it
             warnings.warn(f"Graph {idx} seems to be missing 'num_atom'. Setting to 0.")
             data.num_atom = torch.tensor(0, dtype=torch.long)

        return data

    def len(self) -> int:
        if hasattr(self, 'slices') and self.slices is not None:
            # Ensure slices is not empty and contains valid slice information
            for _, slice_info in self.slices.items():
                if isinstance(slice_info, torch.Tensor):
                    # The number of items is one less than the length of the slice tensor
                    return max(0, slice_info.numel() - 1)
                elif isinstance(slice_info, dict): # Older PyG versions might store slices differently
                     for _, value in slice_info.items():
                          return len(value) - 1 # Assuming slices are stored per attribute
        return 0


# Add this class to GraphDF/aig_dataset.py
from torch.utils.data import Dataset # Add this import if not already present

class AugmentedAIGDataset(Dataset):
    """
    A wrapper dataset that applies randomized topological sort augmentation
    to data loaded by AIGDatasetLoader.
    """
    def __init__(self, base_dataset: AIGDatasetLoader, num_augmentations: int = 1):
        super().__init__()
        if not isinstance(base_dataset, AIGDatasetLoader):
            raise TypeError("base_dataset must be an instance of AIGDatasetLoader")
        self.base_dataset = base_dataset
        self.num_augmentations = max(1, num_augmentations)
        self.original_len = len(self.base_dataset)
        # Inherit max_nodes from the base dataset if needed
        self.max_nodes = getattr(base_dataset, 'max_nodes', 64) # Default to 64 if not found
        print(f"Created AugmentedAIGDataset wrapper with {self.num_augmentations} augmentations per graph.")
        print(f"Original dataset length: {self.original_len}, Augmented length: {self.__len__()}")

    def __len__(self):
        # Return the augmented length
        return self.original_len * self.num_augmentations

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Index {idx} out of bounds for augmented dataset size {self.__len__()}")

        # Determine the original graph index and the augmentation seed
        graph_idx = idx // self.num_augmentations
        aug_seed = idx # Use the overall index as the seed for simplicity, or idx % self.num_augmentations

        # Create a local random generator for this specific augmentation
        local_random = random.Random(aug_seed)

        # Get the base data object using the original graph index
        try:
            data = self.base_dataset.get(graph_idx)
        except Exception as e:
             warnings.warn(f"Error getting base data for graph index {graph_idx} (from augmented index {idx}): {e}. Returning empty data.")
             # Return an empty Data object or handle appropriately
             return Data(x=torch.empty((0, self.base_dataset.data.x.size(-1))), # Match feature dim
                         adj=torch.empty((self.base_dataset.data.adj.size(0), 0, 0)), # Match adj channels
                         num_atom=torch.tensor(0, dtype=torch.long))


        num_actual_nodes = data.num_atom.item()
        apply_ordering = False
        ordered_nodes = []

        if num_actual_nodes > 0:
            try:
                # Reconstruct NetworkX graph
                adj_actual_sparse = data.adj[:2, :num_actual_nodes, :num_actual_nodes].sum(dim=0) > 0
                adj_actual_np = adj_actual_sparse.cpu().numpy()
                G_actual = nx.from_numpy_array(adj_actual_np, create_using=nx.DiGraph)

                # Apply YOUR custom randomized topological sort
                ordered_nodes = custom_randomized_topological_sort(G_actual, local_random)
                apply_ordering = True

            except nx.NetworkXUnfeasible:
                warnings.warn(f"Graph index {graph_idx} (aug idx {idx}) contains a cycle. Topological sort not possible. Returning original order.")
                # Do not set apply_ordering = True
            except Exception as e:
                warnings.warn(f"Error during graph reconstruction or custom topological sort for graph index {graph_idx} (aug idx {idx}): {e}. Returning original order.")
                # Do not set apply_ordering = True

            # Apply permutation ONLY if sort succeeded
            if apply_ordering:
                if len(ordered_nodes) != num_actual_nodes:
                     warnings.warn(f"Custom topological sort produced {len(ordered_nodes)} nodes, expected {num_actual_nodes}. Using original order.")
                else:
                    current_order = np.array(ordered_nodes)
                    padding_order = np.arange(num_actual_nodes, self.max_nodes)
                    full_perm = np.concatenate([current_order, padding_order])
                    full_perm_tensor = torch.from_numpy(full_perm).long()

                    # Apply permutation to x and adj tensors
                    data.x = data.x[full_perm_tensor]
                    data.adj = data.adj[:, full_perm_tensor][:, :, full_perm_tensor]

        # Return the (potentially) reordered data object
        return data