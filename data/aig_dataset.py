import os
import torch
import warnings
import os.path as osp
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.separate import separate

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

