import os
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
import wandb


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch):
    # Convert node features to dense batch format
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float() # Keep as boolean unless needed otherwise

    # Remove self-loops from edges, if any
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)

    # Determine the maximum number of nodes in the batch for dense tensor creation
    max_num_nodes = X.size(1)

    # Convert edges to dense adjacency tensor format
    # edge_attr (bs * num_edges, num_edge_features) -> E (bs, max_n, max_n, num_edge_features)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)

    # Ensure the 'no edge' feature is encoded correctly (typically index 0)
    E = encode_no_edge(E)

    # --- ADDED SYMMETRIZATION ---
    # Ensure the edge feature tensor is symmetric.
    # This averages the features between (i, j) and (j, i).
    # Required because the assertion in PlaceHolder.mask expects symmetry.
    E = 0.5 * (E + torch.transpose(E, 1, 2))
    # --- END SYMMETRIZATION ---


    # Return dense representation and node mask
    # Note: y (global features) are not handled here, assumed to be added later if needed.
    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    """
    Ensures that the first feature dimension (index 0) is 1 for non-edges
    and 0 for edges, and sets diagonal elements to 0.
    Assumes E has shape (bs, n, n, de).
    """
    # Check if edge features exist
    if E.shape[-1] == 0:
        return E # Return as is if no edge features

    # Identify non-edges: where the sum of features across the last dimension is 0
    # This assumes that valid edges always have at least one non-zero feature.
    # Clone E to avoid modifying the input tensor directly if it's used elsewhere,
    # though to_dense_adj usually creates a new tensor.
    E_encoded = E.clone()
    no_edge = torch.sum(E_encoded, dim=3) == 0

    # Set the first feature channel to 1 for non-edges
    first_elt = E_encoded[:, :, :, 0]
    first_elt[no_edge] = 1
    E_encoded[:, :, :, 0] = first_elt

    # Ensure diagonal elements represent no edge (or handle as needed)
    # Create a diagonal mask
    diag_mask = torch.eye(E.shape[1], dtype=torch.bool, device=E.device).unsqueeze(0).expand(E.shape[0], -1, -1)
    # Set diagonal elements to 0 across all feature dimensions
    E_encoded[diag_mask] = 0

    return E_encoded


def update_config_with_new_keys(cfg, saved_cfg):
    # Ensure saved_cfg attributes exist before accessing
    saved_general = getattr(saved_cfg, 'general', None)
    saved_train = getattr(saved_cfg, 'train', None)
    saved_model = getattr(saved_cfg, 'model', None)

    if saved_general:
        for key, val in saved_general.items():
            OmegaConf.set_struct(cfg.general, True)
            with open_dict(cfg.general):
                if key not in cfg.general.keys():
                    setattr(cfg.general, key, val)

    if saved_train:
        OmegaConf.set_struct(cfg.train, True)
        with open_dict(cfg.train):
            for key, val in saved_train.items():
                if key not in cfg.train.keys():
                    setattr(cfg.train, key, val)

    if saved_model:
        OmegaConf.set_struct(cfg.model, True)
        with open_dict(cfg.model):
            for key, val in saved_model.items():
                if key not in cfg.model.keys():
                    setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        if self.y is not None: # Handle cases where y might be None
            self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        # Ensure node_mask is boolean
        node_mask = node_mask.bool()

        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
        combined_edge_mask = (e_mask1 & e_mask2)  # bs, n, n, 1 (use boolean ops)

        if collapse:
            # Only collapse if features exist
            if self.X.numel() > 0:
                self.X = torch.argmax(self.X, dim=-1)
                self.X[~node_mask] = -1 # Use boolean mask directly
            else:
                # Handle case with no node features (e.g., set to -1 or keep empty)
                # This depends on downstream expectations. Setting to -1 might be safest.
                self.X = torch.full(node_mask.shape, -1, dtype=torch.long, device=node_mask.device)


            if self.E.numel() > 0:
                self.E = torch.argmax(self.E, dim=-1)
                # Use the combined boolean mask, remove the squeeze
                self.E[~combined_edge_mask.squeeze(-1)] = -1
            else:
                 # Handle case with no edge features
                 self.E = torch.full(combined_edge_mask.squeeze(-1).shape, -1, dtype=torch.long, device=node_mask.device)

        else:
            # Apply mask for non-collapse case
            if self.X.numel() > 0:
                self.X = self.X * x_mask.float() # Multiply by float mask
            if self.E.numel() > 0:
                self.E = self.E * combined_edge_mask.float() # Multiply by float mask

                # The assertion should now pass because E was symmetrized in to_dense
                # Adding a tolerance with torch.allclose is generally safer than direct comparison
                # especially after potential floating point operations.
                assert torch.allclose(self.E, torch.transpose(self.E, 1, 2), atol=1e-6), "E tensor is not symmetric after masking"
        return self


def setup_wandb(cfg):
    # Ensure wandb attribute exists
    wandb_mode = getattr(cfg.general, 'wandb', 'disabled') # Default to disabled

    if wandb_mode != 'disabled':
        try:
            import wandb # Import here to avoid issues if wandb is not installed
            config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            kwargs = {'name': cfg.general.name,
                      'project': f'graph_ddm_{cfg.dataset.name}',
                      'config': config_dict,
                      'settings': wandb.Settings(_disable_stats=True),
                      'reinit': True,
                      'mode': wandb_mode}
            wandb.init(**kwargs)
            wandb.save('*.txt') # Saves files matching pattern in the run directory
        except ImportError:
            print("Wandb is enabled in config, but the 'wandb' library is not installed. Skipping Wandb setup.")
        except Exception as e:
            print(f"Error setting up Wandb: {e}. Skipping Wandb setup.")

