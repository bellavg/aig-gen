import dgl
import numpy as np
import pydantic
import random
import torch
import yaml

from typing import Optional, Any  # Added Any for less strict typing if needed later


def set_seed(seed=0):
    if seed is None:
        return

    dgl.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DataLoaderYaml(pydantic.BaseModel):
    batch_size: int
    num_workers: int


class BiMPNNYaml(pydantic.BaseModel):
    x_n_emb_size: int
    pe_emb_size: Optional[int] = 0
    y_emb_size: Optional[int] = 0
    num_mpnn_layers: int
    pool: Optional[str] = None
    pe: Optional[str] = None


class OptimizerYaml(pydantic.BaseModel):
    lr: float
    amsgrad: bool


class NodeCountYaml(pydantic.BaseModel):
    loader: DataLoaderYaml
    model: BiMPNNYaml
    num_epochs: int
    optimizer: OptimizerYaml


class NodePredictorYaml(pydantic.BaseModel):
    t_emb_size: int
    out_hidden_size: int
    num_transformer_layers: int
    num_heads: int
    dropout: float


class NodePredYaml(pydantic.BaseModel):
    T: int
    loader: DataLoaderYaml
    num_epochs: int
    graph_encoder: BiMPNNYaml
    predictor: NodePredictorYaml
    optimizer: OptimizerYaml


class EdgePredictorYaml(pydantic.BaseModel):
    t_emb_size: int
    out_hidden_size: int


class EdgePredYaml(pydantic.BaseModel):
    T: int
    loader: DataLoaderYaml
    num_epochs: int
    graph_encoder: BiMPNNYaml
    predictor: EdgePredictorYaml
    optimizer: OptimizerYaml


class GeneralYaml(pydantic.BaseModel):
    dataset: str
    conditional: bool
    patience: Optional[int] = None
    # --- ADDED MISSING FIELDS ---
    path_to_pt_file: Optional[str] = None  # Make optional if not all datasets need it
    num_node_categories: Optional[int] = None  # Make optional if not all datasets need it

    # If these fields are *always* required when dataset is 'aig',
    # you might not make them Optional, but then ensure tpu_tile.yaml doesn't cause issues
    # or add specific validation logic. For now, Optional is safer.


class LayerDAGYaml(pydantic.BaseModel):
    general: GeneralYaml
    node_count: NodeCountYaml
    node_pred: NodePredYaml
    edge_pred: EdgePredYaml


def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    # Validate with Pydantic and convert to dict
    # Pydantic V2 uses model_dump(), V1 uses dict()
    # Assuming Pydantic V2 based on modern usage, otherwise change to .dict()
    try:
        # For Pydantic V2
        config_model = LayerDAGYaml(**yaml_data)
        return config_model.model_dump()
    except AttributeError:
        # Fallback for Pydantic V1
        config_model = LayerDAGYaml(**yaml_data)
        return config_model.dict()

