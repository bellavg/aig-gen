import os
import json
import argparse
import torch
from torch_geometric.loader import DenseDataLoader # Changed from .data import DenseDataLoader
import warnings
import os.path as osp

# --- Assume ggraph and your AIG dataset loader are importable ---
try:
    from data.aig_dataset import AIGDatasetLoader
    from GraphDF import GraphDF # Assuming these are in the current dir or PYTHONPATH
    from GraphAF import GraphAF
    from GraphEBM import GraphEBM
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'ggraph' (including dig.ggraph) is installed, your AIGDatasetLoader class is accessible, "
          "and GraphDF/GraphAF/GraphEBM models are correctly placed and importable.")
    exit(1)
# --- End Imports ---


# --- Base Configuration Dictionary ---
base_conf = {
    "data_name": "aig",
    "model": { # For GraphAF/DF and common EBM params
        "max_size": 64,    # Corresponds to n_atom for EBM
        "node_dim": 4,     # Corresponds to n_atom_type for EBM
        "bond_dim": 3,     # Corresponds to n_edge_type for EBM
        "use_gpu": True,

        # GraphDF/GraphAF specific parameters
        "edge_unroll": 12,
        "num_flow_layer": 12,
        "num_rgcn_layer": 3,
        "nhid": 128,
        "nout": 128,
        "deq_coeff": 0.9,
        "st_type": "exp",
        "use_df": False
    },
    "model_ebm": { # GraphEBM specific model parameters
        "hidden": 64,      # Hidden dim for EnergyFunc's GraphConv layers
        "depth": 2,        # Number of additional GraphConv layers (L=3 total with graphconv1)
        "swish_act": True, # Use Swish activation in EnergyFunc
        "add_self": False, # Add self-connections in GraphConv
        "dropout": 0.0,    # Dropout rate in EnergyFunc
        "n_power_iterations": 1 # Power iterations for Spectral Norm
    },
    "lr": 0.001,
    "weight_decay": 0,
    "batch_size": 128,
    "max_epochs": 50,
    "save_interval": 5,
    "save_dir": "GraphDF/rand_gen_aig_ckpts", # Will be overwritten

    "train_ebm": { # GraphEBM specific training parameters
        "c": 0.0,
        "ld_step": 150,
        "ld_noise": 0.005,
        "ld_step_size": 30,
        "clamp_lgd_grad": True, # Changed from "clamp"
        "alpha": 1.0
    }
}
# --- End Base Configuration ---


def main(args):
    conf = base_conf.copy()
    conf['model'] = base_conf['model'].copy()
    conf['model_ebm'] = base_conf['model_ebm'].copy()
    conf['train_ebm'] = base_conf['train_ebm'].copy()

    # General training params
    conf['lr'] = args.lr
    conf['weight_decay'] = args.weight_decay
    conf['batch_size'] = args.batch_size
    conf['max_epochs'] = args.max_epochs
    conf['save_interval'] = args.save_interval

    # GraphAF/GraphDF specific model params
    conf['model']['edge_unroll'] = args.edge_unroll
    conf['model']['num_flow_layer'] = args.num_flow_layer
    conf['model']['num_rgcn_layer'] = args.num_rgcn_layer
    conf['model']['nhid'] = args.gaf_nhid
    conf['model']['nout'] = args.gaf_nout
    conf['model']['deq_coeff'] = args.deq_coeff
    conf['model']['st_type'] = args.st_type

    # GraphEBM specific model params from args
    conf['model_ebm']['hidden'] = args.ebm_hidden
    conf['model_ebm']['depth'] = args.ebm_depth
    conf['model_ebm']['swish_act'] = args.ebm_swish_act
    conf['model_ebm']['add_self'] = args.ebm_add_self
    conf['model_ebm']['dropout'] = args.ebm_dropout
    conf['model_ebm']['n_power_iterations'] = args.ebm_n_power_iterations


    # GraphEBM specific training params from args
    conf['train_ebm']['c'] = args.ebm_c
    conf['train_ebm']['ld_step'] = args.ebm_ld_step
    conf['train_ebm']['ld_noise'] = args.ebm_ld_noise
    conf['train_ebm']['ld_step_size'] = args.ebm_ld_step_size
    conf['train_ebm']['alpha'] = args.ebm_alpha
    conf['train_ebm']['clamp_lgd_grad'] = args.ebm_clamp_lgd_grad # Updated to use the new arg

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A - CUDA not fully initialized?'}")
    else:
        device = torch.device('cpu')
        print("Using CPU.")
    conf['model']['use_gpu'] = (device.type == 'cuda')

    print("Instantiating AIGDatasetLoader...")
    try:
        dataset = AIGDatasetLoader(
            root=args.data_root,
            name=conf.get('data_name', 'aig'),
            dataset_type="train"
        )
        print(f"Number of training graphs loaded: {len(dataset)}")
        if len(dataset) == 0:
            print(f"Error: Training dataset is empty. Check path: "
                  f"'{osp.join(args.data_root, conf.get('data_name', 'aig'), 'processed', 'train', 'data.pt')}'")
            exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure --data_root points to the correct directory.")
        exit(1)
    except Exception as e:
        print(f"Error instantiating AIGDatasetLoader: {e}")
        exit(1)

    loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)
    print(f"Created DataLoader with batch size {conf['batch_size']}.")

    print(f"Instantiating model: {args.model_type}")
    runner = None
    if args.model_type == 'GraphDF':
        runner = GraphDF() # Assuming GraphDF() takes no args or uses a global config
    elif args.model_type == 'GraphAF':
        runner = GraphAF() # Assuming GraphAF() takes no args or uses a global config
    elif args.model_type == 'GraphEBM':
        try:
            runner = GraphEBM(
                n_atom=conf['model']['max_size'],
                n_atom_type=conf['model']['node_dim'],
                n_edge_type=conf['model']['bond_dim'],
                hidden=conf['model_ebm']['hidden'],
                depth=conf['model_ebm']['depth'],
                swish_act=conf['model_ebm']['swish_act'],
                add_self=conf['model_ebm']['add_self'],
                dropout=conf['model_ebm']['dropout'],
                n_power_iterations=conf['model_ebm']['n_power_iterations'],
                device=device
            )
        except KeyError as e:
            print(f"Error: Missing required parameter {e} in config for GraphEBM initialization.")
            exit(1)
        except Exception as e:
            print(f"Error instantiating GraphEBM: {e}")
            exit(1)
    else:
        print(f"Error: Unknown model type '{args.model_type}'.")
        exit(1)

    if runner is None:
        print(f"Failed to instantiate model runner for {args.model_type}")
        exit(1)

    save_dir = f"{args.model_type}/rand_gen_{conf.get('data_name', 'aig')}_ckpts"
    conf['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model checkpoints will be saved in: {save_dir}")

    print(f"Starting training for {args.model_type}...")
    if args.model_type == 'GraphDF' or args.model_type == 'GraphAF':
        # Assuming these models have a train_rand_gen method with this signature
        runner.train_rand_gen(
            loader=loader,
            lr=conf['lr'],
            wd=conf['weight_decay'],
            max_epochs=conf['max_epochs'],
            model_conf_dict=conf['model'],
            save_interval=conf['save_interval'],
            save_dir=conf['save_dir']
        )
    elif args.model_type == 'GraphEBM':
        try:
            runner.train_rand_gen(
                loader=loader,
                lr=conf['lr'],
                wd=conf['weight_decay'],
                max_epochs=conf['max_epochs'],
                c=conf['train_ebm']['c'],
                ld_step=conf['train_ebm']['ld_step'],
                ld_noise=conf['train_ebm']['ld_noise'],
                ld_step_size=conf['train_ebm']['ld_step_size'],
                clamp_lgd_grad=conf['train_ebm']['clamp_lgd_grad'], # Corrected argument name
                alpha=conf['train_ebm']['alpha'],
                save_interval=conf['save_interval'],
                save_dir=conf['save_dir']
            )
        except KeyError as e:
            print(f"Error: Missing required parameter {e} in 'train_ebm' config for GraphEBM training.")
            exit(1)
        except AttributeError as e:
            print(f"Error: {args.model_type} runner does not have a compatible train_rand_gen method or "
                  f"there's an issue with its parameters: {e}")
            exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during GraphEBM training: {e}")
            exit(1)

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train graph generation models on pre-processed AIG dataset.")

    parser.add_argument('--model_type', type=str, default='GraphAF',
                        choices=['GraphDF', 'GraphAF', 'GraphEBM'],
                        help='Model to train.')
    parser.add_argument('--data_root', default="./data/",
                        help="Root directory for data.")
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device for training.')

    # General Training Hyperparameters
    parser.add_argument('--lr', type=float, default=base_conf['lr'])
    parser.add_argument('--weight_decay', type=float, default=base_conf['weight_decay'])
    parser.add_argument('--batch_size', type=int, default=base_conf['batch_size'])
    parser.add_argument('--max_epochs', type=int, default=base_conf['max_epochs'])
    parser.add_argument('--save_interval', type=int, default=base_conf['save_interval'])

    # GraphAF/GraphDF Specific
    parser.add_argument('--edge_unroll', type=int, default=base_conf['model']['edge_unroll'])
    parser.add_argument('--num_flow_layer', type=int, default=base_conf['model']['num_flow_layer'])
    parser.add_argument('--num_rgcn_layer', type=int, default=base_conf['model']['num_rgcn_layer'])
    parser.add_argument('--gaf_nhid', type=int, default=base_conf['model']['nhid'])
    parser.add_argument('--gaf_nout', type=int, default=base_conf['model']['nout'])
    parser.add_argument('--deq_coeff', type=float, default=base_conf['model']['deq_coeff'])
    parser.add_argument('--st_type', type=str, default=base_conf['model']['st_type'], choices=['exp', 'sigmoid', 'softplus'])

    # GraphEBM Model Hyperparameters
    parser.add_argument('--ebm_hidden', type=int, default=base_conf['model_ebm']['hidden'],
                        help="Hidden dimension for GraphEBM's EnergyFunc.")
    parser.add_argument('--ebm_depth', type=int, default=base_conf['model_ebm']['depth'],
                        help="Depth of GraphEBM's EnergyFunc (number of extra GCN layers). L=depth+1.")
    # For boolean flags like swish_act, add_self for EnergyFunc
    parser.add_argument('--ebm_swish_act', action=argparse.BooleanOptionalAction, default=base_conf['model_ebm']['swish_act'],
                        help="Use Swish activation in GraphEBM's EnergyFunc.")
    parser.add_argument('--ebm_add_self', action=argparse.BooleanOptionalAction, default=base_conf['model_ebm']['add_self'],
                        help="Use self-connections in GraphEBM's GraphConv layers.")
    parser.add_argument('--ebm_dropout', type=float, default=base_conf['model_ebm']['dropout'],
                        help="Dropout rate in GraphEBM's EnergyFunc.")
    parser.add_argument('--ebm_n_power_iterations', type=int, default=base_conf['model_ebm']['n_power_iterations'],
                        help="Number of power iterations for Spectral Norm in GraphEBM.")


    # GraphEBM Training Hyperparameters
    parser.add_argument('--ebm_c', type=float, default=base_conf['train_ebm']['c'])
    parser.add_argument('--ebm_ld_step', type=int, default=base_conf['train_ebm']['ld_step'])
    parser.add_argument('--ebm_ld_noise', type=float, default=base_conf['train_ebm']['ld_noise'])
    parser.add_argument('--ebm_ld_step_size', type=float, default=base_conf['train_ebm']['ld_step_size'])
    parser.add_argument('--ebm_alpha', type=float, default=base_conf['train_ebm']['alpha'])
    # For ebm_clamp_lgd_grad (boolean flag)
    parser.add_argument('--ebm_clamp_lgd_grad', action=argparse.BooleanOptionalAction,
                        default=base_conf['train_ebm']['clamp_lgd_grad'],
                        help='Enable/disable gradient clamping during Langevin dynamics for GraphEBM.')


    args = parser.parse_args()
    main(args)
