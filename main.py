import argparse
from GraphDF.AigDataset import AIGDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DenseDataLoader
from GraphDF.graphdf import GraphDF
import torch

def parse_arguments():
    """Parse command-line arguments for model configuration and training parameters."""
    parser = argparse.ArgumentParser(description="Train GraphDF on AIG dataset")

    # Dataset arguments
    parser.add_argument("--dataset_root", type=str, default="./data", help="Root directory of the dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--num_max_nodes", type=int, default=100, help="Max number of nodes in the dataset")
    parser.add_argument("--edge_dim", type=int, default=3, help="Number of edge features (INV, REG)")
    #TODO rememebr this is 2
    parser.add_argument("--max_pis", type=int, default=6, help="max number of primary inputs in AIG dataset")
    #parser.add_argument("--node_dim", type=int, default=260, help="4+2^(number of pis)")
    parser.add_argument("--node_dim", type=int, default=4, help="4+2^(number of pis)")
    parser.add_argument("--edge_unroll", type=int, default=14, help="Number of edges per node generated possibly, max fanout")
    parser.add_argument("--pad_value", type=int, default=-1, help="Padding value of data")

    # Model configuration  GraphFlow
    parser.add_argument("--nhid", type=int, default=128, help="Hidden dimension size in for the model")
    parser.add_argument("--nout", type=int, default=128, help="Hidden dimension size out for the model")
    parser.add_argument("--num_flow_layer", type=int, default=6, help="Number of flow transformations in the model")
    parser.add_argument("--num_rgcn_layer", type=int, default=2, help="Number of rgcn layers in the model")

    # Training parameters
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of training epochs")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval for saving model checkpoints")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")

    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")

    return parser.parse_args()


def main():
    #TODO
    # add truths,
    # add being able to see input and output nodes for masks,
    # remove virtual edge bs

    args = parse_arguments()

    # Print configuration
    print("Training Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # Load dataset
    print("Loading dataset...")
    #TODO: big todo make sure the fucking truth tables are the right way...
    #TODO: fix padding and choose padding values!
    dataset = AIGDataset(root=args.dataset_root, num_max_node=args.num_max_nodes, max_pis=args.max_pis)

    # Use custom collate function for dynamic padding
    data_loader = DenseDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Loaded dataset with {len(dataset)} graphs.")

    # Initialize and train GraphDF
    graphdf = GraphDF()
    print("Starting training...")
    graphdf.train_rand_gen(loader=data_loader, args=args)
    print("Training completed!")

if __name__ == "__main__":
    main()