import argparse
from CondGraphDF.aig_dataset import AIGDataset
from torch_geometric.loader import DenseDataLoader
from CondGraphDF.graphdf import GraphDF
import numpy as np


def parse_arguments():
    """Parse command-line arguments for model configuration and training parameters."""
    parser = argparse.ArgumentParser(description="Train GraphDF on AIG dataset")

    # Dataset arguments
    parser.add_argument("--dataset_root", type=str, default="./", help="Root directory of the dataset")
    parser.add_argument("--dataset_size", type=int, default=5000, help="size of the dataset sampled randomly")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_max_nodes", type=int, default=120, help="Max number of nodes in the dataset")
    parser.add_argument("--edge_dim", type=int, default=2, help="Number of edge features (INV, REG)")

    parser.add_argument("--node_dim", type=int, default=4, help="Node feature dimension (e.g., 4 + 2^(number of PIs))")
    parser.add_argument("--condition_dim", type=int, default=1, help="Number of classes in the dataset")
    parser.add_argument("--edge_unroll", type=int, default=16, help="Max fanout (number of edges per node generated)")
    parser.add_argument("--pad_value", type=float, default=-np.inf, help="Padding value for dataset")
    parser.add_argument("--train_split_ratio", type=float, default=0.8, help="Ratio of dataset used for training")  # Added
    parser.add_argument("--raw_file_name", type=str, default="condition_graphs_0.pkl", help="Name of the raw dataset file")  # Added

    # Model configuration (GraphFlow)
    parser.add_argument("--nhid", type=int, default=128, help="Hidden dimension size in the model")
    parser.add_argument("--nout", type=int, default=128, help="Output dimension size of the model")
    parser.add_argument("--num_flow_layer", type=int, default=6, help="Number of flow transformations in the model")
    parser.add_argument("--num_rgcn_layer", type=int, default=3, help="Number of RGCN layers in the model")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for model regularization")  # Added
    parser.add_argument("--activation", type=str, default="relu", help="Activation function (e.g., relu, tanh)")  # Added

    # Training parameters
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of training epochs")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval for saving model checkpoints")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")  # Added
    parser.add_argument("--gradient_clip", type=float, default=None, help="Max gradient norm for clipping")  # Added

    # Evaluation and logging
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging metrics during training")  # Added
    parser.add_argument("--eval_interval", type=int, default=1, help="Interval (in epochs) for evaluation")  # Added

    # Hardware options
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")  # Added

    return parser.parse_args()




def main():
    #TODO
    # add truths, add num_inputs, num_outputs
    # remove virtual edge bs
    # add condition

    # Parse arguments
    args = parse_arguments()

    # Convert args to a configuration dictionary
    conf_dict = vars(args)  # Converts the Namespace object to a dictionary

    # Load dataset
    print("Loading dataset...")
    dataset = AIGDataset(root=conf_dict["dataset_root"], conf_dict=conf_dict)

    # Use custom collate function for dynamic padding
    data_loader = DenseDataLoader(dataset, batch_size=conf_dict["batch_size"], shuffle=True)
    print(f"Loaded dataset with {len(dataset)} graphs.")

    # Initialize and train GraphDF
    graphdf = GraphDF()
    print("Starting training...")
    graphdf.train_rand_gen(loader=data_loader, args=args)  # Pass args to training
    print("Training completed!")
    #TODO: make conditional with tts
    #TODO: make number of input and output nodes available
    #TODO: train then RL with validity etc.


if __name__ == "__main__":
    main()