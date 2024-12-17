import argparse
from AigDataset import AIGDataset  # Replace with the filename of your adapted AIGDataset class
  # Replace with the filename of your GraphDF implementation
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DenseDataLoader

import os
import torch
from .model.graphflow import GraphFlowModel


class Generator():
    r"""
    The method base class for graph generation. To write a new graph generation method, create a new class
    inheriting from this class and implement the functions.
    """

    def train_rand_gen(self, loader, *args, **kwargs):
        r"""
        Running training for random generation task.

        Args:
            loader: The data loader for loading training samples.
        """

        raise NotImplementedError("The function train_rand_gen is not implemented!")

    def run_rand_gen(self, *args, **kwargs):
        r"""
        Running graph generation for random generation task.
        """

        raise NotImplementedError("The function run_rand_gen is not implemented!")

    def train_prop_opt(self, *args, **kwargs):
        r"""
        Running training for property optimization task.
        """

        raise NotImplementedError("The function train_prop_opt is not implemented!")

    def run_prop_opt(self, *args, **kwargs):
        r"""
        Running graph generation for property optimization task.
        """

        raise NotImplementedError("The function run_prop_opt is not implemented!")

    def train_const_prop_opt(self, loader, *args, **kwargs):
        r"""
        Running training for constrained optimization task.

        Args:
            loader: The data loader for loading training samples.
        """

        raise NotImplementedError("The function train_const_prop_opt is not implemented!")

    def run_const_prop_opt(self, *args, **kwargs):
        r"""
        Running molecule optimization for constrained optimization task.
        """

        raise NotImplementedError("The function run_const_prop_opt is not implemented!")

class GraphDF(Generator):
    r"""
        The method class for GraphDF algorithm proposed in the paper `GraphDF: A Discrete Flow Model for Molecular Graph Generation <https://arxiv.org/abs/2102.01189>`_. This class provides interfaces for running random generation, property
        optimization, and constrained optimization with GraphDF algorithm. Please refer to the `example codes <https://github.com/divelab/DIG/tree/dig-stable/examples/ggraph/GraphDF>`_ for usage examples.
    """

    def __init__(self):
        super(GraphDF, self).__init__()
        self.model = None

    def get_model(self, task, args, checkpoint_path=None):
        if task == 'rand_gen':
            self.model = GraphFlowModel(args)
        else:
            raise ValueError('Task {} is not supported in GraphDF!'.format(task))
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))

    def load_pretrain_model(self, path):
        load_key = torch.load(path)
        for key in load_key.keys():
            if key in self.model.state_dict().keys():
                self.model.state_dict()[key].copy_(load_key[key].detach().clone())


    def train_rand_gen(self, loader, args):
        """
        Running training for random generation task.
        """
        self.get_model('rand_gen', args)
        self.model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                     weight_decay=args.wd)
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)

        for epoch in range(1, args.max_epochs + 1):
            total_loss = 0
            for batch, data_batch in enumerate(loader):
                optimizer.zero_grad()

                # Extract input features and adjacency matrices
                inp_node_features = data_batch.x  # (B, N, node_dim)
                inp_adj_features = data_batch.adj  # (B, N, N, edge_dim)

                # Create a mask for valid nodes (non-padded)
                mask = (inp_node_features != args.pad_value).all(dim=-1).float()  # (B, N)

                # Forward pass
                out_z = self.model(inp_node_features, inp_adj_features)

                # Mask the loss to exclude padded regions
                log_prob = self.model.dis_log_prob(out_z)  # (B, N)
                loss = (log_prob * mask).mean()  # Masked mean loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                print(f'Training iteration {batch} | loss {loss.item()}')

            avg_loss = total_loss / (batch + 1)
            print(f"Epoch {epoch} | Average loss {avg_loss}")

            if epoch % args.save_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(args.save_dir, f'rand_gen_ckpt_{epoch}.pth'))


def parse_arguments():
    """Parse command-line arguments for model configuration and training parameters."""
    parser = argparse.ArgumentParser(description="Train GraphDF on AIG dataset")

    # Dataset arguments
    parser.add_argument("--dataset_root", type=str, default="./", help="Root directory of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")

    # Model configuration
    parser.add_argument("--edge_dim", type=int, default=2, help="Number of edge features (INV, REG)")
    parser.add_argument("--nhid", type=int, default=128, help="Hidden dimension size in for the model")
    parser.add_argument("--nout", type=int, default=128, help="Hidden dimension size out for the model")
    parser.add_argument("--num_flow_layer", type=int, default=12, help="Number of flow transformations in the model")
    parser.add_argument("--num_rgcn_layer", type=int, default=3, help="Number of rgcn layers in the model")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument("--edge_unroll", type=int, default=15, help="Number of edges per node generated possibly")
    #TODO: made up values for edge unroll and for numrgcn and for nout check very important

    # Training parameters
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of training epochs")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval for saving model checkpoints")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--pad_value", type=int, default=-1, help="Padding value of data")
    parser.add_argument("--max_size", type=int, default=100, help="Maximum size of AIGs")

    return parser.parse_args()


def collate(data_list):
    """Collate graphs into a batch with dynamic padding."""
    PAD_VALUE = -1
    # Determine max input patterns across the batch
    max_input_patterns = max(data.x.size(1) for data in data_list)

    for data in data_list:
        num_nodes, feature_dim = data.x.size()
        pad_features = max_input_patterns - feature_dim

        # Pad node features dynamically
        if pad_features > 0:
            padding = torch.full((num_nodes, pad_features), PAD_VALUE, dtype=torch.float)
            data.x = torch.cat([data.x, padding], dim=1)

    return Batch.from_data_list(data_list)



def main():
    # Parse arguments
    args = parse_arguments()

    # Print configuration
    print("Training Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # Load dataset
    print("Loading dataset...")
    #TODO: big todo make sure the fucking truth tables are the right way...
    dataset = AIGDataset(root=args.dataset_root)

    # Use custom collate function for dynamic padding
    data_loader = DenseDataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    print(f"Loaded dataset with {len(dataset)} graphs.")

    # Initialize and train GraphDF
    graphdf = GraphDF()
    print("Starting training...")
    graphdf.train_rand_gen(loader=data_loader, args=args)
    print("Training completed!")

if __name__ == "__main__":
    main()