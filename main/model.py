import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv

class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim):
        super(GCNLinkPredictor, self).__init__()
        # Use NNConv to incorporate edge attributes
        nn = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_channels * in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels * in_channels, hidden_channels * in_channels)
        )
        self.conv1 = NNConv(in_channels, hidden_channels, nn)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels * 2, 1)

    def forward(self, x, edge_index, edge_attr):
        # Apply convolution layers
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index)

        # Get embeddings for source and target nodes
        src = x[edge_index[0]]  # Shape: [num_edges, hidden_channels]
        dst = x[edge_index[1]]  # Shape: [num_edges, hidden_channels]

        # Concatenate node embeddings
        edge_features = torch.cat([src, dst], dim=1)  # Shape: [num_edges, hidden_channels * 2]

        # Compute edge scores
        edge_scores = self.lin(edge_features)  # Shape: [num_edges, 1]

        # Apply sigmoid to get probabilities
        edge_probs = torch.sigmoid(edge_scores).squeeze(-1)  # Shape: [num_edges]

        return edge_probs

