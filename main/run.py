import torch
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import h5py
from model import GCNLinkPredictor  # Ensure this is updated to handle edge_attr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class LinkPredictionDataset(Dataset):
    def __init__(self, edges, labels, distances, node_features):
        super(LinkPredictionDataset, self).__init__()
        self.edges = edges
        self.labels = labels
        self.distances = distances
        self.node_features = node_features

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        edge = self.edges[idx]
        label = self.labels[idx]
        distance = self.distances[idx]

        # Node indices involved in the edge
        node_indices = edge

        # Extract the features for the two nodes involved
        x = torch.tensor(self.node_features[node_indices], dtype=torch.float)

        # Edge indices within this small graph
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # Shape: [2, 1]

        # Edge attributes (distance)
        edge_attr = torch.tensor([distance], dtype=torch.float).unsqueeze(1)  # Shape: [1, 1]

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.float))

def collate_fn(batch):
    # Batch all the graphs together
    return torch_geometric.data.Batch.from_data_list(batch)

def load_split_data(split_file, node_features_file):
    # Load split data
    data = np.load(split_file)
    X_train = data['X_train']
    y_train = data['y_train']
    dist_train = data['dist_train']
    X_test = data['X_test']
    y_test = data['y_test']
    dist_test = data['dist_test']

    # Load node features
    with h5py.File(node_features_file, 'r') as nf:
        # Concatenate node features from all molecules
        all_node_features = []
        for molecule in nf.keys():
            node_features = nf[molecule][:]
            all_node_features.append(node_features)
        node_features = np.concatenate(all_node_features, axis=0)

    return X_train, y_train, dist_train, X_test, y_test, dist_test, node_features

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train GCN for Link Prediction with Distance Matrix')
    parser.add_argument('--data', type=str, default=r'C:\Users\DCR\Documents\University\project\GVAEs\Project\split_data.npz', 
                        help='Path to split data file with distances')
    parser.add_argument('--node_features', type=str, default=r'C:\Users\DCR\Documents\University\project\GVAEs\Project\data\raw_unique\node_features.h5', 
                        help='Path to node features file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Load split data including distances
    X_train, y_train, dist_train, X_test, y_test, dist_test, node_features = load_split_data(
        args.data, args.node_features
    )

    print(f"Train set: {len(X_train)} edges")
    print(f"Test set: {len(X_test)} edges")

    # Create Dataset and DataLoader
    train_dataset = LinkPredictionDataset(X_train, y_train, dist_train, node_features)
    test_dataset = LinkPredictionDataset(X_test, y_test, dist_test, node_features)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the model with edge_dim=1 (distance)
    model = GCNLinkPredictor(
        in_channels=node_features.shape[1],
        hidden_channels=args.hidden_channels,
        edge_dim=1  # Since the edge attribute is the distance (a scalar)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Training Phase
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc='Training', leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass with edge_attr
            out = model(batch.x, batch.edge_index, batch.edge_attr)

            # Compute loss
            loss = F.binary_cross_entropy(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Validation Phase
        model.eval()
        total_val_loss = 0

        # Lists to store all predictions and labels
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating', leave=False):
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                loss = F.binary_cross_entropy(out, batch.y.view(-1))
                total_val_loss += loss.item() * batch.num_graphs

                # Store probabilities
                all_probs.extend(out.cpu().numpy())

                # Convert probabilities to binary predictions (threshold = 0.5)
                preds = (out > 0.5).cpu().numpy()
                all_preds.extend(preds)

                # Store true labels
                all_labels.extend(batch.y.view(-1).cpu().numpy())

        avg_val_loss = total_val_loss / len(test_loader.dataset)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Compute Evaluation Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        roc_auc = roc_auc_score(all_labels, all_probs)

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1-Score: {f1:.4f}")
        print(f"Validation ROC-AUC: {roc_auc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'gcn_link_predictor_with_distance.pth')
    print("\nModel saved to 'gcn_link_predictor_with_distance.pth'")


if __name__ == "__main__":
    main()
