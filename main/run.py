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
from sklearn.preprocessing import StandardScaler

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
    return Data.from_data_list(batch)

def load_split_data(split_file):
    # Load split data
    data = np.load(split_file)
    X_train = data['X_train']
    y_train = data['y_train']
    dist_train = data['dist_train']
    train_molecule_ids = data['train_molecule_ids']
    X_test = data['X_test']
    y_test = data['y_test']
    dist_test = data['dist_test']
    test_molecule_ids = data['test_molecule_ids']

    return X_train, y_train, dist_train, train_molecule_ids, X_test, y_test, dist_test, test_molecule_ids


def load_node_features(node_features_file):
    with h5py.File(node_features_file, 'r') as nf:
        molecule_names = list(nf.keys())
        node_features_list = []
        node_index_offset = 0

        for idx, molecule in enumerate(molecule_names):
            node_features = nf[molecule][:]
            num_nodes = node_features.shape[0]
            node_features_list.append(node_features)
            node_index_offset += num_nodes

        node_features = np.concatenate(node_features_list, axis=0)

    return node_features

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train GCN for Link Prediction with Distance Matrix')
    parser.add_argument('--data', type=str, default=r'C:\Users\DCR\Documents\University\project\Link Prediction\main\split_data_with_distances.npz', 
                        help='Path to split data file with distances')
    parser.add_argument('--node_features', type=str, default=r'C:\Users\DCR\Documents\University\project\Link Prediction\data\raw_unique\node_features.h5', 
                        help='Path to node features file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    args = parser.parse_args()

    # Load split data including distances and molecule IDs
    X_train, y_train, dist_train, train_molecule_ids, X_test, y_test, dist_test, test_molecule_ids = load_split_data(
        args.data
    )

    # Load node features
    node_features = load_node_features(args.node_features)

    # Normalize node features using training data
    # Get node indices for training data
    train_node_indices = np.unique(X_train.flatten())

    # Fit scaler on training node features
    scaler = StandardScaler()
    train_node_features = node_features[train_node_indices]
    scaler.fit(train_node_features)

    # Transform all node features
    node_features = scaler.transform(node_features)

    # Check for overlapping edges and nodes
    train_edges_set = set(map(tuple, X_train))
    test_edges_set = set(map(tuple, X_test))
    common_edges = train_edges_set.intersection(test_edges_set)
    print(f"Number of overlapping edges between training and test sets: {len(common_edges)}")

    train_nodes = set(X_train.flatten())
    test_nodes = set(X_test.flatten())
    common_nodes = train_nodes.intersection(test_nodes)
    print(f"Number of overlapping nodes between training and test sets: {len(common_nodes)}")

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
    patience = args.patience  # Early stopping patience
    best_val_f1 = 0.0
    epochs_without_improvement = 0

    
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

        # Early Stopping Check
        if f1 > best_val_f1:
            best_val_f1 = f1
            epochs_without_improvement = 0
            # Save the model with the best validation F1 Score
            torch.save(model.state_dict(), 'best_model.pth')
            print("Validation F1 Score improved, model saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement in Validation F1 Score for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    # Save the trained model
    torch.save(model.state_dict(), 'gcn_link_predictor_with_distance.pth')
    print("\nModel saved to 'gcn_link_predictor_with_distance.pth'")


if __name__ == "__main__":
    main()