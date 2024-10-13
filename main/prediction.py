import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import h5py
from tqdm import tqdm
import argparse
from model import GCNLinkPredictor  # Ensure this path is correct
import matplotlib.pyplot as plt
import seaborn as sns

class LinkPredictionDataset(Dataset):
    def __init__(self, node_features, distance_matrix):
        super(LinkPredictionDataset, self).__init__()
        self.node_features = node_features
        self.distance_matrix = distance_matrix
        self.num_atoms = node_features.shape[0]
        self.edges = self.generate_all_possible_edges()
    
    def generate_all_possible_edges(self):
        # Generate all unique pairs (i < j)
        edges = []
        for i in range(self.num_atoms):
            for j in range(i + 1, self.num_atoms):
                edges.append((i, j))
        return edges
    
    def len(self):
        return len(self.edges)
    
    def __getitem__(self, idx):
        edge = self.edges[idx]
        distance = self.distance_matrix[edge[0], edge[1]]
        
        # Extract node features for the two atoms
        x = torch.tensor(self.node_features[list(edge)], dtype=torch.float)
        
        # Define edge_index for the small graph (two nodes)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # Shape: [2, 1]
        
        # Edge attribute (distance)
        edge_attr = torch.tensor([distance], dtype=torch.float).unsqueeze(1)  # Shape: [1, 1]
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def load_model(model_path, in_channels, hidden_channels, edge_dim, device):
    model = GCNLinkPredictor(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        edge_dim=edge_dim
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_ligand_data(node_features_path, distance_matrix_path):
    with h5py.File(node_features_path, 'r') as nf, h5py.File(distance_matrix_path, 'r') as dm:
        # Assuming the ligand data is stored under a single key, e.g., 'ligand'
        ligand_key = list(nf.keys())[0]
        node_features = nf[ligand_key][:]
        distance_matrix = dm[ligand_key][:]
    return node_features, distance_matrix

def predict_bonds(model, dataset, batch_size, device, threshold=0.5):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting Bonds'):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Convert outputs to probabilities
            probs = out.cpu().numpy()
            all_probs.extend(probs)

            # Convert probabilities to binary predictions based on threshold
            preds = (out > threshold).cpu().numpy()
            all_preds.extend(preds)
    
    return all_probs, all_preds

def build_adjacency_matrix(num_atoms, predicted_edges, predictions, probabilities):
    # Initialize adjacency matrices
    adj_matrix_probs = np.zeros((num_atoms, num_atoms), dtype=float)
    adj_matrix_preds = np.zeros((num_atoms, num_atoms), dtype=int)
    
    idx = 0
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            adj_matrix_probs[i, j] = probabilities[idx]
            adj_matrix_probs[j, i] = probabilities[idx]  # Assuming undirected bonds
            
            adj_matrix_preds[i, j] = predictions[idx]
            adj_matrix_preds[j, i] = predictions[idx]  # Assuming undirected bonds
            idx += 1
    return adj_matrix_probs, adj_matrix_preds


def save_adjacency_matrix(adj_matrix, output_path):
    np.save(output_path, adj_matrix)
    print(f"Predicted adjacency matrix saved to {output_path}")

def display_adjacency_matrix(adj_matrix, title='Adjacency Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj_matrix, annot=False, cmap='viridis')
    plt.title(title)
    plt.xlabel('Atom Index')
    plt.ylabel('Atom Index')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict Bonds in a New Ligand using Trained GCN Model')
    parser.add_argument('--model_path', type=str, default= r'C:\Users\DCR\Documents\University\project\Link Prediction\gcn_link_predictor_with_distance.pth',
                        help='Path to the trained model file (e.g., gcn_link_predictor_with_distance.pth)')
    parser.add_argument('--node_features', type=str, default= r'C:\Users\DCR\Documents\University\project\Link Prediction\data\AHG_raw\node_features.h5', 
                        help='Path to the new node features HDF5 file (e.g., new_node_features.h5)')
    parser.add_argument('--distance_matrix', type=str, default= r'C:\Users\DCR\Documents\University\project\Link Prediction\data\AHG_raw\distance_matrix.h5', 
                        help='Path to the new distance matrix HDF5 file (e.g., new_distance_matrix.h5)')
    parser.add_argument('--output_probs', type=str, default= r'C:\Users\DCR\Documents\University\project\Link Prediction\AHG_prediction\predicted_adj_matrix_probs.npy', 
                        help='Path to save the predicted adjacency matrix with probabilities (.npy file)')
    parser.add_argument('--output_preds', type=str, default= r'C:\Users\DCR\Documents\University\project\Link Prediction\AHG_prediction\predicted_adj_matrix_preds.npy', 
                        help='Path to save the predicted adjacency matrix with binary predictions (.npy file)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for predictions')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Threshold to convert probabilities to binary predictions')
    parser.add_argument('--display', action='store_true', default= True,
                        help='Whether to display the adjacency matrices as heatmaps')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load ligand data
    print("Loading ligand data...")
    node_features, distance_matrix = load_ligand_data(args.node_features, args.distance_matrix)
    num_atoms = node_features.shape[0]
    print(f"Number of atoms in the ligand: {num_atoms}")
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = LinkPredictionDataset(node_features, distance_matrix)
    
    # Load model
    print("Loading trained model...")
    in_channels = node_features.shape[1]
    hidden_channels = 64  # Ensure this matches the hidden_channels used during training
    edge_dim = 1  # Since distance is a scalar
    model = load_model(args.model_path, in_channels, hidden_channels, edge_dim, device)
    
    # Predict bonds
    print("Predicting bonds...")
    probabilities, predictions = predict_bonds(model, dataset, args.batch_size, device, threshold=args.threshold)
    
    # Build adjacency matrices
    print("Building adjacency matrices...")
    adj_matrix_probs, adj_matrix_preds = build_adjacency_matrix(num_atoms, dataset.edges, predictions, probabilities)
    
    # Save adjacency matrices
    print("Saving adjacency matrices...")
    save_adjacency_matrix(adj_matrix_probs, args.output_probs)
    save_adjacency_matrix(adj_matrix_preds, args.output_preds)
    
    # Optionally, display the adjacency matrices (defult is True)
    if args.display:
        print("Displaying adjacency matrix with probabilities...")
        display_adjacency_matrix(adj_matrix_probs, title='Predicted Adjacency Matrix with Bond Probabilities')
        
        print("Displaying adjacency matrix with binary predictions...")
        display_adjacency_matrix(adj_matrix_preds, title='Predicted Adjacency Matrix with Binary Predictions')
    
    print("Prediction complete.")

if __name__ == "__main__":
    main()
