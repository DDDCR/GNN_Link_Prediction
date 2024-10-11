import h5py
import numpy as np
import itertools
from sklearn.model_selection import train_test_split

def load_data(node_features_file, adj_matrix_file, distance_matrix_file):
    with h5py.File(node_features_file, 'r') as nf, \
         h5py.File(adj_matrix_file, 'r') as adj, \
         h5py.File(distance_matrix_file, 'r') as dist:
        data = []
        for molecule in nf.keys():
            node_features = nf[molecule][:]
            adj_matrix = adj[molecule][:]
            distance_matrix = dist[molecule][:]
            data.append((molecule, node_features, adj_matrix, distance_matrix))
    return data

def prepare_edge_labels(data):
    all_edges = []
    all_labels = []
    all_distances = []
    positive_count = 0
    negative_count = 0

    for molecule, node_features, adj_matrix, distance_matrix in data:
        num_atoms = node_features.shape[0]
        # Generate all possible unique pairs (i < j)
        pairs = list(itertools.combinations(range(num_atoms), 2))
        edges = []
        labels = []
        distances = []

        for i, j in pairs:
            label = adj_matrix[i, j]
            distance = distance_matrix[i, j]
            edges.append((i, j))
            labels.append(label)
            distances.append(distance)
            if label == 1:
                positive_count += 1
            else:
                negative_count += 1

        all_edges.extend(edges)
        all_labels.extend(labels)
        all_distances.extend(distances)

    print(f"Total positive edges: {positive_count}")
    print(f"Total negative edges: {negative_count}")

    return np.array(all_edges), np.array(all_labels), np.array(all_distances)


def balance_dataset(all_edges, all_labels, all_distances, negative_ratio=1):
    # Separate positive and negative samples
    positive_indices = np.where(all_labels == 1)[0]
    negative_indices = np.where(all_labels == 0)[0]

    num_positive = len(positive_indices)
    num_negative = negative_ratio * num_positive

    if len(negative_indices) > num_negative:
        sampled_negative_indices = np.random.choice(negative_indices, num_negative, replace=False)
    else:
        sampled_negative_indices = negative_indices

    # Combine positive and sampled negative indices
    selected_indices = np.concatenate([positive_indices, sampled_negative_indices])
    np.random.shuffle(selected_indices)

    balanced_edges = all_edges[selected_indices]
    balanced_labels = all_labels[selected_indices]
    balanced_distances = all_distances[selected_indices]

    print(f"Balanced positive edges: {num_positive}")
    print(f"Balanced negative edges: {len(sampled_negative_indices)}")

    return balanced_edges, balanced_labels, balanced_distances


def split_data(balanced_edges, balanced_labels, balanced_distances, test_size=0.2):
    X_train, X_test, y_train, y_test, dist_train, dist_test = train_test_split(
        balanced_edges, balanced_labels, balanced_distances, test_size=test_size, random_state=42, stratify=balanced_labels
    )
    return (X_train, y_train, dist_train), (X_test, y_test, dist_test)


if __name__ == "__main__":
    # Load data
    data = load_data(r'C:\Users\DCR\Documents\University\project\GVAEs\Project\data\raw_unique\node_features.h5', 
                     r'C:\Users\DCR\Documents\University\project\GVAEs\Project\data\raw_unique\adj_matrix.h5',
                     r'C:\Users\DCR\Documents\University\project\GVAEs\Project\data\raw_unique\distance_matrix.h5')

    # Prepare edges and labels
    all_edges, all_labels, all_distances = prepare_edge_labels(data)

    # Balance the dataset
    balanced_edges, balanced_labels, balenced_distances = balance_dataset(all_edges, all_labels, all_distances, negative_ratio=1)

    # Split into train and test sets
    (X_train, y_train, dist_train), (X_test, y_test, dist_test) = split_data(balanced_edges, balanced_labels, balenced_distances, test_size=0.2)

    print(f"Train set: {len(X_train)} edges")
    print(f"Test set: {len(X_test)} edges")

    # Save the split data for use in training
    np.savez('split_data.npz', 
             X_train=X_train, y_train=y_train, dist_train=dist_train,
             X_test=X_test, y_test=y_test, dist_test=dist_test)
