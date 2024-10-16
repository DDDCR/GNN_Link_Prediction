import h5py
import numpy as np
import itertools
from sklearn.model_selection import GroupShuffleSplit

def load_data(node_features_file, adj_matrix_file, distance_matrix_file):
    with h5py.File(node_features_file, 'r') as nf, \
         h5py.File(adj_matrix_file, 'r') as adj, \
         h5py.File(distance_matrix_file, 'r') as dist:
        data = []
        molecule_names = list(nf.keys())
        for idx, molecule in enumerate(molecule_names):
            node_features = nf[molecule][:]
            adj_matrix = adj[molecule][:]
            distance_matrix = dist[molecule][:]
            data.append((idx, molecule, node_features, adj_matrix, distance_matrix))
    return data

def prepare_edge_labels_with_distances(data):
    all_edges = []
    all_labels = []
    all_distances = []
    molecule_ids = []
    node_index_offset = 0  # Running index for nodes

    positive_count = 0
    negative_count = 0

    for idx, molecule, node_features, adj_matrix, distance_matrix in data:
        num_atoms = node_features.shape[0]
        node_indices = np.arange(node_index_offset, node_index_offset + num_atoms)
        node_index_offset += num_atoms

        # Generate all possible unique pairs (i < j)
        pairs = list(itertools.combinations(node_indices, 2))
        edges = []
        labels = []
        distances = []

        for i, j in pairs:
            i_local = i - node_indices[0]  # Local index within the molecule
            j_local = j - node_indices[0]

            label = adj_matrix[i_local, j_local]
            distance = distance_matrix[i_local, j_local]
            edges.append((i, j))
            labels.append(label)
            distances.append(distance)
            molecule_ids.append(idx)  # Use the index as the molecule ID

            if label == 1:
                positive_count += 1
            else:
                negative_count += 1

        all_edges.extend(edges)
        all_labels.extend(labels)
        all_distances.extend(distances)

    print(f"Total positive edges: {positive_count}")
    print(f"Total negative edges: {negative_count}")

    return (np.array(all_edges), np.array(all_labels), np.array(all_distances), np.array(molecule_ids))

def balance_dataset(all_edges, all_labels, all_distances, molecule_ids, negative_ratio=1):
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
    balanced_molecule_ids = molecule_ids[selected_indices]

    print(f"Balanced positive edges: {num_positive}")
    print(f"Balanced negative edges: {len(sampled_negative_indices)}")

    return balanced_edges, balanced_labels, balanced_distances, balanced_molecule_ids

def split_data_grouped(balanced_edges, balanced_labels, balanced_distances, molecule_ids, test_size=0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(balanced_edges, groups=molecule_ids))
    X_train, X_test = balanced_edges[train_idx], balanced_edges[test_idx]
    y_train, y_test = balanced_labels[train_idx], balanced_labels[test_idx]
    dist_train, dist_test = balanced_distances[train_idx], balanced_distances[test_idx]
    train_molecule_ids = molecule_ids[train_idx]
    test_molecule_ids = molecule_ids[test_idx]
    return (X_train, y_train, dist_train, train_molecule_ids), (X_test, y_test, dist_test, test_molecule_ids)




if __name__ == "__main__":
    # Load data
    data = load_data(r'C:\Users\DCR\Documents\University\project\Link Prediction\data\raw_unique\node_features.h5', 
                     r'C:\Users\DCR\Documents\University\project\Link Prediction\data\raw_unique\adj_matrix.h5',
                     r'C:\Users\DCR\Documents\University\project\Link Prediction\data\raw_unique\distance_matrix.h5')

    # Prepare edges, labels, distances, and molecule IDs
    all_edges, all_labels, all_distances, molecule_ids = prepare_edge_labels_with_distances(data)

    # Optionally, normalize distances
    all_distances = (all_distances - all_distances.min()) / (all_distances.max() - all_distances.min())

    # Balance the dataset
    balanced_edges, balanced_labels, balanced_distances, balanced_molecule_ids = balance_dataset(
        all_edges, all_labels, all_distances, molecule_ids, negative_ratio=1
    )

    # Split into train and test sets using grouped splitting
    (X_train, y_train, dist_train, train_molecule_ids), \
    (X_test, y_test, dist_test, test_molecule_ids) = split_data_grouped(
        balanced_edges, balanced_labels, balanced_distances, balanced_molecule_ids, test_size=0.2
    )

    print(f"Train set: {len(X_train)} edges")
    print(f"Test set: {len(X_test)} edges")

    # Save the split data for use in training
    np.savez('split_data_with_distances.npz', 
             X_train=X_train, y_train=y_train, dist_train=dist_train, train_molecule_ids=train_molecule_ids,
             X_test=X_test, y_test=y_test, dist_test=dist_test, test_molecule_ids=test_molecule_ids)