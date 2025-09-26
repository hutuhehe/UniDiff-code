import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import pandas as pd

def kl_divergence(p, q, epsilon=1e-10):
    """Compute KL divergence between two distributions."""
    p = np.clip(p, epsilon, None)  # Avoid division by zero
    q = np.clip(q, epsilon, None)
    return np.sum(p * np.log(p / q))

def earth_mover_distance(train_emb, test_emb):
    """Compute Earth Mover’s Distance (Wasserstein Distance) between train and test."""
    distance_matrix = cdist(train_emb, test_emb, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    emd = distance_matrix[row_ind, col_ind].mean()
    return emd

def mean_nearest_neighbor_distance(train_emb, test_emb):
    """Compute mean nearest neighbor distance of test points to training points."""
    nbrs = NearestNeighbors(n_neighbors=1).fit(train_emb)
    distances, _ = nbrs.kneighbors(test_emb)
    return distances.mean()

def overlapping_ratio(train_emb, test_emb, threshold=5.0):
    """Compute Overlapping Ratio (fraction of test points close to a training point)."""
    nbrs = NearestNeighbors(n_neighbors=1).fit(train_emb)
    distances, _ = nbrs.kneighbors(test_emb)
    return np.mean(distances < threshold)

def compute_alignment_metrics(train_emb, test_emb):
    """Compute all alignment metrics between training and test embeddings."""
    
    # Convert to probability distributions (histograms)
    hist_train, _ = np.histogramdd(train_emb, bins=50, density=True)
    hist_test, _ = np.histogramdd(test_emb, bins=50, density=True)
    
    # KL Divergence
    kl_div = kl_divergence(hist_train.flatten(), hist_test.flatten())

    # Earth Mover’s Distance (EMD)
    emd = earth_mover_distance(train_emb, test_emb)

    # Mean Nearest Neighbor Distance (m-NND)
    m_nnd = mean_nearest_neighbor_distance(train_emb, test_emb)

    # Overlapping Ratio (OR)
    oratio = overlapping_ratio(train_emb, test_emb)

    return {"KL Divergence": kl_div, "Earth Mover’s Distance": emd, "Mean Nearest Neighbor Distance": m_nnd, "Overlapping Ratio": oratio}

# Function to process t-SNE and alignment per label category
def evaluate_label_alignment(train_features, train_labels, test_features, test_labels, perplexity=30):
    """Compute t-SNE and alignment metrics for each label category."""
    
    # Run t-SNE on combined train and test features
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    all_features = np.vstack([train_features, test_features])
    tsne_results = tsne.fit_transform(all_features)

    # Split back into train and test embeddings
    train_emb = tsne_results[:len(train_features)]
    test_emb = tsne_results[len(train_features):]

    # Convert to DataFrame for easier filtering
    train_df = pd.DataFrame(train_emb, columns=['x', 'y'])
    train_df['label'] = train_labels

    test_df = pd.DataFrame(test_emb, columns=['x', 'y'])
    test_df['label'] = test_labels

    # Compute metrics per label
    results = {}
    unique_labels = np.unique(np.concatenate([train_labels, test_labels]))

    for label in unique_labels:
        train_subset = train_df[train_df['label'] == label][['x', 'y']].values
        test_subset = test_df[test_df['label'] == label][['x', 'y']].values
        
        if len(train_subset) == 0 or len(test_subset) == 0:
            continue  # Skip if no samples for a label in either train or test

        metrics = compute_alignment_metrics(train_subset, test_subset)
        results[label] = metrics

    return results

# Example Usage:
# train_features and test_features should be numpy arrays of shape (num_samples, num_features)
# train_labels and test_labels should be 1D numpy arrays of shape (num_samples,)

# results = evaluate_label_alignment(train_features, train_labels, test_features, test_labels)
# print(results)
