
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sns
import os
import pdb

from sklearn.preprocessing import normalize
from itertools import combinations


def plot_tsne(features_dict, labels_dict, title="t-SNE Visualization", perplexity=30, n_iter=1000):
    # Combine features and labels
    all_features = []
    all_labels = []
    model_names = []

    for model_name, features in features_dict.items():
        all_features.append(features)
        all_labels.append(labels_dict[model_name])
        model_names.extend([model_name] * len(features))

    # Stack features into a single matrix
    X_combined = np.vstack(all_features)
    y_combined = np.hstack(all_labels)
    model_labels = np.array(model_names)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=n_iter)
    X_tsne = tsne.fit_transform(X_combined)

    # Create DataFrame
    df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
    df['Model'] = model_labels
    df['Label'] = y_combined.astype(str)  # Convert labels to strings for consistent color mapping

    # Define a fixed color mapping for labels
    unique_labels = sorted(df['Label'].unique())  # Ensure same order every time
    palette_colors = sns.color_palette("deep", n_colors=len(unique_labels))
    fixed_palette = {label: color for label, color in zip(unique_labels, palette_colors)}

    # Plot each model separately
    unique_models = df['Model'].unique()
    fig, axes = plt.subplots(1, len(unique_models), figsize=(10, 6), sharex=True, sharey=True)

    if len(unique_models) == 1:
        axes = [axes]  # Convert single axis to a list for consistency

    for i, model in enumerate(unique_models):
        subset = df[df['Model'] == model]
        sns.scatterplot(data=subset, x="TSNE1", y="TSNE2", hue="Label", palette=fixed_palette, ax=axes[i], alpha=0.7)
        axes[i].set_title(f"t-SNE for {model}")

    plt.suptitle(title)
    plt.show()
    return fig


def plot_pca(train_features, test_features):
    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(train_features)
    test_pca = pca.transform(test_features)  # Use the same transformation!

    plt.figure(figsize=(8,6))
    plt.scatter(train_pca[:,0], train_pca[:,1], alpha=0.5, label="Training", marker="o")
    plt.scatter(test_pca[:,0], test_pca[:,1], alpha=0.5, label="Testing", marker="x")
    plt.legend()
    plt.title("PCA Projection of Training & Testing Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

from sklearn.manifold import TSNE

def plot_tsne_with_classes(train_features, test_features, train_labels, test_labels):
    combined_features = np.vstack([train_features, test_features])
    combined_labels = np.concatenate([train_labels, test_labels])
    dataset_labels = np.array(["Train"] * len(train_features) + ["Test"] * len(test_features))

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(combined_features)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=combined_labels, style=dataset_labels, alpha=0.7)
    plt.title("t-SNE with Class Labels (Training vs. Testing)")
    plt.legend()
    plt.show()
"""    
def apply_faiss_pca_train(train_data, n_components=10, save_path="faiss_pca.bin", device=dev()):

    train_data_np = to_numpy(train_data).astype(np.float32)  # Convert to NumPy if needed

    d = train_data_np.shape[1]  # Number of features
    pca_matrix = faiss.PCAMatrix(d, n_components)

    # Train PCA on training data
    pca_matrix.train(train_data_np)

    # Apply PCA transformation
    reduced_train_data = pca_matrix.apply_py(train_data_np)

    # Save PCA model
    faiss.write_VectorTransform(pca_matrix, save_path)
    print(f"✅ FAISS PCA model saved to {save_path}")

    return to_tensor(reduced_train_data, device), pca_matrix  # Convert back to tensor if needed

def apply_faiss_pca_test(test_data, load_path="faiss_pca.bin", device=dev()):

    test_data_np = to_numpy(test_data).astype(np.float32)  # Convert to NumPy if needed

    # Load saved PCA model
    pca_matrix = faiss.read_VectorTransform(load_path)

    # Apply PCA transformation
    reduced_test_data = pca_matrix.apply_py(test_data_np)

    return to_tensor(reduced_test_data, device)  # Convert back to tensor if needed

"""


def plot_tsne_overlay(features_dict, labels_dict, title="t-SNE Overlay", perplexity=30, n_iter=1000):
    # Combine data
    all_features = []
    all_labels = []
    modalities = []

    for modality, features in features_dict.items():
        all_features.append(features)
        all_labels.append(labels_dict[modality])
        modalities.extend([modality] * len(features))

    X_combined = np.vstack(all_features)
    y_combined = np.hstack(all_labels)
    modality_labels = np.array(modalities)

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=n_iter)
    X_tsne = tsne.fit_transform(X_combined)

    # Create DataFrame for plotting
    df = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
    df["Label"] = y_combined.astype(str)
    df["Modality"] = modality_labels

    # Plot with seaborn — overlay both modalities
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="TSNE1", y="TSNE2",
        hue="Label", style="Modality",
        alpha=0.6, s=15, palette="tab10"
    )
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_tsne_by_modality(features_dict, title="t-SNE: Modality Comparison", perplexity=10, n_iter=1000):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE

    # Combine features
    all_features = []
    modalities = []

    for modality, features in features_dict.items():
        all_features.append(features)
        modalities.extend([modality] * len(features))

    X_combined = np.vstack(all_features)
    modality_labels = np.array(modalities)

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=n_iter)
    X_tsne = tsne.fit_transform(X_combined)

    # Create DataFrame
    df = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
    df["Modality"] = modality_labels

    # Define 2-color palette manually
    two_color_palette = {
        "PCA": "#1f77b4",  # blue
        "RGB": "#d62728",  # red
    }

    # Define distinct shapes (optional)
    shape_dict = {
        "PCA": "o",
        "RGB": "X"
    }

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="TSNE1", y="TSNE2",
        hue="Modality",
        style="Modality",
        markers=shape_dict,
        palette=two_color_palette,
        s=15, alpha=0.6
    )

    plt.title(title)
    plt.legend(title="Modality", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()





def compute_modality_cosine_similarity(train_features, train_labels):
    """
    Splits the features into PCA and RGB halves and computes cosine similarity between them.
    Returns:
        - cosine similarities (1D array)
        - corresponding class labels (1D array)
    """
    N, D = train_features.shape
    D_half = D // 2

    pca_feats = train_features[:, :D_half]
    rgb_feats = train_features[:, D_half:]

    # Normalize to unit vectors
    pca_norm = normalize(pca_feats, axis=1)
    rgb_norm = normalize(rgb_feats, axis=1)

    # Row-wise cosine similarity
    cos_sim = np.sum(pca_norm * rgb_norm, axis=1)
    
    return cos_sim, train_labels






def compute_and_display_class_stats(similarities, labels):
    """
    Prints a summary table of mean similarity per class.
    """
    df = pd.DataFrame({'Similarity': similarities, 'Class': labels})
    stats = df.groupby('Class')['Similarity'].agg(['mean', 'std', 'count']).reset_index()
    print(stats.rename(columns={
        'mean': 'Mean Similarity',
        'std': 'Std Dev',
        'count': 'Sample Count'
    }))
    return stats



'''
def plot_cosine_similarity_boxplot(similarities, labels, title="Cosine Similarity (PCA vs RGB) per Class",save_path = None):
    """
    Creates a boxplot of cosine similarities grouped by class.
    """
    df = pd.DataFrame({
        'Cosine Similarity': similarities,
        'Class': labels
    })

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Class", y="Cosine Similarity", palette="Set3")
    sns.stripplot(data=df, x="Class", y="Cosine Similarity", color='black', size=1.5, alpha=0.3, jitter=0.2)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        boxplot_file = os.path.join(save_path, "cosine_similarity_plot.pdf")
        plt.savefig(boxplot_file, format='pdf')
        #filename = "cos_similarity.csv"
        #csv_file = os.path.join(save_path, "cosine_similarity.csv")
        #df.to_csv(csv_file, index=False)
        #print(f" Plot saved to: {boxplot_file}")
    #plt.show()
'''


def plot_cosine_similarity_boxplot(similarities, labels, title="", save_path=None, class_names=None):
    df = pd.DataFrame({'Cosine Similarity': similarities, 'Class': labels})
    df['Class'] = df['Class'].astype(int)  # ensure numeric (so seaborn orders numerically)

    # enforce numeric order explicitly
    order = sorted(df['Class'].unique())

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=df, x="Class", y="Cosine Similarity", palette="Set3", order=order)
    sns.stripplot(data=df, x="Class", y="Cosine Similarity", color='black', size=1.5,
                  alpha=0.3, jitter=0.2, order=order)

    # optionally replace numeric ticks with names (without changing df)
    if class_names is not None:
        if isinstance(class_names, dict):
            tick_labels = [class_names.get(i, str(i)) for i in order]
        else:  # assume list indexed by class id
            tick_labels = [class_names[i-1] for i in order]
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')
    else:
        plt.xticks(rotation=45, ha='right')

    plt.title(title, fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out_pdf = os.path.join(save_path, "cosine_similarity.pdf")
        plt.savefig(out_pdf, format='pdf')
   
    plt.close()



def compute_all_pairwise_similarities_df(
    features,
    labels,
    use_modalities,                  # e.g., [0,1,2] or [0,1,2,3]
    per_modality_dim=None,
    modality_names=None,             # {0:'rgb', 1:'pca', 2:'sar', 3:'dsm'}
):
    """
    Compute cosine similarity per sample between all unique pairs among `use_modalities`.
    Assumes `features` is concatenated *in exactly the same order* as `use_modalities`.

    features: [N, D] concatenated features of selected modalities in the given order.
    labels:   [N]
    """
    # torch -> numpy if needed
    try:
        import torch
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
    except ImportError:
        pass

    if modality_names is None:
        modality_names = {0: 'rgb', 1: 'pca', 2: 'sar', 3: 'dsm'}

    N, D = features.shape
    M = len(use_modalities)

    # Infer per-modality dim if not provided
    if per_modality_dim is None:
        if D % M != 0:
            raise ValueError("Feature dim not divisible by number of selected modalities.")
        per_modality_dim = D // M

    # Slice chunks in the SAME order as use_modalities
    chunks = {}
    for k, m in enumerate(use_modalities):
        start, end = k * per_modality_dim, (k + 1) * per_modality_dim
        X = normalize(features[:, start:end], axis=1)
        chunks[m] = (modality_names.get(m, f"mod{m}"), X)

    # Pairwise cosine (row-wise since rows are L2-normalized)
    dfs = []
    for mi, mj in combinations(use_modalities, 2):
        name_i, Xi = chunks[mi]
        name_j, Xj = chunks[mj]
        cos_sim = np.sum(Xi * Xj, axis=1)
        df = pd.DataFrame({
            "Cosine Similarity": cos_sim,
            "Class": labels,
            "Modality Pair": f"{name_i}_{name_j}",
            "i": mi,
            "j": mj,
        })
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)





def plot_all_pairs_boxplots(df, class_names=None, save_path="similarity/plots_per_pair"):
    """
    df columns expected: ['Modality Pair', 'Class', 'Cosine Similarity'].
    Saves one PDF per pair; keeps Class numeric here and lets the single-pair helper
    handle optional name mapping for tick labels.
    """
    os.makedirs(save_path, exist_ok=True)

        
    for pair, sub in df.groupby("Modality Pair"):
        safe_pair = pair.replace("/", "-").replace(" ", "_")
        out_dir = os.path.join(save_path, safe_pair)

        plot_cosine_similarity_boxplot(
            similarities=sub["Cosine Similarity"].to_numpy(),
            labels=sub["Class"].to_numpy(),          # keep numeric
            title=f"Cosine Similarity per Class_{pair}",
            save_path=out_dir,
            class_names=class_names                  # helper maps names on ticks if provided
        )


'''
def compute_all_pairwise_similarities_df(train_features, train_labels, num_modalities=3):
    """
    Computes cosine similarity between all unique modality pairs and returns a DataFrame.
    """
    N, D = train_features.shape
    d_per_modality = D // num_modalities

    modalities = [normalize(train_features[:, i*d_per_modality:(i+1)*d_per_modality], axis=1)
                  for i in range(num_modalities)]
    
    dfs = []
    for i, j in combinations(range(num_modalities), 2):
        cos_sim = np.sum(modalities[i] * modalities[j], axis=1)
        df = pd.DataFrame({
            'Cosine Similarity': cos_sim,
            'Class': train_labels,
            'Modality Pair': f"Mod{i}-Mod{j}"
        })
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df
'''

