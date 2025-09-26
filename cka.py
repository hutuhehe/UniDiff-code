import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from guided_diffusion.guided_diffusion.dist_util import dev

# -------------------------
# Compute CKA
# -------------------------
def compute_cka(X, Y, device='cuda', max_samples=None):
    X = X.to(device)
    Y = Y.to(device)
    n = X.shape[0]

    if max_samples and n > max_samples:
        idx = torch.randperm(n, device=device)[:max_samples]
        X, Y = X[idx], Y[idx]
        n = max_samples

    X = torch.nn.functional.normalize(X, dim=1)
    Y = torch.nn.functional.normalize(Y, dim=1)

    K = X @ X.T
    L = Y @ Y.T

    H = torch.eye(n, device=device) - torch.ones((n, n), device=device) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic = torch.trace(Kc @ Lc)
    norm_x = torch.trace(Kc @ Kc)
    norm_y = torch.trace(Lc @ Lc)
    return (hsic / torch.sqrt(norm_x * norm_y)).item()


# -------------------------
# Main
# -------------------------
def main(pretrained_path, adapted_path, modalities, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = dev()

    # Load datasets
    pretrained_data = torch.load(pretrained_path)
    adapted_data = torch.load(adapted_path)
    feat_pre, labels_pre = pretrained_data.tensors
    feat_adapt, labels_adapt = adapted_data.tensors

    print(f"Pretrained: {feat_pre.shape}, Adapted: {feat_adapt.shape}")
    assert torch.equal(labels_pre, labels_adapt), "Labels differ between pretrained and adapted!"

    # Split by modalities
    num_modalities = len(modalities)
    total_dim = feat_pre.shape[1]
    chunk_size = total_dim // num_modalities

    print(f"Modalities: {modalities}")
    print(f"Chunk size per modality: {chunk_size}")

    # Create dictionaries
    pre_splits = {}
    adapt_splits = {}
    for i, mod in enumerate(modalities):
        start, end = i * chunk_size, (i + 1) * chunk_size
        pre_splits[mod] = feat_pre[:, start:end]
        adapt_splits[mod] = feat_adapt[:, start:end]

    # Compute CKA
    print("\n--- CKA: Pretrained vs Adapted ---")
    results = {}
    for mod in modalities:
        cka_val = compute_cka(pre_splits[mod], adapt_splits[mod], device=device)
        results[mod] = cka_val
        print(f"{mod}: {cka_val:.4f}")

    # Combined CKA
    cka_combined = compute_cka(feat_pre, feat_adapt, device=device)
    print(f"Combined: {cka_combined:.4f}")
    results["Combined"] = cka_combined

    # Save results
    with open(os.path.join(output_dir, "cka_results.txt"), "w") as f:
        f.write("CKA: Pretrained vs Adapted\n")
        for mod, val in results.items():
            f.write(f"{mod}: {val:.4f}\n")

    # Bar chart
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="muted")
    plt.title("CKA: Pretrained vs Adapted")
    plt.ylabel("CKA Score")
    for i, val in enumerate(results.values()):
        plt.text(i, val + 0.01, f"{val:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cka_bar_chart.pdf"), dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, default="Berlin_pretrained.pt", help="Path to pretrained .pt file")
    parser.add_argument("--adapted_path", type=str, default="Berlin_adapted.pt", help="Path to adapted .pt file")
    parser.add_argument("--modalities", nargs="+", default=["RGB", "PCA", "SAR"], help="List of modalities (order matters)")
    parser.add_argument("--output_dir", type=str, default="./cka_results", help="Directory for saving results")
    args = parser.parse_args()
    main(args.pretrained_path, args.adapted_path, args.modalities, args.output_dir)
