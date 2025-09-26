import torch
import pandas as pd
from itertools import combinations

# ---------- CKA utility ----------
@torch.no_grad()
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA between two feature matrices for the SAME N samples.
    X, Y: [N, d] (float tensors). Returns scalar in [0,1].
    """
    # ensure float CPU tensors
    X = X.detach().to(dtype=torch.float32, device='cpu')
    Y = Y.detach().to(dtype=torch.float32, device='cpu')

    # column-center
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    # Frobenius-based CKA
    XS = Xc.T @ Xc
    YS = Yc.T @ Yc
    XY = Xc.T @ Yc
    num = (XY ** 2).sum()
    den = torch.linalg.norm(XS, ord='fro') * torch.linalg.norm(YS, ord='fro') + 1e-12
    return (num / den).item()

# ---------- Pairwise CKA over selected modalities ----------
def compute_pairwise_cka_df(features,
                            use_modalities,
                            per_modality_dim=None,
                            modality_names=None):
    """
    Compute overall linear CKA for every unique pair in `use_modalities`.

    Args:
        features: [N, D] concatenated ONLY for the selected modalities,
                  in the SAME order as `use_modalities`. (torch.Tensor or np.ndarray)
        use_modalities: list[int], e.g. [0,1,2] (0=rgb, 1=pca, 2=sar, 3=dsm, ...)
        per_modality_dim: int or None. If None, inferred as D // len(use_modalities).
        modality_names: dict[int,str] or None. Default {0:'rgb',1:'pca',2:'sar',3:'dsm'}.

    Returns:
        pandas.DataFrame with columns: ['Modality Pair','i','j','CKA']
    """
    if modality_names is None:
        modality_names = {0: 'rgb', 1: 'pca', 2: 'sar', 3: 'dsm'}

    # to torch tensor on CPU
    if isinstance(features, torch.Tensor):
        F = features.detach().cpu()
    else:
        F = torch.as_tensor(features, dtype=torch.float32)

    N, D = F.shape
    M = len(use_modalities)

    if per_modality_dim is None:
        if D % M != 0:
            raise ValueError("Feature dim not divisible by number of selected modalities.")
        per_modality_dim = D // M

    # slice chunks in the order of use_modalities
    chunks = {}
    for k, m in enumerate(use_modalities):
        s, e = k * per_modality_dim, (k + 1) * per_modality_dim
        chunks[m] = F[:, s:e]

    # compute CKA for each pair
    rows = []
    for i, j in combinations(use_modalities, 2):
        cka = linear_cka(chunks[i], chunks[j])
        pair_name = f"{modality_names.get(i, f'mod{i}')}_{modality_names.get(j, f'mod{j}')}"
        rows.append({"Modality Pair": pair_name, "i": i, "j": j, "CKA": cka})

    return pd.DataFrame(rows)
