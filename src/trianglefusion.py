import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityTriangleUpdate(nn.Module):
    def __init__(self, num_modalities=3, feature_dim=384, hidden_dim=256):
        super(ModalityTriangleUpdate, self).__init__()
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim

        # MLP for triangle message passing (D×D → hidden → D×D)
        self.pairwise_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * feature_dim)
        )

        # Learnable compression of D×D → compact vector
        self.compress_block = nn.Sequential(
            nn.Linear(feature_dim * feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, modality_features):
        B, M, D = modality_features.shape
        pairwise = torch.einsum("bmd,bnd->bmnd", modality_features, modality_features)  # (B, M, M, D, D)
        updated = torch.zeros_like(pairwise)

        for i in range(M):
            for j in range(M):
                if i == j:
                    updated[:, i, j] = pairwise[:, i, j]
                    continue
                messages = []
                for k in range(M):
                    if k == i or k == j:
                        continue
                    pij = pairwise[:, i, k].reshape(B, -1)
                    pjk = pairwise[:, k, j].reshape(B, -1)
                    cat = torch.cat([pij, pjk], dim=-1)
                    msg = self.pairwise_mlp(cat).reshape(B, D, D)
                    messages.append(msg)
                if messages:
                    messages = torch.stack(messages, dim=0).mean(dim=0)
                    updated[:, i, j] = messages

        updated_flat = updated.view(B, M, M, -1)              # (B, M, M, D*D)
        compressed = self.compress_block(updated_flat)       # (B, M, M, 32)
        return compressed


class TriangleFusionClassifier(nn.Module):
    def __init__(self, feature_dim=384, num_classes=8):
        super().__init__()
        self.triangle_update = ModalityTriangleUpdate(num_modalities=3, feature_dim=feature_dim)

        self.classifier = nn.Sequential(
            nn.Flatten(),                 # (B, 3, 3, 32) → (B, 288)
            nn.Linear(288, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: (B, 3, 384) pixel-level features from 3 modalities (RGB, PCA, SAR)
        """
        compressed = self.triangle_update(x)   # (B, 3, 3, 32)
        logits = self.classifier(compressed)
        return logits
