import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter


from torch.distributions import Categorical
from src.utils import colorize_mask, oht_to_scalar
from src.data_util import get_palette, get_class_names
from src.cross_attention import DualPathNetwork_cross
from src.feature_extractors import create_feature_extractor, collect_features
from src.aggregation_network import AggregationNetwork

from PIL import Image
from collections import defaultdict
import pdb



from guided_diffusion.guided_diffusion.dist_util import dev
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import seaborn as sns

"""

class pixel_classifier(nn.Module):
    def __init__(
        self, num_classes, fused_dim, num_modalities=3,
        use_modalities=None, bottleneck_dim=128, dropout_p=0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fused_dim = fused_dim
        self.num_modalities = num_modalities
        self.per_modality_dim = fused_dim // num_modalities

        if use_modalities is None:
            self.use_modalities = list(range(num_modalities))
        else:
            assert all(0 <= idx < num_modalities for idx in use_modalities)
            self.use_modalities = use_modalities

        in_dim = self.per_modality_dim * len(self.use_modalities)

        self.feature_fusion = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim),
            #nn.Dropout(dropout_p),
        )
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            #nn.Dropout(dropout_p),
            nn.Linear(64, num_classes)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, fused_dim]
        chunks = torch.chunk(x, self.num_modalities, dim=-1)
        selected_feats = [chunks[i] for i in self.use_modalities]
        x_mod = torch.cat(selected_feats, dim=-1)
        x_fused = self.feature_fusion(x_mod)
        return self.classifier(x_fused)

"""
'''
## orginal pixel_classfier adapted to 3 modalities featue input and assin any modality 0- psedo rgb modality 1 pca reduced modaity 2 sar image
### the highest is with oa 0.7556 and aa 0.6667 berin 100 timestep layer 11 
class pixel_classifier(nn.Module):
    """
    use_modalities: 
        Specify which modalities to use by their indices.
        Examples:
            use_modalities=[0]      # Use only the first modality rgb
            use_modalities=[1]      # Use only the first modality pca
            use_modalities=[2]      # Use only the first modality sar
            use_modalities=[1, 2]  # Use the second and third modalities
            use_modalities=None    # Use all modalities (default)
    """
    def __init__(self, num_classes, fused_dim, num_modalities=3, use_modalities=None):
        super(pixel_classifier, self).__init__()
        self.num_classes = num_classes
        self.fused_dim = fused_dim
        self.num_modalities = num_modalities
        self.per_modality_dim = fused_dim // num_modalities

        if use_modalities is None:
            self.use_modalities = list(range(num_modalities))
        else:
            assert all(0 <= idx < num_modalities for idx in use_modalities), "Invalid modality index in use_modalities."
            self.use_modalities = use_modalities

        in_dim = self.per_modality_dim * len(self.use_modalities)

        if num_classes < 30:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, num_classes)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, num_classes)
            )

        # Initialize weights with Kaiming initialization
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, fused_dim]
        # Select features for the chosen modalities and concatenate
        chunks = torch.chunk(x, self.num_modalities, dim=-1)
        selected_feats = [chunks[i] for i in self.use_modalities]
        #pdb.set_trace()
        x_mod = torch.cat(selected_feats, dim=-1)
        return self.layers(x_mod)
'''
'''
class pixel_classifier(nn.Module):
    def __init__(self, num_classes, fused_dim, num_modalities=3, proj_dim=64, dropout_rates=[0.2,0.2,0.2]):
        super(pixel_classifier, self).__init__()

        self.num_modalities = num_modalities
        self.per_modality_dim = fused_dim // num_modalities
        self.proj_dim = proj_dim

        # Default dropout rates if not provided
        if dropout_rates is None:
            dropout_rates = [0.2] * num_modalities
        assert len(dropout_rates) == num_modalities, "Mismatch in number of dropout rates"

        # Store per-modality dropout layers
        self.dropouts = nn.ModuleList([
            nn.Dropout(p=dropout_rates[i]) for i in range(num_modalities)
        ])

        # Modality-specific projection layers (you can add BN here if needed)
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.per_modality_dim, proj_dim),
                # nn.BatchNorm1d(proj_dim),
            )
            for _ in range(num_modalities)
        ])

        # Attention MLP for adaptive fusion
        self.fusion_attn = nn.Sequential(
            nn.Linear(proj_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, fused_feat):
        B, D = fused_feat.shape
        assert D % self.num_modalities == 0

        # Step 1: Split into modality chunks
        chunks = torch.chunk(fused_feat, self.num_modalities, dim=-1)

        # Step 2: Apply projection + dropout per modality
        projected = [
            self.dropouts[i](self.proj_layers[i](chunks[i]))  # [B, proj_dim]
            for i in range(self.num_modalities)
        ]

        # Step 3: Attention-based fusion
        stacked = torch.stack(projected, dim=1)  # [B, M, proj_dim]
        attn_scores = self.fusion_attn(stacked).squeeze(-1)  # [B, M]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, M]
        fused = torch.sum(attn_weights.unsqueeze(-1) * stacked, dim=1)  # [B, proj_dim]
        #fused = fused + torch.mean(stacked, dim=1)  # Residual connection

        # Step 4: Classify
        logits = self.classifier(fused)
        return logits


###  drop out rate each proj then MLP concan oa 0.72

class pixel_classifier(nn.Module):
    def __init__(self, num_classes, fused_dim, num_modalities=3, proj_dim=64):
        super(pixel_classifier, self).__init__()

        self.num_modalities = num_modalities
        self.per_modality_dim = fused_dim // num_modalities

        # Modality-specific projection layers
        self.proj_layers = nn.ModuleList([
            nn.Linear(self.per_modality_dim, proj_dim)
            for _ in range(num_modalities)
        ])

        # Modality-specific dropout layers (customized rates per modality)
        dropout_rates = [0.2, 0.1, 0.2]  # [RGB,  HSI, SAR, DSM] example
        assert len(dropout_rates) == num_modalities, "Specify dropout for each modality!"

        self.dropout_layers = nn.ModuleList([
            nn.Dropout(rate) for rate in dropout_rates
        ])

        # Fusion MLP after concatenation
        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim * num_modalities, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU()
        )

        # Classifier MLP (BatchNorm before ReLU)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, fused_feat):
        """
        fused_feat: [B, fused_dim]
        """
        B, D = fused_feat.shape
        assert D % self.num_modalities == 0, "fused_dim must be divisible by num_modalities"

        # Step 1: Split features by modality
        chunks = torch.chunk(fused_feat, self.num_modalities, dim=-1)

        # Step 2: Project and apply modality-specific dropout
        projected = [
            dropout(proj(chunk))  # each [B, proj_dim]
            for proj, dropout, chunk in zip(self.proj_layers, self.dropout_layers, chunks)
        ]

        # Step 3: Concatenate projected modality features
        concat_feat = torch.cat(projected, dim=-1)  # [B, proj_dim * num_modalities]

        # Step 4: Fuse features using MLP
        fused = self.fusion_mlp(concat_feat)  # [B, proj_dim]

        # Step 5: Classify
        logits = self.classifier(fused)  # [B, num_classes]
        return logits

##  MLP adaptive attention with modality specif proj, OA 0.729 AA 0.65
class pixel_classifier(nn.Module):
    def __init__(self, num_classes, fused_dim, num_modalities=3, proj_dim=64):
        super(pixel_classifier, self).__init__()

        self.num_modalities = num_modalities
        self.per_modality_dim = fused_dim // num_modalities

        # Modality-specific projection layers
        self.proj_layers = nn.ModuleList([
            nn.Linear(self.per_modality_dim, proj_dim)
            for _ in range(num_modalities)
        ])

        self.dropout = nn.Dropout(0.2)

        # Attention MLP to compute adaptive fusion weights
        self.fusion_attn = nn.Sequential(
            nn.Linear(proj_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # outputs a scalar score per modality per sample
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, num_classes)
        )

    def forward(self, fused_feat):
        """
        Input:
            fused_feat: [B, fused_dim]  ← flattened features per pixel
        Output:
            logits: [B, num_classes]
        """
        B, D = fused_feat.shape
        assert D % self.num_modalities == 0, "fused_dim must be divisible by num_modalities"

        # Step 1: Split features into modality chunks
        chunks = torch.chunk(fused_feat, self.num_modalities, dim=-1)  # list of [B, D_m]

        # Step 2: Apply per-modality projections
        projected = [
            self.dropout(proj(chunk))  # each is [B, proj_dim]
            for proj, chunk in zip(self.proj_layers, chunks)
        ]

        # Step 3: Stack and compute attention weights dynamically
        stacked = torch.stack(projected, dim=1)  # [B, M, proj_dim]
        attn_scores = self.fusion_attn(stacked).squeeze(-1)  # [B, M]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, M]

        # Step 4: Weighted sum to get fused representation
        fused = torch.sum(attn_weights.unsqueeze(-1) * stacked, dim=1)  # [B, proj_dim]

        # Step 5: Classify
        logits = self.classifier(fused)  # [B, num_classes]
        return logits



# using common proj layer for each modality ,oa 0.7293, aa 0.6327
class pixel_classifier(nn.Module):
    def __init__(self, num_classes, fused_dim, num_modalities=3, proj_dim=64):
        super(pixel_classifier, self).__init__()

        self.num_modalities = num_modalities
        self.per_modality_dim = fused_dim // num_modalities

        # Shared projection layer across all modalities
        self.shared_proj = nn.Linear(self.per_modality_dim, proj_dim)
        self.dropout = nn.Dropout(0.2)

        # Learnable fusion weights (scalar per modality)
        self.alpha = nn.Parameter(torch.ones(num_modalities))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, num_classes)
        )

    def forward(self, fused_feat):
        """
        Input:
            fused_feat: [B, fused_dim]
        Output:
            logits: [B, num_classes]
        """
        B, D = fused_feat.shape
        assert D % self.num_modalities == 0, "fused_dim must be divisible by num_modalities"

        # Step 1: Split features into modality chunks
        chunks = torch.chunk(fused_feat, self.num_modalities, dim=-1)  # list of [B, D_m]

        # Step 2: Apply shared projection and dropout
        projected = [
            self.dropout(self.shared_proj(chunk))
            for chunk in chunks
        ]  # list of [B, proj_dim]

        # Step 3: Weighted fusion using softmax-normalized learnable alphas
        weights = F.softmax(self.alpha, dim=0)  # [M]
        fused = sum(w * p for w, p in zip(weights, projected))  # [B, proj_dim]

        # Step 4: Classification
        logits = self.classifier(fused)  # [B, num_classes]
        return logits


'''
'''
# modify on Junly 13,2025 a accuracy 0.7499 aa 0.64 for berlin

class pixel_classifier(nn.Module):
    def __init__(self, num_classes, fused_dim, num_modalities=3, proj_dim=64):
        super(pixel_classifier, self).__init__()

        self.num_modalities = num_modalities
        self.per_modality_dim = fused_dim // num_modalities  # e.g., 1200 → 400 if 3 modalities

        # Modality-specific projection layers
        self.proj_layers = nn.ModuleList([
            nn.Linear(self.per_modality_dim, proj_dim)
            for _ in range(num_modalities)
        ])

        self.dropout = nn.Dropout(0.2)

        # Learnable modality fusion weights
        self.alpha = nn.Parameter(torch.ones(num_modalities))

        
        # Classifier MLP optimized for < 30 classes
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, num_classes)
        )
        

    def forward(self, fused_feat):
        """
        Input:
            fused_feat: [B, fused_dim]  ← flattened features per pixel
        Output:
            logits: [B, num_classes]
        """
        # pdb.set_trace()
        B, D = fused_feat.shape
        assert D % self.num_modalities == 0, "fused_dim must be divisible by num_modalities"

        # Step 1: Split features by modality
        chunks = torch.chunk(fused_feat, self.num_modalities, dim=-1)

        # Step 2: Project each modality and apply dropout
        projected = [
            self.dropout(proj(chunk))
            for proj, chunk in zip(self.proj_layers, chunks)
        ]  # list of [B, proj_dim]

        # Step 3: Softmax fusion
        weights = F.softmax(self.alpha, dim=0)
        fused = sum(w * p for w, p in zip(weights, projected))  # [B, proj_dim]

        # Step 4: Classify
        logits = self.classifier(fused)  # [B, num_classes]
        return logits
'''
'''
## maks out version ,just change modality_mask

class pixel_classifier(nn.Module):
    def __init__(self, num_classes, fused_dim, num_modalities=3, proj_dim=64):
        super(pixel_classifier, self).__init__()

        self.num_modalities = num_modalities
        self.per_modality_dim = fused_dim // num_modalities  # e.g., 1200 → 300 if 4 modalities

        # Modality-specific projection layers
        self.proj_layers = nn.ModuleList([
            nn.Linear(self.per_modality_dim, proj_dim)
            for _ in range(num_modalities)
        ])

        self.dropout = nn.Dropout(0.2)

        # Learnable fusion weights
        self.alpha = nn.Parameter(torch.ones(num_modalities))

        # MLP classifier for pixel-wise prediction
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, num_classes)
        )

    def forward(self, fused_feat, modality_mask=None):
        """
        fused_feat: [B, fused_dim] (flattened features per pixel)
        modality_mask: Optional [num_modalities] binary tensor
        """
        B, D = fused_feat.shape
        assert D % self.num_modalities == 0, "fused_dim must be divisible by num_modalities"

        # Step 1: Split by modality
        chunks = torch.chunk(fused_feat, self.num_modalities, dim=-1)

        # Step 2: Project each modality separately
        projected = [
            self.dropout(proj(chunk))
            for proj, chunk in zip(self.proj_layers, chunks)
        ]  # list of [B, proj_dim]

        # Step 3: Handle modality mask
        if modality_mask is None:
            modality_mask = torch.ones(self.num_modalities, device=fused_feat.device)
        elif isinstance(modality_mask, list):
            modality_mask = torch.tensor(modality_mask, device=fused_feat.device, dtype=torch.float)
        else:
            modality_mask = modality_mask.to(fused_feat.device).float()
        

        # Step 4: Compute fusion weights with hard masking
        masked_alpha = self.alpha * modality_mask + (-1e9) * (1 - modality_mask)
        weights = F.softmax(masked_alpha, dim=0)

        # Step 5: Fuse projected features
        fused = sum(w * p for w, p in zip(weights, projected))  # [B, proj_dim]

        # Step 6: Classify per pixel
        logits = self.classifier(fused)  # [B, num_classes]
        return logits

'''

'''
#####  predict_labels for maxvoting version
def predict_labels(models, x_spatial, size):
    if isinstance(x_spatial, np.ndarray):
        x_spatial = torch.from_numpy(x_spatial)

    #pdb.set_trace()
    mean_seg = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []
    
    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            #pdb.set_trace()
            #preds = models[MODEL_NUMBER](x_spatial.cuda(),x_spectral.cuda()) # if using dualnetwork model
            preds = models[MODEL_NUMBER](x_spatial.cuda())  # if using pixel classfier model
            #preds = models[MODEL_NUMBER](x_spectral.cuda()) # if using  conv1d classfier model
            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            img_seg = oht_to_scalar(preds)
            img_seg = img_seg.reshape(*size)
            img_seg = img_seg.cpu().detach()

            seg_mode_ensemble.append(img_seg)
        mean_seg = mean_seg / len(all_seg)

        full_entropy = Categorical(mean_seg).entropy()

        js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 2)[0]
    return img_seg_final, top_k
'''

'''
class pixel_classifier(nn.Module):
    """
    use_modalities: indices of modalities to use (0=rgb, 1=pca, 2=sar).
    fused_dim: per-modality feature dimension (NOT total).
    """
    def __init__(self, num_classes, fused_dim, use_modalities=[0,1,2]):
        super(pixel_classifier, self).__init__()
        self.num_classes = num_classes
        self.per_modality_dim = int(fused_dim)
        self.use_modalities = list(use_modalities)
        assert all(idx >= 0 for idx in self.use_modalities), "Invalid modality index."
        in_dim = self.per_modality_dim * len(self.use_modalities)

        if num_classes < 30:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, num_classes),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, num_classes),
            )

        self.init_weights()
        #pdb.set_trace()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Sanity check
        #pdb.set_trace()

        if x.shape[-1] != self.per_modality_dim * len(self.use_modalities):
            raise ValueError("Input feature dimension does not match expected size.")
        return self.layers(x)
'''
# modify on Aug 14 to comaptibel with augsburg dataset



class pixel_classifier(nn.Module):
    """
    Pixel classifier with optional Augsburg-specific 2-layer MLP.
    category: string name of dataset ("Augsburg", "Berlin", etc.)
    use_modalities: indices of modalities to use (0=rgb, 1=pca, 2=sar, etc.)
    fused_dim: per-modality feature dimension (NOT total).
    """
    def __init__(self, num_classes, fused_dim, use_modalities=[0, 1, 2], category=None):
        super(pixel_classifier, self).__init__()
        self.num_classes = num_classes
        self.per_modality_dim = int(fused_dim)
        self.use_modalities = list(use_modalities)
        assert all(idx >= 0 for idx in self.use_modalities), "Invalid modality index."

        # Normalize category string for safe matching
        category = (category or "").strip().lower()
        in_dim = self.per_modality_dim * len(self.use_modalities)

        if category == "augsburg":
            # Augsburg-specific: simpler 2-layer MLP
            self.layers = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, num_classes),
            )
        else:
            # Default small-class case
            self.layers = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, num_classes),
            )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.shape[-1] != self.per_modality_dim * len(self.use_modalities):
            raise ValueError("Input feature dimension does not match expected size.")
        return self.layers(x)


## Single-Model Soft Voting (Probability Output) with Ensemble-Compatible Structure
def predict_labels(models, x_spatial, size):
    if isinstance(x_spatial, np.ndarray):
        x_spatial = torch.from_numpy(x_spatial)

    softmax_f = nn.Softmax(dim=1)
    mean_seg = None

    with torch.no_grad():
        for model in models:
            preds = model(x_spatial.cuda())       # (H*W, num_classes)
            probs = softmax_f(preds)              # (H*W, num_classes)
            if mean_seg is None:
                mean_seg = probs
            else:
                mean_seg += probs

        mean_seg = mean_seg / len(models)
        mean_seg = mean_seg.view(*size, -1)       # (H, W, num_classes)
        #pdb.set_trace()

    return mean_seg.cpu().detach()



def load_sorted_patches_npy(folder_path):
    # List all files in the directory
    files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]

    #files = os.listdir(folder_path)
    # Sort filenames by stripping the prefix and the '.png', then converting to integer
    #sorted_filenames_img = sorted(files, key=lambda x: int(x[len("hyperspectral"):-len(".jpg")]))
    sorted_filenames_label = sorted(files, key=lambda x: int(x[len("hyperspectral"):-len(".npy")]))
    # Load images into a list
    patches_label= [np.load(os.path.join(folder_path, file)) for file in sorted_filenames_label]
    
    return patches_label

### max voting version 
'''
def create_inference_map(patches, width=1920, height=352, patch_size=64, overlap=32):
    """
    Construct an inference map from overlapping patches by voting.

    Parameters:
    - patches: numpy array of patches.
    - width: width of the original image.
    - height: height of the original image.
    - patch_size: size of each patch.
    - overlap: overlap between patches.

    Returns:
    - inference_map: a numpy array representing the voting result of the class labels.
    """
    stride = patch_size - overlap

    # Initialize the voting structure
    voting_map = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
   
    # Calculate patch coordinates
    patch_coords = [(i * stride, j * stride) for i in range((height - patch_size) // stride + 1)
                    for j in range((width - patch_size) // stride + 1)]

    # Fill the voting map
    for patch, (x, y) in zip(patches, patch_coords):
        for i in range(patch_size):
            for j in range(patch_size):
                pixel_class = patch[i, j]
                voting_map[x + i][y + j][pixel_class] += 1

    # Initialize the final inference map
    inference_map = np.zeros((height, width), dtype=int)

    # Determine the mode for each pixel
    for x in range(height):
        for y in range(width):
            pixel_votes = voting_map[x][y]
            if pixel_votes:
                inference_map[x, y] = max(pixel_votes, key=pixel_votes.get)

    return inference_map


'''


# soft voting version 

def create_inference_map(patches, width=1920, height=352, patch_size=64, overlap=32, num_classes=None):
    """
    Construct an inference map from overlapping soft-voting patches (probabilities).

    Parameters:
    - patches: list or array of [patch_size, patch_size, num_classes]
    - width, height: size of original image
    - patch_size: size of each patch
    - overlap: amount of overlap
    - num_classes: number of classes (optional, will infer if not provided)

    Returns:
    - inference_map: [height, width] array of class predictions
    """
    stride = patch_size - overlap

    if num_classes is None:
        num_classes = patches[0].shape[-1]

    # Accumulators for probabilities and count
    prob_map = np.zeros((height, width, num_classes), dtype=np.float32)
    count_map = np.zeros((height, width, 1), dtype=np.float32)

    # Calculate patch coordinates
    patch_coords = [
        (i * stride, j * stride)
        for i in range((height - patch_size) // stride + 1)
        for j in range((width - patch_size) // stride + 1)
    ]

    # Fill in the probability and count maps
    for patch, (x, y) in zip(patches, patch_coords):
        prob_map[x:x+patch_size, y:y+patch_size, :] += patch
        count_map[x:x+patch_size, y:y+patch_size, 0] += 1

    # Average probabilities where overlaps occurred
    count_map[count_map == 0] = 1  # Avoid division by zero
    avg_prob_map = prob_map / count_map

    # Argmax over class probabilities
    inference_map = np.argmax(avg_prob_map, axis=-1)  # shape: [height, width]

    return inference_map



def save_predictions(args, image_paths, preds):

    palette = get_palette(args['category'])
    os.makedirs(os.path.join(args['exp_dir'], 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], 'visualizations'), exist_ok=True)

    
    prediction_path = os.path.join(args['exp_dir'],'predictions')
    print(f"save the predcitons lables to {prediction_path}")
    for i, pred in enumerate(preds):
        filename = os.path.splitext(os.path.basename(image_paths[i]))[0]
        #filename = image_paths[i].split('/')[-1].split('.')[0]
        
        # pred = np.squeeze(pred)   # for maxvoting it remove    
        np.save(os.path.join(prediction_path, filename + '.npy'), pred)
        if(i%100 ==0 ):
          print(f"saving {i}th file ")

    patches_npy = load_sorted_patches_npy(prediction_path) # shape list of [h, w, classes]

    #inference_map = create_inference_map(patches_npy,width=1920, height=352,patch_size=64, overlap=32)
    #start_row = 352-349
    #start_col = 1920-1905

    inference_map = create_inference_map(patches_npy,width=args['img_width_adjusted'], 
                                         height=args['img_height_adjusted'],patch_size=64, overlap=32,num_classes= args["number_class"])
    start_row = args['img_height_adjusted']- args['img_height_orig']
    start_col = args['img_width_adjusted'] - args['img_width_orig']

    inference_map = inference_map[start_row:, start_col:]
    np.save(os.path.join(args['exp_dir'], 'visualizations', 'inference_map.npy'),inference_map)


    test_label = np.load(args['test_label_path'])
    ignore_label = args['ignore_label']
    valid_mask = test_label!= ignore_label
    inference_map_filted = inference_map[valid_mask]

    test_label_filted = test_label[valid_mask]-1 # don't forget to minus 1

    adjusted_inferene_map = inference_map +1
    masked_inference_map = np.where(valid_mask, adjusted_inferene_map, 0)
    
    mask = colorize_mask(masked_inference_map, palette)
    Image.fromarray(mask).save(
      os.path.join(args['exp_dir'], 'visualizations', 'inference_map.jpg')
        )
    visualization_path = os.path.join(args['exp_dir'], 'visualizations', 'inference_map.jpg')
    print(f"Save inference map to{visualization_path} ")


 
    return test_label_filted,inference_map_filted

def calculate_metric_per_class_plot_cm(args,test_label_filted, inference_map_filted):
   
    class_labels = get_class_names(args['category'])

    predictions_flat = inference_map_filted.flatten()  # prediction labels, exclude unlabeled , label =0 pixels
    test_labels_flat = test_label_filted.flatten() # true label exclude unlabeled ,label = 0 pixels

    # Create the confusion matrix
    cm = confusion_matrix(test_labels_flat, predictions_flat)

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Saving the plot to a file
    plt.savefig(os.path.join(args['exp_dir'], 'visualizations', 'confusion_matrix.png'))  # Save as PNG file
    plt.savefig(os.path.join(args['exp_dir'], 'visualizations', 'confusion_matrix.svg'), format='svg')  # Save as SVG file for vectorized output
    plt.savefig(os.path.join(args['exp_dir'], 'visualizations', 'confusion_matrix.pdf'), format='pdf')  # Save as PDF file for documents

     # Calculate metrics
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    #TN = np.sum(cm) - (FP + FN + TP)
    #TN  = np.sum(cm) * np.ones_like(TP) - (np.sum(cm, axis=0) + np.sum(cm, axis=1) - TP)

    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    #accuracy = (TP + TN) / np.sum(cm)
    f1_scores = 2 * precision * recall / (precision + recall)
    miou = TP / (TP + FP + FN)
    overall_accuracy = np.sum(TP) / np.sum(cm)
    kappa = cohen_kappa_score(test_labels_flat, predictions_flat)

    print("Per-Class Metrics:")
    for idx, label in enumerate(class_labels):
      print(f"{class_labels[idx]} -   Recall: {recall[idx]:.4f}, Precision: {precision[idx]:.4f},F1 Score: {f1_scores[idx]:.4f}, IoU: {miou[idx]:.4f}")

    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"AA: {np.nanmean(recall):.4f}")
    print(f"Kappa Coefficient: {kappa:.4f}")
    print(f"Mean IoU: {np.nanmean(miou):.4f}")
    print(f"Mean F1 Score: {np.nanmean(f1_scores):.4f}")
    

    # Open the text file for writing
    with open( os.path.join(args['exp_dir'], 'visualizations', 'metrics_output.txt'), 'w') as file:
        print("Per-Class Metrics:", file=file)
        for idx, label in enumerate(class_labels):
            print(f"{class_labels[idx]} -  Recall: {recall[idx]:.4f}, Precision: {precision[idx]:.4f}, F1 Score: {f1_scores[idx]:.4f}, IoU: {miou[idx]:.4f}", file=file)
        print(f"Overall Accuracy: {overall_accuracy:.4f}", file=file)
        print(f"AA: {np.nanmean(recall):.4f}",file = file)
        print(f"Kappa Coefficient: {kappa:.4f}", file=file)
        print(f"Mean IoU: {np.nanmean(miou):.4f}", file=file)
        print(f"Mean F1 Score: {np.nanmean(f1_scores):.4f}", file=file)




#  using iou, accuracy , f1 scores to mesure model 
def calculate_metric_per_class(args,test_label, inference_map, num_classes):

    class_names = get_class_names(args['category'])
    ignore_label = args['ignore_label']
    iou_scores = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)   
    f1_scores = np.zeros(num_classes)

    mask = test_label!= ignore_label
    # Calculate accuracy only over masked (labeled) areas
    masked_pred = np.where(mask == 1, inference_map, np.nan)
    masked_gt = np.where(mask == 1, test_label, np.nan)
   

    #Calculate correct predictions
    correct_predictions = masked_gt == masked_pred
    # Calculate accuracy
    accuracy = np.sum(correct_predictions)/np.sum(mask)
    miou_list  = []
    for cls in range(1, num_classes+1):
        # Calculate intersection: True Positives (TP)
        TP = (masked_pred == cls) & (masked_gt == cls)
        TP = np.sum(TP)
        TN = np.sum((masked_pred != cls) & (masked_gt != cls))
        
        # Calculate union: TP + False Positives (FP) + False Negatives (FN)
        FP_mask = (masked_gt != cls) & (masked_pred == cls)
        FN_mask = (masked_gt == cls) & (masked_pred != cls)
        FP = np.sum(FP_mask)
        FN = np.sum(FN_mask)
       
        # Calculating Precision and Recall
        accuracy[cls] = TP/(TP+TN)
        precision[cls] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[cls] = TP / (TP + FN) if (TP + FN) > 0 else 0
        if (precision[cls] + recall[cls]) >0:
          f1_scores[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls])
        else:
          f1_scores[cls] = 0
        union = TP + FP + FN
        if union == 0:
            iou_scores[cls] = np.nan  # or set to zero, depending on how you want to handle this case
        else:
            iou_scores[cls] = TP / union  

        print(f" {class_names[cls]} Iou: {iou_scores[cls]:.4}, Precision: {precision[cls]:.4},F1 score : {f1_scores[cls]}, Recall: {recall[cls]}")


    return np.array(iou_scores).mean(),accuracy,recall, f1_scores




def compute_iou(args, preds, gts, print_per_class_ious=True):
    class_names = get_class_names(args['category'])

    ids = range(args['number_class'])

    unions = Counter()
    intersections = Counter()

    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == args['ignore_label']: 
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()
    
    ious = []
    for target_num in ids:
        if target_num == args['ignore_label']: 
            continue
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        ious.append(iou)
        if print_per_class_ious:
            print(f"IOU for {class_names[target_num]} {iou:.4}")
    return np.array(ious).mean()


def load_ensemble(args, device='cpu'):
    #pdb.set_trace()
    models = []
    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']
        
        model = pixel_classifier(num_classes= args['number_class'], fused_dim= args['dim'][-1],
                                 use_modalities = args["use_modalities"],category=args["category"]) # if using pixel classfier
 
        model.load_state_dict(state_dict)
        model = model.to(device)
        models.append(model.eval())
    return models
