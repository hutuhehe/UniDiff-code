# UniDiff-MM: Parameter-Efficient Diffusion for Multimodal Remote Sensing

This repository contains the implementation of **UniDiff-MM** (WACV 2026 submission).  
UniDiff-MM adapts pretrained diffusion backbones for multimodal remote-sensing (HSI + SAR) using a parameter-efficient framework with joint timestep–modality conditioning.

---

## 🔧 Environment Setup

Install the environment with pip:

```bash
python -m venv diffusion_env
source diffusion_env/bin/activate      # Linux/Mac
# OR
.\diffusion_env\Scripts\activate       # Windows

pip install -r requirements.txt
```


---

## 📂 Data Preparation

We use two multimodal datasets: **Berlin** and **Augsburg**.

### 1. Download
- [Berlin dataset](https://rslab.utdallas.edu/data/berlin-hsi-sar)  
- [Augsburg dataset](https://rslab.utdallas.edu/data/augsburg-hsi-sar)




### 2. Organize
Place raw data under:
```
project_root/
├── data/
│   ├── berlin_raw/
│   │   ├── HSI_Berlin.tif
│   │   └── SAR_Berlin.tif
│   └── augsburg_raw/
│       ├── HSI_Augsburg.tif
│       └── SAR_Augsburg.tif
```

### 3. Preprocess into patches
Run the provided Jupyter notebooks:

- `hyperspectral snip Berlin.ipynb`  
- `hyperspectral snip Augsburg.ipynb`

Each notebook:
- Loads HSI + SAR cubes  
- Splits into **64×64 patches** with **stride 32**  
- Saves patches under `exp_set/`:

```
exp_set/
├── berlin/
│   ├── HSI_patches/
│   ├── SAR_patches/
│   └── labels/
└── augsburg/
    ├── HSI_patches/
    ├── SAR_patches/
    └── labels/
```

---

## ⚙️ Dataset Configurations

We provide JSON configs in `exp_set/`:

- `dataset_hsi.json` → HSI-only patches  
- `dataset_hsi_sar.json` → HSI + SAR fusion patches  
- `dataset_pretrain.json` → Pretraining setup  

---

## 🚀 Training & Evaluation

### Stage A: Adaptation
Example (HSI-only, Berlin):
```bash
python train_adapt.py --dataset_config exp_set/dataset_hsi.json
```

HSI + SAR fusion:
```bash
python train_adapt.py --dataset_config exp_set/dataset_hsi_sar.json
```

### Stage B: Classification
Example (Berlin):
```bash
python train_cls.py --dataset_config exp_set/dataset_hsi_sar.json
```

### Pretraining
```bash
python train_pretrain.py --dataset_config exp_set/dataset_pretrain.json
```

### Evaluation
```bash
python evaluate.py --dataset berlin --checkpoint checkpoints/berlin_best.pth
```

---




