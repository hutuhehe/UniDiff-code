# UniDiff-MM: Parameter-Efficient Diffusion for Multimodal Remote Sensing

## Data

We evaluate our approach on two hyperspectral image (HSI) datasets: **Augsburg** and **Berlin**.  
Both datasets can be downloaded from the [ISPRS S2FL repository](https://github.com/danfenghong/ISPRS_S2FL).

### 1) Dataset Setup

**Datasets:** ISPRS S2FL — Berlin and Augsburg  
**Download:** https://github.com/danfenghong/ISPRS_S2FL

**Tiling Configuration:**  
We divide each HSI cube and SAR image into **64×64 patches**.  
- **Default:** `stride = 32` → 50% overlap  

**Preprocessing Notebooks:**
- [`hyperspectral_snip_Berlin.ipynb`](./hyperspectral_snip_Berlin.ipynb)  
- [`hyperspectral_snip_Augsburg.ipynb`](./hyperspectral_snip_Augsburg.ipynb)  

These notebooks handle:  
- Loading HSI + SAR data cubes  
- Splitting into 64×64 patches (with specified stride)  
- Saving processed data under `datasets/` directory

**Directory Structure:**
```
datasets/
├── Augsburg_benchmark/
│   ├── Augsburg_data_32/          # Stage A (unlabeled adaptation)
│   ├── Augsburg_train_data_32/    # Stage B (labeled training)
│   ├── Augsburg_test_data_32/     # Stage B (test tiles)
│   └── test_label_Augsburg.npy    # Stage B (test labels)
└── Berlin_benchmark/
    ├── Berlin_data_32/            # Stage A (unlabeled adaptation)
    ├── Berlin_train_data_32/      # Stage B (labeled training)
    ├── Berlin_test_data_32/       # Stage B (test tiles)
    └── test_label_Berlin.npy      # Stage B (test labels)
```

### 2) Pre-trained Model

We use the **64×64 ImageNet-pretrained diffusion model** from OpenAI's [guided-diffusion repository](https://github.com/openai/guided-diffusion).

**Required Checkpoint:**
- Download: [`64x64_diffusion.pt`](https://github.com/openai/guided-diffusion)
- **Placement:** `checkpoints/ddpm/64x64_diffusion.pt`



### 3) Dataset Configurations

We provide JSON configuration files for different experimental setups:

**Configuration Files:**
```
exp_set/
├── Augsburg/
│   ├── dataset_hsi.json          # HSI-only modality
│   ├── dataset_hsi_sar.json      # HSI + SAR multimodal
│   └── dataset_pretrain.json     # Pretraining setup
└── Berlin/
    ├── dataset_hsi.json          # HSI-only modality
    ├── dataset_hsi_sar.json      # HSI + SAR multimodal
    └── dataset_pretrain.json     # Pretraining setup
```

**Configuration Types:**
- **`dataset_hsi.json`** → HSI-only modality experiments
- **`dataset_hsi_sar.json`** → HSI + SAR multimodal experiments  
- **`dataset_pretrain.json`** → Pretraining configuration

## Training Pipeline

*Alternatively, you can use the `stageA_B.ipynb` notebook for streamlined training.*

### 4) Stage A — Adaptation (Unlabeled)

This stage uses only image tiles (**no labels**) for domain adaptation.

**Example Training Command (Augsburg):**
```bash
# Set up paths and flags
DATA_DIR="datasets/Augsburg_benchmark/Augsburg_data_32"  # unlabeled tiles
LOGDIR="runs/adapt/augsburg64"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 \
             --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine \
             --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True \
             --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"

TRAIN_FLAGS="--lr 1e-4 --batch_size 32"

# Run adaptation training  (saving the model every 2000 steps as adapted checkpoints)
python image_train.py --data_dir $DATA_DIR $MODEL_FLAGS $TRAIN_FLAGS
```


### 5) Stage B — Classification Training and Inference

This stage uses labeled data for supervised classification training.

**Example Training Command (Berlin with HSI+SAR):**
```bash
# Model flags for classification (note: class_cond True)
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
             --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine \
             --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True \
             --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"

# Run classification training with JSON config
python train_hsi_sar.py --exp exp_set/Berlin_benchmark/datasetDDPM_hsi_sar.json $MODEL_FLAGS
```


## References

- Nichol, A. Q., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. NeurIPS. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)  
- Baranchuk, D., Voynov, A., Rubachev, I., Khrulkov, V., & Babenko, A. (2022). *Label-Efficient Semantic Segmentation with Diffusion Models*. ICLR. [OpenReview](https://openreview.net/forum?id=SlxSY2UZQT)  
- Hong, D., Yokoya, N., Ge, N., Chanussot, J., & Zhu, X. X. (2021). *Spectral–Spatial Foundation Learning: Benchmark Dataset for Hyperspectral and Multimodal Remote Sensing*. ISPRS Journal. [GitHub](https://github.com/danfenghong/ISPRS_S2FL)
