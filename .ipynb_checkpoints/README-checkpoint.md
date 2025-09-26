
# One Pixel, Two Feature Sets: Enhancing Hyperspectral Image Segmentation with Diffusion-Model Extracted Spatial Features


## Dataset

The experiment is evaluated on two Hyperspectral Images (HSI): Augsburg and Berlin. 

The datasets can be downloaded from [this GitHub repository](https://github.com/danfenghong/ISPRS_S2FL

## Pre-trained model 
We use 64x64 pre-trained diffusion model with ImageNet, the pre-trained diffusion model can be download from https://github.com/openai/guided-diffusion, 64x64 diffusion: 64x64_diffusion.pt

### Data preprocessing 
We first divide the HSI cubes into 64x64 spatial size segments. Please refer to the notebook `hyperspectral snip Belin.ipynb' and 'hyperspectral snip Augsburg.ipynb' for detailed steps.



### Note: the code is modified based on https://github.com/openai/guided-diffusion and https://github.com/yandex-research/ddpm-segmentation. Thanks for generously sharing code.
