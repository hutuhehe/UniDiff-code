"""
Train a diffusion model on images.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import optimizer
from torch.utils.data.dataloader import Sampler
from tqdm import tqdm
import json
import os
import copy

import torch.distributed as dist
import argparse

from guided_diffusion.guided_diffusion import dist_util, logger
from guided_diffusion.guided_diffusion.image_datasets import load_data
from guided_diffusion.guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.guided_diffusion.train_util import TrainLoop
from guided_diffusion.guided_diffusion.unet import AttentionBlock

import pdb

import loralib as lora
from src.lora import  (apply_lora_to_unet_model, print_all_lora_conv1d_layers)
from torch._utils import _unflatten_dense_tensors

def main():


    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

     # Load pre-trained weights into the model (before applying LoRA)
  
    # pretrained_model_path = 'checkpoints/ddpm/64x64_diffusion.pt'
    pretrained_model_path = 'checkpoints/ddpm/64x64_diffusion.pt'
    logger.log(f"Loading pre-trained model from {pretrained_model_path}...")
    pdb.set_trace()
    model.load_state_dict(torch.load(pretrained_model_path, map_location=dist_util.dev()))
    """
    # adjusted with lora if use lora
    model = apply_lora_to_unet_model(model)
    print_all_lora_conv1d_layers(model)
    """
    """
    # by setting requires_grad doesn't work Feb 6
    for name, param in model.named_parameters():
        if 'label_emb' not in name:
            param.requires_grad = False  # Completely stop gradient computation
    """
    # This sets requires_grad to False for all parameters without the string "lora_" in their names
    #lora.mark_only_lora_as_trainable(model)

    #lora_model = apply_lora_to_attention_blocks(model, lora_config)

    model.to(dist_util.dev())
    #model.load_state_dict(torch.load(pretrained_model_path, map_location=dist_util.dev()))
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    
    
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False,
        random_crop=True,
        random_flip=True,
    )
    
    logger.log("training...")
    
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.95",  # comma-separated list of EMA values
        log_interval=50,
        save_interval=500,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser





if __name__ == '__main__':
    main()

