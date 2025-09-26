"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os

import numpy as np
import torch.distributed as dist

from guided_diffusion.guided_diffusion import dist_util, logger
from guided_diffusion.guided_diffusion.image_datasets import load_data
from guided_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torch as th
import time
import pdb

def main():
    args = create_argparser().parse_args()

    timestamp = time.strftime("%m%d_%H%M")
    logdir = f"exp_result/Berlin_bpd_patch/berlin_{timestamp}"
    logger.configure(logdir, format_strs=["stdout", "log"])


    dist_util.setup_dist()
    #logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # modify that load model path with finetued
    #pretrained_model_path = "checkpoints/ddpm/64x64_diffusion.pt"
    pretrained_model_path = 'checkpoints/berlin_rgb_pca_patch/model000500.pt'


  
    logger.log(f"Loading pre-trained model from {pretrained_model_path}...")
    model.load_state_dict(th.load(pretrained_model_path))
    
    
    
   # model.load_state_dict(
   #     dist_util.load_state_dict(args.model_path, map_location="cpu")
   # )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
    )

    logger.log("evaluating...")
    logger.log(f"Number of samples: {args.num_samples}")
    logger.log(f"Data directory: {args.data_dir}")
    run_bpd_evaluation(model, diffusion, data, args.num_samples, args.clip_denoised,logger)


def run_bpd_evaluation(model, diffusion, data, num_samples, clip_denoised,logger):
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0

    # since we run on a singlegpu
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        #pdb.set_trace()
        batch = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        minibatch_metrics = diffusion.calc_bpd_loop(
            model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        
        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].mean(dim=0) / world_size
            if dist.is_initialized():
                dist.all_reduce(terms)
            term_list.append(terms.detach().cpu().numpy())

        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.mean() / world_size
        if dist.is_initialized():
            dist.all_reduce(total_bpd)
        all_bpd.append(total_bpd.item())
        num_complete += world_size * batch.shape[0]
        logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd)}")
    
    for name, terms in all_metrics.items():
        out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
        logger.log(f"saving {name} terms to {out_path}")
        np.savez(out_path, np.mean(np.stack(terms), axis=0))
     
    
    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=True, num_samples=1000, batch_size=1, model_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
