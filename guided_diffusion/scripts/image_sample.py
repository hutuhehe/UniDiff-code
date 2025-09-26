"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion.guided_diffusion import dist_util, logger
from guided_diffusion.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


from src.lora import  apply_lora_to_unet_model
from PIL import Image
import pdb



def save_images(tensor, save_dir='sample_images', prefix='image'):
    """
    Save a batch of images from a tensor to a directory.

    :param tensor: A torch tensor of shape [Batch, Height, Width, Channels] in uint8 format.
    :param save_dir: Directory where images will be saved.
    :param prefix: Prefix for the saved image filenames.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Iterate through the batch and save each image
    for i in range(tensor.shape[0]):
        img = tensor[i].cpu().numpy()  # Convert to NumPy array
        img_pil = Image.fromarray(img)  # Convert to PIL Image
        img_pil.save(os.path.join(save_dir, f'{prefix}_{i+1}.png'))  # Save the image

    print(f'Images saved to {save_dir}')




def main():
    args = create_argparser().parse_args()
    pdb.set_trace()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    """
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    pdb.set_trace()
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    

    """
    # model = apply_lora_to_unet_model(model)
    
    pretrained_model_path = 'checkpoints/berlin_pca_noise150/mixed_model.pth'
    #pretrained_model_path = 'checkpoints/berlin_rgb_pca_patch/model001000.pt'
    pdb.set_trace()

    state_dict = th.load(pretrained_model_path, map_location='cpu')

  
    logger.log(f"Loading pre-trained model from {pretrained_model_path}...")
    model.load_state_dict(th.load(pretrained_model_path))
    
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=2, high=3, size=(args.batch_size,), device=dist_util.dev()
            )

            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        # directly save  to sample_images folder
        save_dir = 'sample_images/berlin_pca_noise150_model2000mixed'
        save_images(sample, save_dir=save_dir, prefix='sample')
        pdb.set_trace()
        
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
