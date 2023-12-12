import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from animatediff.data.dataset import WebVid10M, PexelsDataset
from animatediff.utils.util import save_videos_grid, pad_image
from controlnet_aux import DWposeDetector
from accelerate import Accelerator
from einops import repeat
from animate import MagicAnimate


def main(

        # >>>>>> new params >>>>>> #
        image_encoder_path: str,

        # <<<<<< new params <<<<<< #

        image_finetune: bool,

        name: str,
        use_wandb: bool,
        launcher: str,

        output_dir: str,
        pretrained_model_path: str,
        pretrained_appearance_encoder_path: str,

        train_data: Dict,
        validation_data: Dict,
        cfg_random_null_text: bool = True,
        cfg_random_null_text_ratio: float = 0.1,

        unet_checkpoint_path: str = "",
        unet_additional_kwargs: Dict = {},
        ema_decay: float = 0.9999,
        noise_scheduler_kwargs=None,

        max_train_epoch: int = -1,
        max_train_steps: int = 100,
        validation_steps: int = 100,
        validation_steps_tuple: Tuple = (-1,),

        learning_rate: float = 3e-5,
        scale_lr: bool = False,
        lr_warmup_steps: int = 0,
        lr_scheduler: str = "constant",

        trainable_modules: Tuple[str] = (None,),
        num_workers: int = 8,
        train_batch_size: int = 1,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        checkpointing_epochs: int = 5,
        checkpointing_steps: int = -1,

        mixed_precision_training: bool = True,
        enable_xformers_memory_efficient_attention: bool = True,

        global_seed: int = 42,
        is_debug: bool = False,
):
    # Accelerate
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
    )
    seed = global_seed + accelerator.process_index
    torch.manual_seed(seed)

    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        # OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    if accelerator.state.deepspeed_plugin is not None and \
            accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto":
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = train_batch_size

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    # load dwpose detector, see controlnet_aux: https://github.com/patrickvonplaten/controlnet_aux
    # specify configs, ckpts and device, or it will be downloaded automatically and use cpu by default
    det_config = '/yolox_l_8xb8-300e_coco.py'
    det_ckpt = '/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
    pose_config = '/dwpose-l_384x288.py'
    pose_ckpt = '/dw-ll_ucoco_384.pth'

    local_rank = accelerator.device

    dwpose_model = DWposeDetector(
        det_config=det_config,
        det_ckpt=det_ckpt,
        pose_config=pose_config,
        pose_ckpt=pose_ckpt,
        device=local_rank
    )
    # -------- magic_animate --------#
    model = MagicAnimate(train_batch_size=train_batch_size,
                         device=local_rank,
                         unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs))

    # ----- load image encoder ----- #
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
    image_encoder.requires_grad_(False)
    image_processor = CLIPImageProcessor()

    # Set trainable parameters
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if accelerator.is_main_process:
        accelerator.print(f"trainable params number: {len(trainable_params)}")
        accelerator.print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
        model.appearance_encoder.enable_gradient_checkpointing()
        model.controlnet.enable_gradient_checkpointing()

    model.unet.enable_xformers_memory_efficient_attention()
    model.appearance_encoder.enable_xformers_memory_efficient_attention()
    model.controlnet.enable_xformers_memory_efficient_attention()

    weight_type = torch.float16
    model.to(local_rank, dtype=weight_type)
    image_encoder.to(local_rank, dtype=weight_type)

    # Get the training dataset
    train_dataset = PexelsDataset(**train_data)

    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    if accelerator.is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")

    model, optimizer = accelerator.prepare(model, optimizer)

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        model.train()

        for step, batch in enumerate(train_dataloader):

            # Data batch sanity check
            if global_step % 1000 == 0:
                # pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = batch['pixel_values'].cpu()
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                for idx, pixel_value in enumerate(pixel_values):
                    pixel_value = pixel_value[None, ...]
                    save_videos_grid(pixel_value,
                                     f"{output_dir}/sanity_check/global_{global_step}.gif",
                                     rescale=True)

            ### >>>> Training >>>> ###

            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank, dtype=weight_type)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = model.vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = model.vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # >>>>>>>>>>>> Get the image embedding for conditioning >>>>>>>>>>>>#
            with torch.no_grad():
                ref_pil_images = []
                encoder_hidden_states = []

                for batch_id in range(batch['pixel_values'].shape[0]):
                    # get one frame randomly
                    frame_idx = random.randint(0, video_length - 1)
                    image_np = batch['pixel_values'][batch_id, frame_idx].permute(1, 2, 0).numpy()
                    image_np = (image_np * 0.5 + 0.5) * 255
                    ref_pil_image = Image.fromarray(image_np.astype(np.uint8))
                    ref_pil_images.append(ref_pil_image)

                    # debug
                    # if accelerator.is_main_process:
                    #     ref_pil_images[0].save("ref_img.jpg")

                    # get fine-grained embeddings
                    ref_pil_image_pad = pad_image(ref_pil_image)
                    clip_image = image_processor(images=ref_pil_image_pad, return_tensors="pt").pixel_values
                    image_emb = image_encoder(clip_image.to(local_rank, dtype=weight_type),
                                              output_hidden_states=True).hidden_states[-2]
                    encoder_hidden_states.append(image_emb)

                encoder_hidden_states = torch.cat(encoder_hidden_states)

            # <<<<<<<<<<< Get the image embedding for conditioning <<<<<<<<<<<<<#

            # >>>>>>>>>>>> get dwpose conditions >>>>>>>>>>>> #
            with torch.inference_mode():
                video_values = rearrange(pixel_values, "b c h w -> b h w c")
                image_np = (video_values * 0.5 + 0.5) * 255
                image_np = image_np.cpu().numpy().astype(np.uint8)
                num_frames = image_np.shape[0]

                dwpose_conditions = []
                for frame_id in range(num_frames):
                    pil_image = Image.fromarray(image_np[frame_id])
                    dwpose_image = dwpose_model(pil_image, output_type='np', image_resolution=pixel_values.shape[-1])
                    dwpose_conditions.append(dwpose_image)

                    # debug
                    # if accelerator.is_main_process:
                    #     img = Image.fromarray(dwpose_image)
                    #     img.save(f"pose_{frame_id}.jpg")

                dwpose_conditions = np.array(dwpose_conditions)


            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            model_pred = model(init_latents=noisy_latents,
                               image_prompts=encoder_hidden_states,
                               timestep=timesteps,
                               source_image=np.array(ref_pil_images[0]),  # FIXME: only support train_batch_size=1
                               motion_sequence=dwpose_conditions,
                               random_seed=seed)
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # use accelerator
            accelerator.backward(loss, retain_graph=True)
            accelerator.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            ### <<<< Training <<<< ###
            is_main_process = accelerator.is_main_process

            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)

            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": model.state_dict(),
                }
                if step == len(train_dataloader) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch + 1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")

            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []

                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)
                prompts = validation_data['prompt_videos']
                val_video_length = validation_data['val_video_length']

                for idx, prompt in enumerate(prompts):

                    batch_size = 1  # FIXME
                    sample_size = (768, 512)
                    video_data = PexelsDataset(json_path=[prompt],
                                               sample_size=sample_size,  # for fashion dataset
                                               is_test=True,
                                               sample_n_frames=val_video_length,
                                               sample_stride=1)

                    pixel_values_val = []
                    for bsz_idx in range(batch_size):
                        pixels = video_data[bsz_idx]['pixel_values']
                        pixel_values_val.append(pixels.unsqueeze(0))

                    pixel_values_val = torch.cat(pixel_values_val, dim=0)

                    # get pose conditions
                    with torch.inference_mode():
                        video_values = rearrange(pixel_values_val, "b f c h w -> (b f) h w c")
                        image_np = (video_values * 0.5 + 0.5) * 255
                        image_np = image_np.cpu().numpy().astype(np.uint8)
                        num_frames = image_np.shape[0]

                        dwpose_conditions = []
                        for frame_id in range(num_frames):
                            pil_image = Image.fromarray(image_np[frame_id])
                            dwpose_image = dwpose_model(pil_image, output_type='np',
                                                        image_resolution=pixel_values_val.shape[-1])
                            dwpose_conditions.append(dwpose_image)
                        dwpose_conditions = np.array(dwpose_conditions)

                    # get reference image
                    ref_pil_images_val = []
                    encoder_hidden_states_val = []
                    with torch.inference_mode():
                        for batch_id in range(pixel_values_val.shape[0]):
                            # get one frame randomly
                            frame_idx = random.randint(0, val_video_length - 1)
                            image_np = pixel_values_val[batch_id, frame_idx].permute(1, 2, 0).numpy()
                            image_np = (image_np * 0.5 + 0.5) * 255
                            ref_pil_image = Image.fromarray(image_np.astype(np.uint8))
                            ref_pil_images_val.append(ref_pil_image)

                            # get fine-grained embeddings
                            ref_pil_image_pad = pad_image(ref_pil_image)
                            clip_image = image_processor(images=ref_pil_image_pad, return_tensors="pt").pixel_values
                            image_emb = image_encoder(clip_image.to(local_rank, dtype=weight_type),
                                                      output_hidden_states=True).hidden_states[-2]

                            # negative image embeddings
                            image_np_neg = np.zeros_like(image_np)
                            ref_pil_image_neg = Image.fromarray(image_np_neg.astype(np.uint8))
                            ref_pil_image_pad = pad_image(ref_pil_image_neg)
                            clip_image_neg = image_processor(images=ref_pil_image_pad, return_tensors="pt").pixel_values
                            image_emb_neg = image_encoder(clip_image_neg.to(local_rank, dtype=weight_type),
                                                          output_hidden_states=True).hidden_states[-2]

                            image_emb = torch.cat([image_emb_neg, image_emb])

                            encoder_hidden_states_val.append(image_emb)

                        encoder_hidden_states_val = torch.cat(encoder_hidden_states_val)

                    sample = model.infer(
                        source_image=np.array(ref_pil_images_val[0]),
                        image_prompts=encoder_hidden_states_val,
                        motion_sequence=dwpose_conditions,
                        random_seed=seed,
                        step=validation_data['num_inference_steps'],
                        guidance_scale=validation_data['guidance_scale'],
                        size=(sample_size[1], sample_size[0])
                    )
                    save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                    samples.append(sample)

                samples = torch.concat(samples)
                save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                save_videos_grid(samples, save_path)
                logging.info(f"Saved samples to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
