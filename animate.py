# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) MagicAnimate Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.
import argparse
import argparse
import datetime
import inspect
import os
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict

import torch

from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler

from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.magic_animate.unet_controlnet import UNet3DConditionModel
from animatediff.magic_animate.controlnet import ControlNetModel
from animatediff.magic_animate.appearance_encoder import AppearanceEncoderModel
from animatediff.magic_animate.mutual_self_attention import ReferenceAttentionControl

from animatediff.magic_animate.pipeline import AnimationPipeline as TrainPipeline
from animatediff.utils.util import save_videos_grid

from accelerate.utils import set_seed
from animatediff.utils.videoreader import VideoReader
from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path


class MagicAnimate(torch.nn.Module):
    def __init__(self,
                 config="configs/training/animation.yaml",
                 device=torch.device("cuda"),
                 train_batch_size=1, unet_additional_kwargs=None):
        super().__init__()

        print("Initializing MagicAnimate Pipeline...")
        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)

        config = OmegaConf.load(config)
        self.device = device
        self.train_batch_size = train_batch_size

        inference_config = OmegaConf.load(config.inference_config)
        motion_module = config.motion_module

        if unet_additional_kwargs is None:
            unet_additional_kwargs = OmegaConf.to_container(inference_config.unet_additional_kwargs)

        ### >>> create animation pipeline >>> ###
        self.tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")
        if config.pretrained_unet_path:
            self.unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_unet_path,
                                                                unet_additional_kwargs=unet_additional_kwargs)
        else:
            self.unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_model_path, subfolder="unet",
                                                                unet_additional_kwargs=unet_additional_kwargs)

        self.appearance_encoder = AppearanceEncoderModel.from_pretrained(config.pretrained_appearance_encoder_path,
                                                                         subfolder="appearance_encoder").to(self.device)

        if config.pretrained_vae_path is not None:
            self.vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path)
        else:
            self.vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae")

        ### Load controlnet
        self.controlnet = ControlNetModel.from_pretrained(config.pretrained_controlnet_path)

        self.vae.to(device=self.device, dtype=torch.float16)
        self.unet.to(device=self.device, dtype=torch.float16)
        self.text_encoder.to(device=self.device, dtype=torch.float16)
        self.controlnet.to(device=self.device, dtype=torch.float16)
        self.appearance_encoder.to(device=self.device, dtype=torch.float16)

        # 1. unet ckpt
        # 1.1 motion module
        if unet_additional_kwargs['use_motion_module']:
            motion_module_state_dict = torch.load(motion_module, map_location="cpu")
            if "global_step" in motion_module_state_dict: func_args.update(
                {"global_step": motion_module_state_dict["global_step"]})
            motion_module_state_dict = motion_module_state_dict[
                'state_dict'] if 'state_dict' in motion_module_state_dict else motion_module_state_dict
            try:
                # extra steps for self-trained models
                state_dict = OrderedDict()
                for key in motion_module_state_dict.keys():
                    if key.startswith("module."):
                        _key = key.split("module.")[-1]
                        state_dict[_key] = motion_module_state_dict[key]
                    else:
                        state_dict[key] = motion_module_state_dict[key]
                motion_module_state_dict = state_dict
                del state_dict
                missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
                assert len(unexpected) == 0
            except:
                _tmp_ = OrderedDict()
                for key in motion_module_state_dict.keys():
                    if "motion_modules" in key:
                        if key.startswith("unet."):
                            _key = key.split('unet.')[-1]
                            _tmp_[_key] = motion_module_state_dict[key]
                        else:
                            _tmp_[key] = motion_module_state_dict[key]
                missing, unexpected = self.unet.load_state_dict(_tmp_, strict=False)
                assert len(unexpected) == 0
                del _tmp_
            del motion_module_state_dict

        self.pipeline = TrainPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            controlnet=self.controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            # NOTE: UniPCMultistepScheduler
        ).to(device)

        self.L = config.L
        print("Initialization Done!")

    def infer(self, source_image, image_prompts, motion_sequence, random_seed, step, guidance_scale, size=(512, 768)):
        prompt = n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        samples_per_video = []
        # manually set random seed for reproduction
        if random_seed != -1:
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()

        if isinstance(motion_sequence, str):
            if motion_sequence.endswith('.mp4'):
                control = VideoReader(motion_sequence).read()
                if control[0].shape[0] != size:
                    control = [np.array(Image.fromarray(c).resize(size)) for c in control]
                control = np.array(control)
        else:
            control = motion_sequence

        # if source_image.shape[0] != size:
        #     source_image = np.array(Image.fromarray(source_image).resize((size, size)))
        H, W, C = source_image.shape

        init_latents = None
        original_length = control.shape[0]
        if control.shape[0] % self.L > 0:
            control = np.pad(control, ((0, self.L - control.shape[0] % self.L), (0, 0), (0, 0), (0, 0)), mode='edge')
        generator = torch.Generator(device=self.device)
        generator.manual_seed(torch.initial_seed())

        sample = self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            prompt_embeddings=image_prompts,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            video_length=len(control),
            controlnet_condition=control,
            init_latents=init_latents,
            generator=generator,
            appearance_encoder=self.appearance_encoder,
            source_image=source_image,
            context_frames=self.L
        ).videos

        source_images = np.array([source_image] * original_length)
        source_images = rearrange(torch.from_numpy(source_images), "t h w c -> 1 c t h w") / 255.0
        samples_per_video.append(source_images)

        control = control / 255.0
        control = rearrange(control, "t h w c -> 1 c t h w")
        control = torch.from_numpy(control)
        samples_per_video.append(control[:, :, :original_length])

        samples_per_video.append(sample[:, :, :original_length])

        samples_per_video = torch.cat(samples_per_video)

        return samples_per_video

    def forward(self, init_latents, image_prompts, timestep, source_image, motion_sequence, random_seed):
        """
        :param init_latents: the most important input during training
        :param timestep: another important input during training
        :param source_image: an image in np.array
        :param motion_sequence: np array, (f, h, w, c) (0, 255)
        :param random_seed:
        :param size: width=512, height=768 by default
        :return:
        """
        prompt = n_prompt = ""
        random_seed = int(random_seed)

        samples_per_video = []
        # manually set random seed for reproduction
        if random_seed != -1:
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()

        control = motion_sequence
        H, W, C = source_image.shape

        generator = torch.Generator(device=self.device)
        generator.manual_seed(torch.initial_seed())

        # project from (batch_size, 257, 1280) to (batch_size, 16, 768)
        image_prompts = self.unet.image_proj_model(image_prompts)

        noise_pred = self.pipeline.train(
            prompt,
            prompt_embeddings=image_prompts,
            negative_prompt=n_prompt,
            timestep=timestep,
            width=W,
            height=H,
            video_length=len(control),
            controlnet_condition=control,
            init_latents=init_latents,  # add noise to latents
            generator=generator,
            appearance_encoder=self.appearance_encoder,
            source_image=source_image,
        )

        return noise_pred
