from animate import MagicAnimate
import glob
import torch
import random
from collections import OrderedDict
from animatediff.utils.util import save_videos_grid
import numpy as np
from animatediff.data.dataset import PexelsDataset
from einops import rearrange
from PIL import Image
from animate import init_dwpose
import json
import os


def seed_everything(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # Numpy
    torch.manual_seed(seed_value)  # PyTorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed_value)


seed_value = 42  # choose any seed value you want
seed_everything(seed_value)

if __name__ == "__main__":

    prompts = json.load(open("./fashion_test_vidoes.json"))
    random.shuffle(prompts)

    batch_size = 1
    device = torch.device("cuda:0")
    val_video_length = 16
    sample_size = (768, 512)

    # load model
    model = MagicAnimate(device=device, L=16)

    # load pretrained weight
    model_checkpoint_path = "./models/dwpose_animate.ckpt"

    model_checkpoint_path = torch.load(model_checkpoint_path, map_location="cpu")
    state_dict = model_checkpoint_path["state_dict"] if "state_dict" in model_checkpoint_path else model_checkpoint_path
    new_td = OrderedDict()
    for k, v in state_dict.items():
        new_td[k[len('module.'):]] = v
    info = model.load_state_dict(new_td, strict=False)
    print(info)

    # load dwpose model
    dwpose_model = init_dwpose(device)
    model.to(device, dtype=torch.float16)

    # load dataset
    video_data = PexelsDataset(json_path=prompts,
                               sample_size=sample_size,
                               is_test=True,
                               sample_n_frames=val_video_length,
                               sample_stride=1)

    for idx, fname in enumerate(prompts):
        video_name = fname.split("/")[-1]

        # get driving videos
        pixels = video_data[idx]['pixel_values']
        pixel_values_val = pixels.unsqueeze(0)

        with torch.inference_mode():
            # get pose conditions
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
            dwpose_conditions = torch.from_numpy(dwpose_conditions).unsqueeze(0) / 255.0
            dwpose_conditions = rearrange(dwpose_conditions, 'b f h w c -> b f c h w')

            # get reference image
            ref_pil_images_val = []
            for batch_id in range(pixel_values_val.shape[0]):
                frame_idx = random.randint(0, val_video_length - 1)
                image_np = pixel_values_val[batch_id, frame_idx].permute(1, 2, 0).numpy()
                image_np = (image_np * 0.5 + 0.5) * 255
                ref_pil_image = Image.fromarray(image_np.astype(np.uint8))
                ref_pil_images_val.append(ref_pil_image)

            # infer a video
            sample = model.infer(
                source_image=np.array(ref_pil_images_val[0]),
                image_prompts=None,
                motion_sequence=dwpose_conditions,
                random_seed=42,
                step=25,
                guidance_scale=2,
                size=(sample_size[1], sample_size[0])
            )

            # save a video
            save_videos_grid(sample, f"sample_{idx}.mp4", fps=8)
