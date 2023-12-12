# magic_animate_unofficial

This unofficial training code is primarily adapted from  [Magic Animate](https://github.com/magic-research/magic-animate) 
and [Animatediff](https://github.com/guoyww/AnimateDiff). 


## ToDo
- [x] **Release Training Code.**
- [ ] **Release pre-trained weights.**


## Features
- Utilizes Deepspeed training with a resolution of 768*512, and a batch size = 2 per GPU using V100-32G. 
- We've altered the condition from dense-pose to dwpose, which differs from Magic Animate. Feel free to revert this change if necessary. 
- We employ a fine-grained image prompt from  [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) instead of the EMPTY text prompt used in Magic Animate. You can revert this change if required.

```python
from animatediff.magic_animate.resampler import Resampler
# define a resampler
image_proj_model = Resampler(
                dim=cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=64,
                embedding_dim=1280,
                output_dim=cross_attention_dim,
                ff_mult=4,
            )
# extract fine-grained features of reference image for cross-attention guidance
# project from (batch_size, 257, 1280) to (batch_size, 64, 768); replace empty text embeddings with this.
encoder_hidden_states = image_proj_model(image_prompts)
```
## Requirements

see 'requirements.txt'

To support DWPose which is dependent on MMDetection, MMCV and MMPose
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```


## Data

To prepare your videos, create a JSON file with a list of video directories. 
Then, update the "json_path" value in "./configs/training/aa_train_stage1.yaml" to point to your JSON file.
You can use the public [fashion dataset](https://drive.google.com/drive/folders/17-BoVYRnG6WLymJ4q2tw-JJp_TC3u52P?usp=sharing) for fast prototyping.


## Training


```bash
# the default settings support 1 node, 8 gpus training
export HOST_NUM=1           # total nodes 
export INDEX=0              # node rank [0, 1, .., HOST_NUM]
export CHIEF_IP=localhost   # mater_ip for multi-node training
export HOST_GPU_NUM=8       # total number of gpus each node

PROCESS_NUM=$((HOST_GPU_NUM * HOST_NUM))
echo ${PROCESS_NUM}

accelerate launch --gpu_ids all --use_deepspeed --num_processes ${PROCESS_NUM} \
  --deepspeed_config_file ./configs/zero2_config.json \
  --num_machines ${HOST_NUM} --machine_rank ${INDEX} --main_process_ip ${CHIEF_IP} --main_process_port 2006 \
  --deepspeed_multinode_launcher standard \
  train.py --config configs/training/aa_train_stage1.yaml
```

## Evalutaion
The evaluation code is identical to that of [Magic Animate](https://github.com/magic-research/magic-animate).

## Acknowledgements
The majority of the network designs are provided by [magic-animate](https://github.com/magic-research/magic-animate/tree/main). 
[AnimateDiff](https://github.com/guoyww/AnimateDiff) supplies the training code. 
Additionally, there is a concurrent nice work  [unofficial animate anyone](https://github.com/guoqincode/AnimateAnyone-unofficial/tree/main).
