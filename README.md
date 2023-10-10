# MAGVIT: Masked Generative Video Transformer

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/language-model-beats-diffusion-tokenizer-is/video-generation-on-ucf-101)](https://paperswithcode.com/sota/video-generation-on-ucf-101?p=language-model-beats-diffusion-tokenizer-is)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/language-model-beats-diffusion-tokenizer-is/video-prediction-on-kinetics-600-12-frames)](https://paperswithcode.com/sota/video-prediction-on-kinetics-600-12-frames?p=language-model-beats-diffusion-tokenizer-is)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/magvit-masked-generative-video-transformer/video-prediction-on-bair-robot-pushing-1)](https://paperswithcode.com/sota/video-prediction-on-bair-robot-pushing-1?p=magvit-masked-generative-video-transformer)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/magvit-masked-generative-video-transformer/video-generation-on-bair-robot-pushing)](https://paperswithcode.com/sota/video-generation-on-bair-robot-pushing?p=magvit-masked-generative-video-transformer)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/magvit-masked-generative-video-transformer/video-prediction-on-something-something-v2)](https://paperswithcode.com/sota/video-prediction-on-something-something-v2?p=magvit-masked-generative-video-transformer)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/magvit-masked-generative-video-transformer/text-to-video-generation-on-something)](https://paperswithcode.com/sota/text-to-video-generation-on-something?p=magvit-masked-generative-video-transformer)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/language-model-beats-diffusion-tokenizer-is/image-generation-on-imagenet-512x512)](https://paperswithcode.com/sota/image-generation-on-imagenet-512x512?p=language-model-beats-diffusion-tokenizer-is)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/language-model-beats-diffusion-tokenizer-is/image-generation-on-imagenet-256x256)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=language-model-beats-diffusion-tokenizer-is)


[[Paper](https://arxiv.org/abs/2212.05199)] | [[Project Page](https://magvit.cs.cmu.edu)] | [[Colab]()]

Official code and models for the CVPR 2023 paper:

**[MAGVIT: Masked Generative Video Transformer](https://arxiv.org/abs/2212.05199)** \
Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, Lu Jiang\
CVPR 2023

## Summary

We introduce MAGVIT to tackle various video synthesis tasks with a single model, where we demonstrate its quality, efficiency, and flexibility.

If you find this code useful in your research, please cite

```
@inproceedings{yu2023magvit,
  title={{MAGVIT}: Masked generative video transformer},
  author={Yu, Lijun and Cheng, Yong and Sohn, Kihyuk and Lezama, Jos{\'e} and Zhang, Han and Chang, Huiwen and Hauptmann, Alexander G and Yang, Ming-Hsuan and Hao, Yuan and Essa, Irfan and Jiang, Lu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

## Disclaimers

*Please note that this is not an officially supported Google product.*

*Checkpoints are based on training with publicly available datasets. Some datasets contain limitations, including non-commercial use limitations. Please review terms and conditions made available by third parties before using models and datasets provided.*

## Installation

There is a conda environment file for running with GPUs.
CUDA 11 and CuDNN 8.6 is required for JAX.
[This VM Image](https://console.cloud.google.com/marketplace/product/nvidia-ngc-public/nvidia-gpu-optimized-vmi) has been tested.

```sh
conda env create -f environment.yaml
conda activate magvit
```

Alternatively, you can install the dependencies via

```sh
pip install -r requirements.txt
```

## Pretrained models

Model weights and loading instructions are coming soon.

### MAGVIT 3D-VQ models

**Model**|**Size**|**Input**|**Output**|**Codebook size**|**Dataset**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
3D-VQ|B|16 frames x 64x64|4x16x16|1024| BAIR Robot Pushing
3D-VQ|L|16 frames x 64x64|4x16x16|1024| BAIR Robot Pushing
3D-VQ|B|16 frames x 128x128|4x16x16|1024| UCF-101
3D-VQ|L|16 frames x 128x128|4x16x16|1024| UCF-101
3D-VQ|B|16 frames x 128x128|4x16x16|1024| Kinetics-600
3D-VQ|L|16 frames x 128x128|4x16x16|1024| Kinetics-600
3D-VQ|B|16 frames x 128x128|4x16x16|1024| Something-Something-v2
3D-VQ|L|16 frames x 128x128|4x16x16|1024| Something-Something-v2

### MAGVIT transformers

Each transformer model must be used with its corresponding 3D-VQ tokenizer of the same dataset and model size.

**Model**|**Task**|**Size**|**Dataset**|**FVD**
:-----:|:-----:|:-----:|:-----:|:-----:
Transformer|Class-conditional|B|UCF-101 |159
Transformer|Class-conditional|L|UCF-101 |76
Transformer|Frame prediction | B | BAIR Robot Pushing |76 (48)
Transformer|Frame prediction | L | BAIR Robot Pushing |62 (31)
Transformer|Frame prediction (5) |B| Kinetics-600 |24.5
Transformer|Frame prediction (5) |L| Kinetics-600 |9.9
Transformer|Multi-task-8 | B | BAIR Robot Pushing |32.8
Transformer|Multi-task-8 | L | BAIR Robot Pushing |22.8
Transformer|Multi-task-10 | B | Something-Something-v2 | 43.4
Transformer|Multi-task-10 | L | Something-Something-v2 | 27.3

<!-- ## Usage

### Inference
Inference pretrained models in the [colab]().

### Training new models
Instructions for training new models can be [found here](). -->
