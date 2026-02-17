
<div align="center">
<h2><font color="red"> FastVMT⚡️: Eliminating Redundancy in Video Motion Transfer</h2>

[Yue Ma](placeholder_url), [Zhikai Wang](placeholder_url), [Tianhao Ren](placeholder_url), [Mingzhe Zheng](placeholder_url), [Hongyu Liu](placeholder_url), [Jiayi Guo](placeholder_url), [Kunyu Feng](placeholder_url), [Yuxuan Xue](placeholder_url), [Zixiang Zhao](placeholder_url), [Konrad Schindler](placeholder_url), [Qifeng Chen](placeholder_url), [Linfeng Zhang](placeholder_url)


<strong>is Accpeted by ICLR 2026</strong>
 
<a href='placeholder_arxiv_url'><img src='https://img.shields.io/badge/ArXiv-2602.05551-red'></a>
<a href='https://fastvmt.github.io/'>
  <img src='https://img.shields.io/badge/Project-Page-Green'>
</a>
[![GitHub](https://img.shields.io/github/stars/mayuelala/FastVMT?style=social)](https://github.com/mayuelala/FastVMT)

</div>
<!-- Add demo GIFs here -->
<!--
<table class="center">
  <td><img src="docs/gif_results/demo1.gif"></td>
  <td><img src="docs/gif_results/demo2.gif"></td>
  <tr>
  <td width=25% style="text-align:center;">"source → target"</td>
  <td width=25% style="text-align:center;">"source → target"</td>
</tr>
</table>
-->

## 🎏 Abstract
<b>TL; DR: <font color="red">FastVMT</font> eliminates redundancy in video motion transfer, enabling fast and efficient motion pattern transfer from reference videos to generated content.</b>

<details><summary>CLICK for the full abstract</summary>

> Video motion transfer aims to synthesize videos by generating visual content according to a text prompt while transferring the motion pattern observed in a reference video. Recent methods predominantly use the Diffusion Transformer (DiT) architecture. To achieve satisfactory runtime, several methods attempt to accelerate the computations in the DiT, but fail to address structural sources of inefficiency. In this work, we identify and remove two types of computational redundancy in earlier work: \emph{\textbf{motion redundancy}} arises because the generic DiT architecture does not reflect the fact that frame-to-frame motion is small and smooth; \emph{\textbf{gradient redundancy}} occurs if one ignores that gradients change slowly along the diffusion trajectory. To mitigate motion redundancy, we mask the corresponding attention layers to a local neighborhood such that interaction weights are not computed unnecessarily distant image regions. To exploit gradient redundancy, we design an optimization scheme that reuses gradients from previous diffusion steps and skips unwarranted gradient computations. On average, FastVMT achieves a \textit{\textcolor{Blue}{\textbf{3.43}}}$\times$ speedup without degrading the visual fidelity or the temporal consistency of the generated videos. 

</details>

## 📀 Demo Video

https://github.com/user-attachments/assets/a4c0a8e5-578a-4534-93aa-c1a1960edb37


## 📋 Changelog

- 2026.01.28 Initial release with efficient tile-based AMF support

## 🚧 Todo

- [ ] Add more examples and demo videos
- [ ] Add support with CPU-offload to support low VRAM GPUs

## ✨ Features

- **Attention Motion Flow (AMF)**: Custom implementation for transferring motion patterns from reference videos to generated content
- **Efficient Tile-based AMF**: Optimized computation with reduced memory usage while maintaining accuracy
- **Flexible Inference Modes**: Support for multiple generation modes (`effi_AMF`, `No_transfer`)
- **VRAM Management**: Built-in CPU offload strategies for running on consumer GPUs

## 🛡 Setup Environment


```bash
# Create conda environment
conda create -n fastvmt python=3.10
conda activate fastvmt

# Install dependencies
cd FastVMT
pip install -e . # Editing mode
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 12.x
- 80GB+ GPU VRAM recommended (can run on lower VRAM with CPU offload)

## 📥 Model Download

We support two model variants:

| Model | VRAM Required | Quality |
|-------|---------------|---------|
| Wan2.1-T2V-1.3B | ~24GB | Good |
| Wan2.1-T2V-14B | ~80GB | Best |

### Using download script (Recommended)

```bash
# Download 14B model (default, best quality)
python examples/download_model.py --model 14b

# Download 1.3B model (lower VRAM requirement)
python examples/download_model.py --model 1.3b

# Download both models
python examples/download_model.py --model all
```

<details><summary>Or use ModelScope CLI directly:</summary>

```bash
# Download 14B model
modelscope download --model Wan-AI/Wan2.1-T2V-14B --local_dir ./models/Wan2.1-T2V-14B

# Download 1.3B model
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir ./models/Wan2.1-T2V-1.3B
```

</details>

## ⚔️ FastVMT Editing

#### Quick Start

```bash
# Using 1.3B model (lower VRAM)
python examples/wan_1.3b_text_to_video.py

# Using 14B model (better quality)
python examples/wan_14b_text_to_video.py
```

#### Python API

```python
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models([
    "models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
], torch_dtype=torch.bfloat16)

pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# Load reference video for motion transfer
ref_video = VideoData("data/source.mp4", height=480, width=832)

# Generate with motion transfer (num_frames auto-inferred from input_video)
video = pipe(
    prompt="Documentary photography style. A lively puppy running quickly on a green grass field. The puppy has brown-yellow fur, ears perked up, with a focused and joyful expression. Sunlight shines on it, making the fur look extra soft and shiny. The background is an open grass field, occasionally dotted with wildflowers, with blue sky and white clouds visible in the distance. Strong perspective, capturing the puppy's dynamic movement and the vitality of the surrounding grass. Medium shot, side tracking view.",
    negative_prompt="low quality, blurry",
    num_inference_steps=50,
    denoising_strength=0.75,
    input_video=ref_video,
    seed=42,
    tiled=True,
    sf=4,
    mode="effi_AMF",
)
save_video(video, "output.mp4", fps=15, quality=5)
```

#### Motion Transfer Modes

| Mode | Description | Speed | Accuracy |
|------|-------------|-------|----------|
| `effi_AMF` | Efficient tile-based Attention Motion Flow (default) | Fast | Good |
| `No_transfer` | Standard generation without motion transfer | Fastest | N/A |

## 📁 Project Structure

<details><summary>Click for directory structure</summary>

```
FastVMT/
├── diffsynth/                    # Core library
│   ├── models/                   # Model implementations
│   │   ├── wan_video_dit.py     # Modified DiT with Q/K extraction
│   │   ├── wan_video_vae.py     # Video VAE encoder/decoder
│   │   └── wan_video_text_encoder.py
│   ├── pipelines/               # Inference pipelines
│   │   └── wan_video.py         # Pipeline with AMF implementation
│   ├── schedulers/              # Noise schedulers (Flow Matching)
│   ├── prompters/               # Prompt processing
│   └── vram_management/         # Memory optimization utilities
├── examples/                    # Example scripts
├── models/                      # Model checkpoints
├── requirements.txt            # Dependencies
└── setup.py                    # Package setup
```

</details>

## 🔧 Key Modifications

This repository includes the following modifications to the original DiffSynth-Studio:

### 1. `diffsynth/models/wan_video_dit.py`
- Added Q/K tensor extraction in self-attention layers for AMF computation
- Custom forward pass preserving spatial size information

### 2. `diffsynth/pipelines/wan_video.py`
- Implemented Attention Motion Flow (AMF) computation algorithm
- Added efficient tile-based AMF variant for reduced memory usage
- Integrated guidance optimization steps for motion transfer
- Added tracking loss for improved temporal consistency

## 📍 Citation

If you use this code, please cite:

```bibtex
@article{ma2025fastvmt,
  title={FastVMT: Eliminating Redundancy in Video Motion Transfer},
  author={Ma, Yue and Wang, Zhikai and Ren, Tianhao and Zheng, Mingzhe and Liu, Hongyu and Guo, Jiayi and Feng, Kunyu and Xue, Yuxuan and Zhao, Zixiang and Schindler, Konrad and Chen, Qifeng and Zhang, Linfeng},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 📜 License

This project is open source and licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

## 💗 Acknowledgements

This repository borrows heavily from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and [Wan Video](https://github.com/Wan-Video/Wan2.1). Thanks to the authors for sharing their code and models.

## 🧿 Maintenance

This is the codebase for our research work. If you have any questions or ideas to discuss, feel free to open an issue.


