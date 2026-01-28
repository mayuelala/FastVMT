import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download
import os, re, argparse


# Download models
snapshot_download("Wan-AI/Wan2.1-T2V-14B", local_dir="/root/autodl-tmp/pretrained_model/Wan-AI/Wan2.1-T2V-14B")