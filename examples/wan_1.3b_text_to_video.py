import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download
import os, re
import argparse


# Default model paths
DEFAULT_MODEL_DIR = "models/Wan2.1-T2V-1.3B"
DEFAULT_DIT_MODEL = "diffusion_pytorch_model.safetensors"
DEFAULT_T5_MODEL = "models_t5_umt5-xxl-enc-bf16.pth"
DEFAULT_VAE_MODEL = "Wan2.1_VAE.pth"


def get_model_paths(model_dir):
    """Get model paths from the specified directory."""
    return [
        os.path.join(model_dir, DEFAULT_DIT_MODEL),
        os.path.join(model_dir, DEFAULT_T5_MODEL),
        os.path.join(model_dir, DEFAULT_VAE_MODEL),
    ]


def get_next_video_path(output_dir="results", prefix="video", ext=".mp4"):
    """
    Find all prefixN.ext files in output_dir and return the next available path.
    For example, if video1.mp4 and video2.mp4 exist, returns results/video3.mp4
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    nums = []
    for fn in os.listdir(output_dir):
        m = pattern.match(fn)
        if m:
            nums.append(int(m.group(1)))
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(output_dir, f"{prefix}{next_num}{ext}")
def main(args):
    # Load models
    model_manager = ModelManager(device="cpu")
    model_paths = get_model_paths(args.model_dir)
    model_manager.load_models(
        model_paths,
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)


    video = VideoData("/path/to/reference_video.mp4", height=480, width=832)
    # Text-to-video
    video = pipe(
        prompt="Documentary photography style. A lively puppy running quickly on a green grass field. The puppy has brown-yellow fur, ears perked up, with a focused and joyful expression. Sunlight shines on it, making the fur look extra soft and shiny. The background is an open grass field, occasionally dotted with wildflowers, with blue sky and white clouds visible in the distance. Strong perspective, capturing the puppy's dynamic movement and the vitality of the surrounding grass. Medium shot, side tracking view.",
        negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frame, cluttered background, three legs, many people in background, walking backwards",
        num_inference_steps=50,
        input_video=video,
        input_video_path = "/path/to/reference_video.mp4",
        seed=args.seed,
        tiled=True,
        num_frames=41,
        sf=args.sf,
        test_latency=args.test_latency,
        latency_dir=args.latency_dir,
        mode=args.mode,
    )
    save_video(video, get_next_video_path(output_dir=args.output_dir), fps=15, quality=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WanVideo Text-to-Video Example")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing model files (default: models/Wan2.1-T2V-1.3B)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output videos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_tile", action="store_true", help="Use tiled processing for large videos")
    parser.add_argument("--sf", type=int, default=5, help="Spatial factor for AMF computation (default: 1)")
    parser.add_argument("--test_latency", action="store_true", help="Test latency of the model")
    parser.add_argument("--latency_dir", type=str, default=None, help="Directory to save latency logs")
    parser.add_argument("--mode", type=str, default=None, choices=['No_transfer', 'AMF', 'effi_AMF'],help="Mode for the video generation, e.g., 'No_transfer'")
    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main(args)
