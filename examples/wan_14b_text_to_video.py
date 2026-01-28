import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download
import os, re, argparse


# Default model paths
DEFAULT_MODEL_DIR = "models/Wan2.1-T2V-14B"
DEFAULT_DIT_MODELS = [
    "diffusion_pytorch_model-00001-of-00006.safetensors",
    "diffusion_pytorch_model-00002-of-00006.safetensors",
    "diffusion_pytorch_model-00003-of-00006.safetensors",
    "diffusion_pytorch_model-00004-of-00006.safetensors",
    "diffusion_pytorch_model-00005-of-00006.safetensors",
    "diffusion_pytorch_model-00006-of-00006.safetensors",
]
DEFAULT_T5_MODEL = "models_t5_umt5-xxl-enc-bf16.pth"
DEFAULT_VAE_MODEL = "Wan2.1_VAE.pth"


def get_model_paths(model_dir):
    """Get model paths from the specified directory."""
    dit_paths = [os.path.join(model_dir, m) for m in DEFAULT_DIT_MODELS]
    return [
        dit_paths,
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
    pipe.enable_vram_management(num_persistent_param_in_dit=None) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

    video = VideoData("/path/to/reference_video.mp4", height=480, width=832)
    # Text-to-video with motion transfer
    video = pipe(
        prompt="Documentary photography style. An African elephant walking steadily from right to left, trunk swinging slowly, in front of towering red sand dunes and scattered desert shrubs, under clear desert sunlight, realistic documentary style, tracking shot from behind side angle.",
        negative_prompt="vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frame, cluttered background, three legs, many people in background, walking backwards",
        num_inference_steps=50,
        input_video=video,
        seed=args.seed, 
        tiled=True,
        num_frames=81,
        sf=args.sf,
        test_latency=args.test_latency,
        latency_dir=args.latency_dir,
        mode=args.mode,
    )
    save_video(video, get_next_video_path(output_dir=args.output_dir), fps=15, quality=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WanVideo Text-to-Video Example")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing model files (default: models/Wan2.1-T2V-14B)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output videos")
    #parser.add_argument("--prompt", type=str, help="Text prompt for video generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sf", type=int, default=4, help="Spatial factor for AMF computation (default: 1)")
    parser.add_argument("--test_latency", action="store_true", help="Test latency of the model")
    parser.add_argument("--latency_dir", type=str, default=None, help="Directory to save latency logs")
    parser.add_argument("--mode", type=str, default=None, choices=['No_transfer', 'AMF', 'effi_AMF'],help="Mode for the video generation, e.g., 'No_transfer'")
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main(args)






