from typing_extensions import Literal, TypeAlias

# Wan Video models
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_vace import VaceWanModel

# Extensions
from ..extensions.RIFE import IFNet
from ..extensions.ESRGAN import RRDBNet


model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash, state_dict_keys_hash_with_shape, model_names, model_classes, model_resource)

    # Wan Video models
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "a61453409b67cd3246cf0c3bebad47ba", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
    (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
    (None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "civitai"),
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "dbd5ec76bbf977983f972c151d545389", ["wan_video_motion_controller"], [WanMotionControllerModel], "civitai"),

    # Extensions
    (None, "9b9313d104ac4df27991352fec013fd4", ["rife"], [IFNet], "civitai"),
    (None, "6b7116078c4170bfbeaedc8fe71f6649", ["esrgan"], [RRDBNet], "civitai"),
]

huggingface_model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (architecture_in_huggingface_config, huggingface_lib, model_name, redirected_architecture)
    ("MarianMTModel", "transformers.models.marian.modeling_marian", "translator", None),
    ("BloomForCausalLM", "transformers.models.bloom.modeling_bloom", "beautiful_prompt", None),
    ("Qwen2ForCausalLM", "transformers.models.qwen2.modeling_qwen2", "qwen_prompt", None),
]

patch_model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash_with_shape, model_name, model_class, extra_kwargs)
]

preset_models_on_huggingface = {
    # RIFE
    "RIFE": [
        ("AlexWortega/RIFE", "flownet.pkl", "models/RIFE"),
    ],
}

preset_models_on_modelscope = {
    # RIFE
    "RIFE": [
        ("Damo_XR_Lab/cv_rife_video-frame-interpolation", "flownet.pkl", "models/RIFE"),
    ],
    # ESRGAN
    "ESRGAN_x4": [
        ("AI-ModelScope/Real-ESRGAN", "RealESRGAN_x4.pth", "models/ESRGAN"),
    ],
}

Preset_model_id: TypeAlias = Literal[
    "RIFE",
    "ESRGAN_x4",
]
