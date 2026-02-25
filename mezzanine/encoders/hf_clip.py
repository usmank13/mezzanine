from __future__ import annotations

from dataclasses import dataclass

from ..registry import ENCODERS

from .hf_vision import HFVisionEncoder, HFVisionEncoderConfig


@dataclass
class HFCLIPVisionEncoderConfig(HFVisionEncoderConfig):
    model_name: str = "openai/clip-vit-base-patch32"
    # CLIP is often stable with mean pooling, but allow overrides.


class HFCLIPVisionEncoder(HFVisionEncoder):
    """CLIP vision encoder wrapper.

    This is just HFVisionEncoder with a different default checkpoint name.
    """

    NAME = "clip_vision"
    DESCRIPTION = "CLIP vision backbone (HF) with configurable pooling/layer."

    def __init__(self, cfg: HFCLIPVisionEncoderConfig, device: str = "cuda"):
        super().__init__(cfg, device=device)


# Register
ENCODERS.register("clip_vision")(HFCLIPVisionEncoder)
