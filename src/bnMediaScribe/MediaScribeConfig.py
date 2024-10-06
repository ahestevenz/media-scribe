# -*- coding: utf-8 -*-
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import ClassVar
from typing import Dict
from typing import List

import torch
import yaml
from pydantic import BaseModel
from pydantic import field_validator
from pydantic import FieldValidationInfo


class ModelImageType(str, Enum):
    CIVITAI_TEST = "civitai_test"
    CIVITAI = "civitai"
    SD_3 = "sd_3"
    SD_XL = "sd_xl"


class ModelTextType(str, Enum):
    LLAMA_8B = "llama_8b"
    LLAMA_70B = "llama_70b"


class OutputType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class LlamaModelScribeConfig(BaseModel):
    model_type: ModelTextType = ModelTextType.LLAMA_8B
    model_name: Path
    max_num_historical_messages: int = 5
    system_prompt: str
    max_tokens: int = 1024
    max_input_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9

    @field_validator("max_tokens", "max_input_tokens", mode="before")
    def max_tokens_must_be_positive(
        cls,
        value: int,
        info: FieldValidationInfo,
    ):
        if value <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return value

    @field_validator("max_num_historical_messages", mode="before")
    def check_history_limit(cls, value):
        if value <= 0:
            raise ValueError(
                "max_num_historical_messages must be a positive integer",
            )
        return value

    @property
    def model_path(self):
        return self.model_dir / self.model_filename

    class Config:
        protected_namespaces = ()

    def get_generate_args(self):
        """Return the arguments required for the generate function."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


class StableDiffusionScribeConfig(BaseModel):
    model_type: ModelImageType = (
        ModelImageType.CIVITAI
    )  # Choose between SD 1.5, SDXL, or CivitAI
    load_refiner: bool = True
    num_inference_steps: int = 50
    guidance_scale: float = 0.7
    root_models_path: Path
    root_output_dir: Path

    # A dictionary mapping model types to their respective model paths
    model_paths: ClassVar[Dict[ModelImageType, List[Path]]] = {
        ModelImageType.SD_3: [
            Path(
                "models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671/"
            ),
            Path("models--Stable-Diffusion-XL/sd_xl_refiner_1.0.safetensors"),
        ],
        ModelImageType.SD_XL: [
            Path("models--Stable-Diffusion-XL/sd_xl_base_1.0.safetensors"),
            Path("models--Stable-Diffusion-XL/sd_xl_refiner_1.0.safetensors"),
        ],
        ModelImageType.CIVITAI: [
            Path(
                "models--civitai/juggernautXL_juggXIByRundiffusion.safetensors"
            ),
            Path("models--Stable-Diffusion-XL/sd_xl_refiner_1.0.safetensors"),
        ],
    }

    @property
    def base_model_path(self):
        """Return the model path based on the selected model_type."""
        return self.root_models_path / self.model_paths[self.model_type][0]

    @property
    def refiner_model_path(self):
        """Set model_dir based on the model_type."""
        return self.root_models_path / self.model_paths[self.model_type][1]

    class Config:
        protected_namespaces = ()


class MediaScribeConfig(BaseModel):
    sd_config: StableDiffusionScribeConfig
    llama_config: LlamaModelScribeConfig
    device: str = "mps"
    verbose: bool = False

    @field_validator("device", mode="before")
    def check_device(cls, v, info: FieldValidationInfo):
        if v not in ["cpu", "cuda", "mps"]:
            raise ValueError("Device must be either 'cpu', 'cuda', or 'mps'")
        if v == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA is not available on this machine, please use 'cpu'",
            )
        if v == "mps" and not torch.backends.mps.is_available():
            raise ValueError(
                "MPS is not available on this machine, please use 'cpu'",
            )
        return v

    @classmethod
    def from_yaml(cls, file_path: str):
        """Load the configuration from a YAML file."""
        with open(file_path) as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)

    def _convert_to_serializable(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, dict):
            return {
                k: self._convert_to_serializable(v) for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        return obj

    def to_yaml(self, file_path: Path):
        """Save the current instance's configuration to a YAML file."""
        config_data = self.dict()
        with open(file_path, "w") as file:
            yaml.dump(self._convert_to_serializable(config_data), file)
