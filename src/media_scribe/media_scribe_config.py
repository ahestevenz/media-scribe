# -*- coding: utf-8 -*-
from __future__ import annotations

from enum import Enum
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator


class ModelImageType(str, Enum):
    CIVITAI_TEST = "civitai_test"
    CIVITAI = "civitai"
    SD_3 = "sd_3"
    SD_XL = "sd_xl"
    PIX_2_PIX = "pix_2_pix"
    SD_1_5_IMG_2_IMG = "sd_1_5_img_2_img"


class ModelTextType(str, Enum):
    LLAMA_8B = "llama_8b"
    LLAMA_70B = "llama_70b"


class OutputType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class LlamaModelScribeConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: ModelTextType = ModelTextType.LLAMA_8B
    model_name: Path
    max_num_historical_messages: int = 5
    system_prompt: str
    max_tokens: int = 1024
    max_input_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9

    @field_validator("max_tokens", "max_input_tokens", mode="before")
    @classmethod
    def max_tokens_must_be_positive(cls, value: int, info: ValidationInfo) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return value

    @field_validator("max_num_historical_messages", mode="before")
    @classmethod
    def check_history_limit(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(
                "max_num_historical_messages must be a positive integer",
            )
        return value

    @property
    def model_path(self) -> Path:
        return self.model_name

    def get_generate_args(self) -> dict:
        """Return the arguments required for the generate function."""
        return {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": True,
        }


class StableDiffusionScribeConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: ModelImageType = ModelImageType.CIVITAI
    model_paths: dict[ModelImageType, list[Path | None]]
    load_refiner: bool = True
    num_inference_steps: int = 50
    guidance_scale: float = 0.7
    root_models_path: Path
    root_output_dir: Path

    @property
    def base_model_path(self) -> Path | None:
        """Return the model path based on the selected model_type."""
        return self.model_paths[self.model_type][0]

    @property
    def refiner_model_path(self) -> Path | None:
        """Return the refiner model path based on the selected model_type."""
        return self.model_paths[self.model_type][1]


class MediaScribeConfig(BaseModel):
    sd_config: StableDiffusionScribeConfig
    llama_config: LlamaModelScribeConfig
    device: str = "mps"
    verbose: bool = False

    @field_validator("device", mode="before")
    @classmethod
    def check_device(cls, v: str, info: ValidationInfo) -> str:
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
    def from_yaml(cls, file_path: str | Path) -> MediaScribeConfig:
        """Load the configuration from a YAML file."""
        with open(file_path) as file:
            config_data = yaml.safe_load(file)

        llama_config_data = config_data["llama_config"]
        sd_config_data = config_data["sd_config"]

        root_models_path = Path(sd_config_data["root_models_path"])
        model_paths = {
            ModelImageType(k): [
                root_models_path / p if p else None for p in v
            ]
            for k, v in sd_config_data.pop("model_paths").items()
        }

        llama_config = LlamaModelScribeConfig(**llama_config_data)
        sd_config = StableDiffusionScribeConfig(
            **sd_config_data, model_paths=model_paths)

        return cls(
            llama_config=llama_config,
            sd_config=sd_config,
            device=config_data["device"],
            verbose=config_data.get("verbose", False),
        )

    def _convert_to_serializable(self, obj: object) -> object:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        return obj

    def to_yaml(self, file_path: Path) -> None:
        """Save the current instance's configuration to a YAML file."""
        config_data = self.model_dump()
        with open(file_path, "w") as file:
            yaml.dump(self._convert_to_serializable(config_data), file)
