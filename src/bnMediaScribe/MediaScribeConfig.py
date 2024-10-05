from pathlib import Path
from pydantic import BaseModel, FieldValidationInfo, field_validator
from enum import Enum
import torch
import yaml

class ModelType(str, Enum):
    SD_15 = "sd_1.5"
    SD_3 = "sd_3"
    SD_XL = "sd_xl"
    CIVITAI = "civitai"
    LLAMA_8B = "llama_8b"
    LLAMA_70B = "llama_70b"

class OutputType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"

class LlamaModelScribeConfig(BaseModel):
    model_name: Path
    max_num_historical_messages: int = 5
    system_prompt: str
    model_size: str = "8B"  # Options: '8B', '70B'
    max_tokens: int = 1024
    max_input_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    
    @field_validator("max_tokens", "max_input_tokens", mode="before")
    def max_tokens_must_be_positive(cls, value: int, info: FieldValidationInfo):
        if value <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return value
    
    @field_validator("max_num_historical_messages", mode="before")
    def check_history_limit(cls, value):
        if value <= 0:
            raise ValueError('max_num_historical_messages must be a positive integer')
        return value
    
    @property
    def model_path(self):
        return self.model_dir/self.model_filename
    
    class Config:
        protected_namespaces = ()
    
    def get_generate_args(self):
        """Return the arguments required for the generate function."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
    

class StableDiffusionScribeConfig(BaseModel):
    model_type: ModelType  # Choose between SD 1.5, SDXL, or CivitAI
    model_dir: Path
    base_model_filename: Path # Required
    refiner_model_filename: Path # Required
    output_type: OutputType = OutputType.IMAGE  # Can be IMAGE or VIDEO
    output_dir: OutputType = OutputType.IMAGE  # Can be IMAGE or VIDEO
    num_frames: int = 30  # Only relevant for videos
    fps: int = 10  # Frames per second for video generation
    
    # A dictionary mapping model types to their respective model paths
    model_paths = {
        ModelType.SD_15: Path("/models/sd_1.5"),
        ModelType.SD_3: Path("/models/sd_3"),
        ModelType.SD_XL: Path("/models/sd_xl"),
        ModelType.CIVITAI: Path("/models/civitai"),
        ModelType.LLAMA_8B: Path("/models/llama_8b"),
        ModelType.LLAMA_70B: Path("/models/llama_70b")
    }

    @property
    def  base_model_path(self):
        """Return the model path based on the selected model_type."""
        return self.model_paths[self.model_type]
    
    @property
    def  refiner_model_path(self):
        """Set model_dir based on the model_type."""
        self.model_dir = self.get_model_path()
    
    
    class Config:
        protected_namespaces = ()

class MediaScribeConfig(BaseModel):
    sd_config: StableDiffusionScribeConfig = None
    llama_config: LlamaModelScribeConfig = None
    root_ouput_path: Path = None
    device: str = "mps"
    verbose: bool = False
    
    @field_validator("device", mode="before")
    def check_device(cls, v, info: FieldValidationInfo):
        if v not in ["cpu", "cuda", "mps"]:
            raise ValueError("Device must be either 'cpu', 'cuda', or 'mps'")
        if v == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this machine, please use 'cpu'")
        if v == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available on this machine, please use 'cpu'")
        return v

    def get_model_path(self):
        # If LLaMA config is present, return LLaMA model path
        if self.llama_config:
            return self.llama_config.model_path
        
        # If Stable Diffusion config is present, return SD model path
        if self.sd_config:
            return self.sd_config.model_path
        
        raise ValueError("No valid configuration found for model path.")
    
    @classmethod
    def from_yaml(cls, file_path: str):
        """Load the configuration from a YAML file."""
        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        return cls(**config_data)
