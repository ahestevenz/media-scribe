# -*- coding: utf-8 -*-
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from diffusers import (
    StableDiffusion3Pipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image
from transformers import CLIPTokenizer

from media_scribe.media_scribe_config import MediaScribeConfig, ModelImageType

from loguru import logger


class ImageVideoScribe:
    def __init__(self, config: MediaScribeConfig):
        self.config = config
        self.verbose = config.verbose
        self.load_refiner = self.config.sd_config.load_refiner
        self.load_img2img = False
        self.generated_directory = self._generate_directory()
        self.device = torch.device(config.device)
        self._clip_tokenizer_truncate = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self._clip_tokenizer_split = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self._load_model_pipelines()

    def _generate_directory(self) -> Path:
        current_datetime = datetime.now()
        directory = (
            self.config.sd_config.root_output_dir
            / f"{current_datetime.strftime('%Y%m%d_%H%M%S')}_{self.config.sd_config.model_type}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _load_model_pipelines(self) -> None:
        match self.config.sd_config.model_type:
            case (
                ModelImageType.CIVITAI
                | ModelImageType.CIVITAI_TEST
                | ModelImageType.SD_XL
            ):
                self.base_model_pipe = StableDiffusionXLPipeline.from_single_file(
                    self.config.sd_config.base_model_path,
                    torch_dtype=torch.float16,
                ).to(self.device)
            case ModelImageType.SD_3:
                self.base_model_pipe = StableDiffusion3Pipeline.from_single_file(
                    self.config.sd_config.base_model_path,
                    torch_dtype=torch.float16,
                ).to(self.device)
            case ModelImageType.PIX_2_PIX:
                self.base_model_pipe = (
                    StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        self.config.sd_config.base_model_path,
                        torch_dtype=torch.float16,
                    ).to(self.device)
                )
                self.load_img2img = True
            case ModelImageType.SD_1_5_IMG_2_IMG:
                self.base_model_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.config.sd_config.base_model_path,
                    torch_dtype=torch.float16,
                ).to(self.device)
                self.load_img2img = True
            case _:
                raise NotImplementedError("Method does not exist!")

        if self.load_refiner and not self.load_img2img:
            self.refiner_model_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                self.config.sd_config.refiner_model_path,
                torch_dtype=torch.float16,
            ).to(self.device)

    def _truncate_prompt(self, prompt: str) -> str:
        tokens = self._clip_tokenizer_truncate(
            prompt, truncation=True, max_length=77, return_tensors="pt"
        )
        return self._clip_tokenizer_truncate.decode(
            tokens["input_ids"][0], skip_special_tokens=True
        )

    def _split_prompt(self, prompt: str, max_tokens: int = 77) -> list[str]:
        tokens = self._clip_tokenizer_split.encode(prompt, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return [prompt]
        words = prompt.split()
        chunk_1 = []
        chunk_2 = []
        current_tokens = 0
        for word in words:
            word_tokens = self._clip_tokenizer_split.encode(
                word, add_special_tokens=False
            )
            if current_tokens + len(word_tokens) > max_tokens:
                chunk_2.append(word)
            else:
                chunk_1.append(word)
                current_tokens += len(word_tokens)
        return [" ".join(chunk_1), " ".join(chunk_2)]

    def _preprocess_image(self, image_path: Path) -> Image.Image:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((512, 512))
        return image

    def generate_image_from_image(
        self,
        prompt: str,
        img_path: Path,
        strength: float = 0.75,
        negative_prompt: str = "",
    ) -> Path:
        if not img_path.exists():
            raise FileNotFoundError("Image not found")
        if not self.load_img2img:
            raise RuntimeError(
                "Image-to-image generation models are not loaded. Please verify the configuration file."
            )
        image_path: Path = self.generated_directory / "edited_generated_image.png"
        init_image = self._preprocess_image(img_path)
        prompt = self._truncate_prompt(prompt)

        edited_image = self.base_model_pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            negative_prompt=negative_prompt,
            num_inference_steps=self.config.sd_config.num_inference_steps,
            guidance_scale=self.config.sd_config.guidance_scale,
        ).images[0]

        edited_image.save(image_path)
        self.config.to_yaml(self.generated_directory / "config.yml")
        return image_path

    def generate_image(self, prompt: str, negative_prompt: str = "") -> Path:
        if self.load_img2img:
            raise RuntimeError(
                "Image-to-image generation models are loaded. Please verify the configuration file."
            )
        filename = "generated_image"
        prompt = self._truncate_prompt(prompt)
        ctx = torch.autocast("cuda") if self.device.type == "cuda" else nullcontext()

        with ctx:
            base_image = self.base_model_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=self.config.sd_config.num_inference_steps,
                guidance_scale=self.config.sd_config.guidance_scale,
            ).images[0]

        image_path: Path = self.generated_directory / f"base_{filename}.png"

        if self.verbose:
            base_image.save(image_path)

        if self.load_refiner:
            with ctx:
                refined_image = self.refiner_model_pipe(
                    prompt=prompt,
                    image=base_image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=self.config.sd_config.num_inference_steps,
                    guidance_scale=self.config.sd_config.guidance_scale,
                ).images[0]

            image_path = self.generated_directory / f"refined_{filename}.png"
            refined_image.save(image_path)
        else:
            base_image.save(image_path)

        self.config.to_yaml(self.generated_directory / "config.yml")
        return image_path

    def generate_video(
        self,
        prompts: list[str],
        fps: int = 8,
        num_inference_steps: int = 20,
        negative_prompt: str = "",
    ) -> Path:
        """Generate a short animated WebP from a sequence of text prompts.

        Each prompt becomes one frame, so keep the list short (4-12 entries)
        for reasonable generation time on MPS.  Individual frames are also
        saved as PNGs in the output directory for full-quality inspection.
        """
        if self.load_img2img:
            raise RuntimeError(
                "generate_video requires a text-to-image model. "
                "PIX_2_PIX and SD_1_5_IMG_2_IMG are not supported."
            )
        if not prompts:
            raise ValueError("prompts must not be empty")

        ctx = torch.autocast("cuda") if self.device.type == "cuda" else nullcontext()
        frames: list[Image.Image] = []

        for i, prompt in enumerate(prompts):
            logger.info(f"Generating frame {i + 1}/{len(prompts)}")
            truncated = self._truncate_prompt(prompt)
            with ctx:
                image = self.base_model_pipe(
                    prompt=truncated,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=self.config.sd_config.guidance_scale,
                ).images[0]
            frame_path = self.generated_directory / f"frame_{i:03d}.png"
            image.save(frame_path)
            frames.append(image)

        video_path = self.generated_directory / "generated_video.webp"
        frames[0].save(
            video_path,
            format="WEBP",
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,
            loop=0,
            quality=85,
        )
        logger.info(f"Video saved to {video_path}")
        self.config.to_yaml(self.generated_directory / "config.yml")
        return video_path
