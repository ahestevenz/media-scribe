# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import torch
from bnMediaScribe import MediaScribeConfig
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTokenizer

# from loguru import logger


class ImageVideoScribe:
    def __init__(self, config: MediaScribeConfig.MediaScribeConfig):
        self.config = config
        self.verbose = config.verbose
        self.generated_directory = self._generate_directory()
        self.device = torch.device(config.device)
        self._load_model_pipelines()

    def _generate_directory(
        self,
    ) -> Path:
        current_datetime = datetime.now()
        directory = (
            self.config.sd_config.root_output_dir
            / f"{current_datetime.strftime('%Y%m%d_%H%M%S')}_{self.config.sd_config.model_type}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _load_model_pipelines(
        self,
    ):
        match self.config.sd_config.model_type:
            case (
                MediaScribeConfig.ModelImageType.CIVITAI
                | MediaScribeConfig.ModelImageType.CIVITAI_TEST
                | MediaScribeConfig.ModelImageType.SD_XL
            ):
                self.base_model_pipe = (
                    StableDiffusionXLPipeline.from_single_file(
                        self.config.sd_config.base_model_path,
                        torch_dtype=torch.float16,
                    ).to(self.device)
                )
                pass
            case MediaScribeConfig.ModelImageType.SD_3:
                self.base_model_pipe = (
                    StableDiffusionXLPipeline.from_pretrained(
                        self.config.sd_config.base_model_path,
                        torch_dtype=torch.float16,
                    ).to(self.device)
                )
                pass
            case _:
                raise NotImplementedError("Method does not exist!")

        if self.config.sd_config.load_refiner:
            self.refiner_model_pipe = (
                StableDiffusionXLImg2ImgPipeline.from_single_file(
                    self.config.sd_config.refiner_model_path,
                    torch_dtype=torch.float16,
                ).to(self.device)
            )

    def _truncate_prompt(self, prompt: str) -> str:
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        tokens = tokenizer(
            prompt, truncation=True, max_length=77, return_tensors="pt"
        )
        return tokenizer.decode(
            tokens["input_ids"][0], skip_special_tokens=True
        )

    def _split_prompt(self, prompt: str, max_tokens: int = 77) -> List[str]:
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return [prompt]
        words = prompt.split()
        chunk_1 = []
        chunk_2 = []
        current_tokens = 0
        for word in words:
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            if current_tokens + len(word_tokens) > max_tokens:
                chunk_2.append(word)
            else:
                chunk_1.append(word)
                current_tokens += len(word_tokens)
        return [" ".join(chunk_1), " ".join(chunk_2)]

    def generate_image(self, prompt: str, negative_prompt: str = "") -> Path:
        filename = "generated_image"
        prompt = self._truncate_prompt(prompt)
        if self.device.type == "cuda":
            with torch.autocast("cuda"):
                base_image = self.base_model_pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=self.config.sd_config.num_inference_steps,
                    guidance_scale=self.config.sd_config.guidance_scale,
                ).images[0]
        else:
            base_image = self.base_model_pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=self.config.sd_config.num_inference_steps,
                guidance_scale=self.config.sd_config.guidance_scale,
            ).images[0]

        image_path: Path = self.generated_directory / f"base_{filename}.png"

        if self.verbose:
            base_image.save(image_path)

        if self.config.sd_config.load_refiner:
            if self.device.type == "cuda":
                with torch.autocast("cuda"):
                    refined_image = self.refiner_model_pipe(
                        prompt,
                        image=base_image,
                        negative_prompt=negative_prompt,
                        num_inference_steps=self.config.sd_config.num_inference_steps,
                        guidance_scale=self.config.sd_config.guidance_scale,
                    ).images[0]
            else:
                refined_image = self.refiner_model_pipe(
                    prompt,
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

    # TODO: generate video once image generation workflow is done
    # def generate_video(self, prompts: List[str]):

    #     frames = []
    #     fps = 23 # TODO: Get it from config
    #     frame_count = len(prompts)
    #     # Generate frames by modifying the prompt slightly each time
    #     for i in range(frame_count):
    #         # Modify the prompt to add variation, e.g., changing lighting or scene elements
    #         prompt = prompts[i]

    #         # Generate an image
    #         if self.device.type == "cuda":
    #             with torch.autocast("cuda"):
    #                 image = self.base_model_pipe(prompt=prompt).images[0]
    #         else:
    #             image = self.base_model_pipe(prompt=prompt).images[0]

    #         # Save the frame to the directory and add to frames list
    #         frame_path = os.path.join(self.root_ouput_path, f"frame_{i:03d}.png")
    #         image.save(frame_path)
    #         frames.append(np.array(image))

    #         print(f"Generated frame {i}/{frame_count}")

    #     # Save frames as a video using OpenCV
    #     video_path = os.path.join(self.root_ouput_path,video_filename)
    #     frame_height, frame_width, _ = frames[0].shape
    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #     video = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    #     # Write frames to the video
    #     for frame in frames:
    #         video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    #     # Release the video writer
    #     video.release()

    #     print(f"Video saved at {video_path}")
