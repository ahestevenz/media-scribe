from __future__ import annotations

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionXLPipeline
from loguru import logger
from MediaScribeConfig import MediaScribeConfig
from PIL import Image


class ImageVideoScribe:
    def __init__(self, config: MediaScribeConfig):
        self.verbose = config.verbose
        self.root_ouput_path = self.config.root_ouput_path
        self.config = config.sd_config
        self.device = torch.device(config.device)
        self.base_model_pipe = StableDiffusionXLPipeline.from_pretrained(
            self.config.base_model_path).to(self.device)
        self.refiner_model_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.config.refiner_model_path).to(self.device)

    ###############

    # import os, uuid
    # import torch
    # from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

    # root_model_path = "/Users/ahestevenz/.cache/huggingface/hub/models--Stable-Diffusion-XL"
    # image_filename = f"{uuid.uuid4()}.png"
    # directory = "/Users/ahestevenz/testing_llm/SDXL"
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # # Path to the CivitAI checkpoint model (downloaded from CivitAI)
    # civitai_checkpoint_path = "/Users/ahestevenz/Desktop/juggernautXL_juggXIByRundiffusion.safetensors"

    # # Load the CivitAI checkpoint as the base model pipeline
    # pipe = StableDiffusionXLPipeline.from_single_file(civitai_checkpoint_path, torch_dtype=torch.float16)

    # # Move the model to the appropriate device
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    # pipe.to(device)

    # # Generate an initial image from a prompt
    # prompt = "Create a 1280x720 image of a mystical red planet with glowing landscapes, floating islands, and swirling auroras. Unicorns with radiant horns gallop across crimson plains, while flying foxes with shimmering wings glide between crystal formations. Magical stardust sparkles as twin moons cast an ethereal glow."
    # if device.type == "cuda":
    #     with torch.autocast("cuda"):
    #         # Generate the base image using the pipeline (CUDA case)
    #         base_image = pipe(prompt).images[0]
    # else:
    #     # If not using CUDA, don't use autocast
    #     base_image = pipe(prompt).images[0]

    # # Save the initial image
    # base_image.save(os.path.join(directory, f"base_{image_filename}"))

    # # Optionally, refine the image using a refiner model
    # refiner_model_path = f"{root_model_path}/sd_xl_refiner_1.0.safetensors"

    # # Load the refiner model
    # refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(refiner_model_path, torch_dtype=torch.float16)

    # # Move the refiner to the appropriate device
    # refiner_pipe.to(device)

    # # Refine the initial image using the refiner model
    # if device.type == "cuda":
    #     with torch.autocast("cuda"):
    #         # Refine the base image using the pipeline (CUDA case)
    #         refined_image = refiner_pipe(prompt=prompt, image=base_image).images[0]
    # else:
    #     # If not using CUDA, don't use autocast
    #     refined_image = refiner_pipe(prompt=prompt, image=base_image).images[0]

    # # Save the refined image
    # refined_image.save(os.path.join(directory, f"refined_{image_filename}"))

    ##############
    def generate_image(self, prompt: str) -> Path:
        if self.device.type == 'cuda':
            with torch.autocast('cuda'):
                base_image = self.base_model_pipe(prompt).images[0]
        else:
            base_image = self.base_model_pipe(prompt).images[0]

        if self.verbose:
            logger.
            base_image.save(os.path.join(directory, f'base_{image_filename}'))

        # Refine the initial image using the refiner model
        if self.device.type == 'cuda':
            with torch.autocast('cuda'):
                # Refine the base image using the pipeline (CUDA case)
                refined_image = self.refiner_model_pipe(
                    prompt=prompt, image=base_image).images[0]
        else:
            # If not using CUDA, don't use autocast
            refined_image = self.refiner_model_pipe(
                prompt=prompt, image=base_image).images[0]

        return refined_image

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
