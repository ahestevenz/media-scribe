# -*- coding: utf-8 -*-
"""Command line script to generate a coherent video from a single text prompt via SVD."""
from __future__ import annotations

import argparse
import cProfile as profile
import sys
from pathlib import Path

from loguru import logger
from transformers import logging

import media_scribe.utils as utils
from media_scribe.image_video_scribe import ImageVideoScribe
from media_scribe.llama_text_scribe import LlamaTextScribe
from media_scribe.media_scribe_config import MediaScribeConfig

logging.set_verbosity_error()

__author__ = ["Ariel Hernandez <ahestevenz@bleiben.ar>"]
__copyright__ = "Copyright (c) 2024 Ariel Hernandez"
__license__ = "MIT"


def _main(args):
    """Actual program (without command line parsing). This is so we can profile.
    Parameters
    ----------
    args: namespace object as returned by ArgumentParser.parse_args()
    """
    if not Path(args["conf"]).exists():
        logger.error(
            f'{args["conf"]} does not exist. Please check config.yaml path and try again',
        )
        return -1

    media_config = MediaScribeConfig.from_yaml(Path(args["conf"]))

    if not args["no_llama"]:
        llama_model = LlamaTextScribe(media_config)
        prompt = utils.start_text_interaction(llama_model, generate_image=True)
    else:
        logger.info("Introduce your prompt to generate the video:")
        print("Prompt: ")
        prompt = input(" ")

    logger.info(
        "Enter a negative prompt for video generation (leave blank if none):")
    print("Negative Prompt: ")
    negative_prompt = input(" ")

    video_model = ImageVideoScribe(config=media_config)
    try:
        video_model.generate_video_svd(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=args["num_frames"],
            fps=args["fps"],
            motion_bucket_id=args["motion_bucket_id"],
            noise_aug_strength=args["noise_aug_strength"],
            num_inference_steps=args["num_inference_steps"],
            decode_chunk_size=args["decode_chunk_size"],
        )
    except Exception as ex:
        logger.error(
            f"An error occurred: {ex.__class__.__name__} - {ex}", exc_info=True)
    return 0


def main():
    """CLI for video generation"""

    argparser = argparse.ArgumentParser(
        description="Welcome to Media Scribe for video generation",
    )

    argparser.add_argument(
        "-c",
        "--conf",
        help="YAML configuration file",
        required=True,
    )

    argparser.add_argument(
        "-n",
        "--num-frames",
        help="Number of video frames to generate (14 or 25 for SVD)",
        type=int,
        default=25,
        required=False,
    )

    argparser.add_argument(
        "--fps",
        help="Frames per second for the output video",
        type=int,
        default=7,
        required=False,
    )

    argparser.add_argument(
        "--motion-bucket-id",
        help="Controls motion amount (0–255); higher = more motion",
        type=int,
        default=127,
        required=False,
    )

    argparser.add_argument(
        "--noise-aug-strength",
        help="Noise augmentation on the anchor frame (0.0–1.0); higher = more variation",
        type=float,
        default=0.02,
        required=False,
    )

    argparser.add_argument(
        "--num-inference-steps",
        help="Number of denoising steps",
        type=int,
        default=25,
        required=False,
    )

    argparser.add_argument(
        "--decode-chunk-size",
        help="Frames decoded at once by the VAE; lower values reduce peak memory (default: 4)",
        type=int,
        default=4,
        required=False,
    )

    argparser.add_argument(
        "-nl",
        "--no-llama",
        help="Disable loading the Llama model to improve the video prompt",
        action="store_true",
        default=False,
        required=False,
    )

    # Default Args
    argparser.add_argument(
        "-v",
        "--verbose",
        help="Increase logging output  (default: INFO)"
        "(can be specified several times)",
        action="count",
        default=0,
    )
    argparser.add_argument(
        "-p",
        "--profile",
        help="Run with profiling and store output in given file",
        metavar="output.prof",
    )
    args = vars(argparser.parse_args())

    _V_LEVELS = ["INFO", "DEBUG"]
    loglevel = min(len(_V_LEVELS) - 1, args["verbose"])
    logger.remove()
    logger.add(sys.stdout, level=_V_LEVELS[loglevel])

    utils.setup_ctrl_q_handler()

    try:
        if args["profile"] is not None:
            logger.info("Start profiling")
            r = 1
            profile.runctx(
                "r = _main(args)",
                globals(),
                locals(),
                filename=args["profile"],
            )
            logger.info("Done profiling")
        else:
            logger.info("Running without profiling")
            r = _main(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")
        return 0

    return r


if __name__ == "__main__":
    exit(main())
