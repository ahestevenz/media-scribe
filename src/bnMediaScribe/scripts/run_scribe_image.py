# -*- coding: utf-8 -*-
"""Command line script to upload a whole directory tree."""
from __future__ import annotations

import argparse
import cProfile as profile
import sys
from pathlib import Path

from bnMediaScribe import (
    ImageVideoScribe,
    MediaScribeConfig,
    utils)
from loguru import logger
from transformers import logging

logging.set_verbosity_error()

__author__ = ["Ariel Hernandez <ahestevenz@bleiben.ar>"]
__copyright__ = "Copyright 2024 Bleiben. All rights reserved."
__license__ = """General Public License"""


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

    media_config = MediaScribeConfig.MediaScribeConfig.from_yaml(
        Path(args["conf"]),
    )
    if not args["no_llama"]:
        llama_model = LlamaTextScribe.LlamaTextScribe(media_config)
        prompt = utils.start_text_interation(llama_model, generate_image=True)
    else:
        logger.info("Introduce your prompt to generate the image:")
        print("Prompt: ")
        prompt = input(" ")

    logger.info(
        "Enter a negative prompt for image generation (leave blank if none):")
    print("Negative Prompt: ")
    negative_prompt = input(" ")

    image_model = ImageVideoScribe.ImageVideoScribe(media_config)
    try:
        if args["path_image"]:
            strength = float(args["strength"])
            image_path = Path(args["path_image"])
            image_model.generate_image_from_image(
                prompt, image_path, strength, negative_prompt
            )
        else:
            image_model.generate_image(prompt, negative_prompt)
    except Exception as ex:
        logger.error(
            f"An error occurred: {ex.__class__.__name__} - {ex}", exc_info=True)
    return 0


def main():
    """CLI for upload the encripted files"""

    # Module specific
    argparser = argparse.ArgumentParser(
        description="Welcome to Media Scribe for image generation",
    )

    argparser.add_argument(
        "-c",
        "--conf",
        help="YAML configuration file",
        required=True,
    )

    argparser.add_argument(
        "-i",
        "--path-image",
        help="Path to the reference image that will be used as a starting point to generate a new one.",
        default=None,
        required=False,
    )

    argparser.add_argument(
        "-s",
        "--strength",
        help="Strength",
        default=0.75,
        required=False,
    )

    argparser.add_argument(
        "-nl",
        "--no-llama",
        help="Disable loading the Llama model to improve image prompt generation",
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

    return r


if __name__ == "__main__":
    exit(main())
