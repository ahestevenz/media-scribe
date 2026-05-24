# -*- coding: utf-8 -*-
from __future__ import annotations

import signal
import sys
import termios

from loguru import logger

from media_scribe.llama_text_scribe import LlamaTextScribe


def setup_ctrl_q_handler() -> None:
    """Register Ctrl+Q as a clean exit key (no traceback)."""

    def _handle_quit(signum, frame):
        logger.info("Cancelled by user (Ctrl+Q)")
        sys.exit(0)

    signal.signal(signal.SIGQUIT, _handle_quit)
    try:
        attr = termios.tcgetattr(sys.stdin.fileno())
        attr[6][termios.VQUIT] = b"\x11"
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, attr)
    except (termios.error, OSError):
        pass


def start_text_interaction(
    llama_model: LlamaTextScribe,
    generate_image: bool = True,
) -> str:
    prompt = ""
    if generate_image:
        logger.info("Introduce your prompt to generate the image:")
        print("You: ")
        prompt = input(" ")

        response = input(
            "Do you want to improve the prompt? (yes/no): ").strip().lower()
    else:
        response = "yes"

    if response == "no":
        logger.info("Returning the original prompt...")
        return prompt

    elif response == "yes":
        logger.info("Welcome to the LLaMA 3 Interactive Text Generator!")
        logger.info(
            "Type 'exit' to quit the program and use the last LLaMA response as the prompt.\n",
        )

        last_llama_response = ""

        while True:
            print("You:")
            user_prompt = input(" ")

            if user_prompt.lower() in ["exit", "quit"]:
                print("Have a nice day :-) !")
                prompt = last_llama_response if last_llama_response else prompt
                break
            try:
                generated_text = llama_model.generate_text(user_prompt)
                print("LLaMA:")
                print(f" {generated_text}")
                last_llama_response = generated_text
            except Exception as e:
                logger.error(f"Error generating text: {e}")

        return prompt
    else:
        logger.warning("Invalid response. Returning the original prompt.")
        return prompt
