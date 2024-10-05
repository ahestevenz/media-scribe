from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import List
from typing import String
from typing import Tuple

from bnMediaScribe import LlamaTextScribe as llama
from loguru import logger as logging


def start_text_interation(
    llama_model: llama.LlamaTextScribe,
    generate_image: bool = True,
) -> String:
    if generate_image:
        logging.info("Introduce your prompt to generate the image:")
        print("You: ")
        prompt = input(" ")

        response = (
            input("Do you want to improve the prompt? (yes/no): ").strip().lower()
        )
    else:
        response = "yes"

    if response == "no":
        logging.info("Returning the original prompt...")
        return prompt

    elif response == "yes":
        logging.info("Welcome to the LLaMA 3 Interactive Text Generator!")
        logging.info(
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
                logging.error(f"Error generating text: {e}")

        return prompt
    else:
        logging.warning("Invalid response. Returning the original prompt.")
        return prompt
