from bnMediaScribe import MediaScribeConfig as config
import torch
import torch
from transformers import AutoTokenizer, LlamaForCausalLM


class LlamaTextScribe:
    def __init__(self, config: config.MediaScribeConfig):
        self.verbose = config.verbose
        self.config = config.llama_config
        self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = LlamaForCausalLM.from_pretrained(self.config.model_name, torch_dtype=torch.float16)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.to(self.device)
        self.messages = [
            {"role": "system", "content": self.config.system_prompt}
        ]
        
    def add_user_message(self, user_input: str):
        """Add user input to the message history."""
        self.messages.append({"role": "user", "content": user_input})
    
    def add_assistant_message(self, assistant_output: str):
        """Add assistant response to the message history."""
        self.messages.append({"role": "assistant", "content": assistant_output})
    
    def build_prompt(self):
        """Builds a complete prompt for LLaMA by concatenating all messages."""
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.messages])
    
    def summarize_chat(self):
        """Build a prompt that instructs the model to summarize the conversation"""
        conversation_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.messages])
        summary_prompt = f"Please provide a concise summary of the following conversation in two sentences or less:\n{conversation_text}\nSummary:"

        inputs = self.tokenizer(summary_prompt, return_tensors="pt").to(self.device)
        summary_output = self.model.generate(**inputs, max_length=150)
        summary = self.tokenizer.decode(summary_output[0], skip_special_tokens=True)        
        return summary.split("Summary:")[-1].strip()
    
    def truncate_if_needed(self, full_prompt: str) -> str:
        """Truncate the conversation history to fit within the token limit, if necessary."""
        tokenized_prompt = self.tokenizer(full_prompt, return_tensors="pt")["input_ids"]
        if tokenized_prompt.shape[1] > self.config.max_input_tokens:
            truncation_point = tokenized_prompt.shape[1] - self.config.max_input_tokens
            truncated_prompt = self.tokenizer.decode(tokenized_prompt[0, truncation_point:], skip_special_tokens=True)
            return truncated_prompt
        return full_prompt
    
    def generate_text(self, user_message: str) -> str:
        """Generate text based on a prompt and accumulated conversation history."""
        if len(self.messages) > self.config.max_num_historical_messages:
            summary = self.summarize_chat()
            self.messages = [{"role": "system", "content": self.config.system_prompt},
                             {"role": "system", "content": f"Summary of previous conversation: {summary}"}]
        self.add_user_message(user_message)
        
        # Preparing the prompt
        prompt = self.build_prompt()
        prompt = self.truncate_if_needed(prompt)
        prompt += "\nAssistant:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_length=self.config.max_tokens, temperature=0.7, pad_token_id=self.tokenizer.pad_token_id)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
        self.add_assistant_message(generated_text)
        return generated_text


    
    