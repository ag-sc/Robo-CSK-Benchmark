import re

import torch
import transformers

from utils.prompter import Prompter


class GemmaPrompter(Prompter):
    def __init__(self, max_new_tokens=10, temp = None, sampling = False):
        super().__init__("gemma-2-27b-it")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(f'google/{self.model_name}')
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            f'google/{self.model_name}',
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.max_new_tokens = max_new_tokens

        # For Self Consistency
        self.temperature = temp 
        self.sampling = sampling

    def prompt_model(self, system_msg: str, user_msg: str, question: str):
        messages = [{"role": "user", "content": f"{system_msg}\n{user_msg}\n{question}"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.model.device),
                                      max_new_tokens=self.max_new_tokens,
                                      do_sample=self.sampling,  # No randomness if False
                                      temperature=self.temperature,
                                      top_p=None)
        outputs = self.tokenizer.decode(outputs[0, len(inputs):])
        match = re.search(r"<start_of_turn>model(.*?)<eos>", outputs, re.DOTALL)
        result = match.group(1).strip() if match else None
        return result
