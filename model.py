from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM
import torch
from time import time


class HuggingfaceModel:
    def __init__(self, model_name: str):
        print(f"Start loading model {model_name}...")
        start_time = time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        # self.model.eval()
        diff_time = time() - start_time
        print(f"Model {model_name} is loaded successfully. Device: {self.model.device}")
        print(f"Time: {int(diff_time)}s")

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        appy_chat_template: bool = True,
        enable_thinking: bool = True,
    ):
        with torch.no_grad():
            start_time = time()
            input_text = prompt
            if appy_chat_template:
                messages = (
                    [{"role": "system", "content": system_prompt}] if system_prompt else []
                )
                messages.append({"role": "user", "content": prompt})
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,  # Switches between thinking and non-thinking modes. Default is True.
                )
            model_inputs = self.tokenizer([input_text], return_tensors="pt").to(
                self.model.device
            )
            model_outputs = self.model.generate(
                **model_inputs, max_new_tokens=max_new_tokens, temperature=temperature
            )
            output_ids = model_outputs[0][len(model_inputs.input_ids[0]) :].tolist()
            response = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip("\n")
            print(f"Time: {int(time() - start_time)}s")
            return response
