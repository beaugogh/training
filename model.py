from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM
import torch


class HuggingfaceModel:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        # self.model.eval()
        print(f"Model {model_name} loaded successfully. Device: {self.model.device}")

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        thinking: bool = False,
        max_new_tokens: int = 32768,
    ):
        with torch.no_grad():
            messages = (
                [{"role": "system", "content": system_prompt}] if system_prompt else []
            )
            messages.append({"role": "user", "content": prompt})
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,  # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = self.tokenizer([input_text], return_tensors="pt").to(
                self.model.device
            )
            model_outputs = self.model.generate(
                **model_inputs, max_new_tokens=max_new_tokens
            )
            output_ids = model_outputs[0][len(model_inputs.input_ids[0]) :].tolist()
            response = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip("\n")
            return response
