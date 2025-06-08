from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("Hello, world!")
    model_name = "models/Qwen3-4B"
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )


if __name__ == "__main__":
    main()
