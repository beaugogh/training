import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from transformers import TrainingArguments


def main():
    print("Begin")
    dataset = load_dataset("stanfordnlp/imdb", split="train")

    training_args = SFTConfig(
        max_length=512,
        output_dir="opt-350m-imdb",
    )
    trainer = SFTTrainer(
        "facebook/opt-350m",
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train()
    print("End")


if __name__ == "__main__":
    main()
