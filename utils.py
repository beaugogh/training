import json
from datasets import Dataset, Features, Value


def cast_datast_to_instruction_format(
    dataset: Dataset, orig_prompt_feat_name: str, orig_completion_feat_name: str
) -> Dataset:
    new_features = Features({"prompt": Value("string"), "completion": Value("string")})

    # Apply renaming first
    dataset = dataset.rename_column(orig_prompt_feat_name, "prompt")
    dataset = dataset.rename_column(orig_completion_feat_name, "completion")

    # Cast to new feature types
    dataset = dataset.cast(new_features)
    return dataset


def save_jsonl(data, file_path):
    """
    Save a list of dictionaries to a JSONL file.

    :param data: A list of dictionary objects.
    :param file_path: The path to the output JSONL file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")
        print(f"{len(data)} items are saved to {file_path}")
