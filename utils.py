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
