from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from transformers import AutoTokenizer
from pathlib import Path

from util import save_to_disk


URL_PREFIX_1 = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data"
URL_PREFIX_2 = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"
LANG_FILES = {
    "eng": [
        f"{URL_PREFIX_2}/00{str(i)}_00000.parquet" for i in range(6)
    ],
    "cmn_Hani": [
        f"{URL_PREFIX_1}/cmn_Hani/train/000_0000{str(i)}.parquet" for i in range(4)
    ],
    "arb_Arab": [
        f"{URL_PREFIX_1}/arb_Arab/train/000_0000{str(i)}.parquet" for i in range(4)
    ],
    "hin_Deva": [
        f"{URL_PREFIX_1}/hin_Deva/train/00{i}_00000.parquet" for i in range(4)
    ],
}


def group_texts(examples: dict, block_size=128):
    # Concatenate all texts.
    try:
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    except Exception:
        print(examples)
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main(args):
    # Load the dataset
    print("Load dataset...")
    # Load the dataset
    dataset_list = []
    for lang in LANG_FILES.keys():
        n_files = len(LANG_FILES[lang]) // 2
        url_list = LANG_FILES[lang][int(args.shard_index) * n_files: (int(args.shard_index)+1) * n_files]
        curr_dataset = load_dataset("parquet", data_files=url_list, num_proc=args.num_workers)
        curr_dataset = curr_dataset["train"]

        # Remove the examples with empty text
        print("Removing examples with empty text...")
        curr_dataset = curr_dataset.filter(lambda example: len(example["text"]) > 0, num_proc=args.num_workers)
        # Strip the text
        print("Stripping the text...")
        curr_dataset = curr_dataset.map(
            lambda example: {"text": example["text"].strip()},
            num_proc=args.num_workers
        )

        # Load the tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            cache_dir=args.cache_dir
        )
        if "mistral" in args.tokenizer_name_or_path.lower():
            # Mistral-7B-v0.3 at least started with bos token to split articles
            pass
        elif "qwen3" in args.tokenizer_name_or_path.lower():
            # Qwen3 tokenizer need to append eos token to the end manually
            tokenizer.add_eos_token = True
        elif "gemma-4" in args.tokenizer_name_or_path.lower():
            # gemma-4 at least started with bos token to split articles
            pass
        else:
            NotImplementedError("Currently only take `mistral`, `qwen3`, `gemma-4` into consideration.")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize the dataset
        print("Tokenizing the dataset...")
        curr_dataset = curr_dataset.map(
            lambda examples: tokenizer(examples["text"], add_special_tokens=True),
            batched=True,
            num_proc=args.num_workers,
            remove_columns=curr_dataset.column_names,
        )

        curr_dataset = curr_dataset.shuffle(seed=42)

        # Group the texts
        print("Grouping the texts...")
        curr_dataset = curr_dataset.map(
            lambda examples: group_texts(examples, args.max_length),
            batched=True,
            num_proc=args.num_workers,
        )

        # Filter out the examples with length not equal to max_length
        print("Filtering out examples with length not equal to max_length...")
        curr_dataset = curr_dataset.filter(lambda example: len(example["input_ids"]) == args.max_length,
                                           num_proc=args.num_workers)
        curr_dataset = curr_dataset.select(range(512000))

        dataset_list.append(curr_dataset)

    dataset = concatenate_datasets(dataset_list)
    dataset = dataset.shuffle(seed=3407)
    # Save the tokenized dataset to a file
    print("Saving tokenized dataset...")
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir()
    save_to_disk(dataset, Path(args.output_dir) / str(args.shard_index) / "train", num_proc=args.num_workers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output data directory"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to the cache directory"
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        required=True,
        help="Name or path of the tokenizer to use"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes to use"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length of the tokenized sequences"
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        help="Index of the shard to use",
    )
    args = parser.parse_args()
    main(args)

