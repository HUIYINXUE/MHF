from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path

from util import save_to_disk


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

    # Split by chunks of block_size.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main(args):
    # Load the dataset
    print("Loading dataset...")
    # Load the dataset
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
        split="train",
        num_proc=args.num_workers
    )  # 200B tokens
    dataset = dataset.shard(num_shards=40, index=args.shard_index)  # ~5.5B tokens

    # Split the dataset into train and test
    if args.shard_index == 0:
        dataset = dataset.train_test_split(test_size=0.1, shuffle=True)
        train_dataset = dataset["train"]
        dev_dataset = dataset["test"]
        del dataset
    else:
        train_dataset = dataset
        dev_dataset = None

    # Remove the examples with empty text
    print("Removing examples with empty text...")
    train_dataset = train_dataset.filter(lambda example: len(example["text"]) > 0, num_proc=args.num_workers)
    if dev_dataset is not None:
        dev_dataset = dev_dataset.filter(lambda example: len(example["text"]) > 0, num_proc=args.num_workers)

    # Strip the text
    print("Stripping the text...")
    train_dataset = train_dataset.map(
        lambda example: {"text": example["text"].strip()},
        num_proc=args.num_workers
    )
    if dev_dataset is not None:
        dev_dataset = dev_dataset.map(
            lambda example: {"text": example["text"].strip()},
            num_proc=args.num_workers
        )

    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    print("Tokenizing the dataset...")
    train_dataset = train_dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        num_proc=args.num_workers,
        remove_columns=train_dataset.column_names,
    )
    if dev_dataset is not None:
        dev_dataset = dev_dataset.map(
            lambda examples: tokenizer(examples["text"]),
            batched=True,
            num_proc=args.num_workers,
            remove_columns=dev_dataset.column_names,
        )

    # Shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=21)

    # Group the texts
    print("Grouping the texts...")
    train_dataset = train_dataset.map(
        lambda examples: group_texts(examples, args.max_length),
        batched=True,
        num_proc=args.num_workers,
    )
    if dev_dataset is not None:
        dev_dataset = dev_dataset.map(
            lambda examples: group_texts(examples, args.max_length),
            batched=True,
            num_proc=args.num_workers,
        )

    # Filter out the examples with length not equal to max_length
    print("Filtering out examples with length not equal to max_length...")
    train_dataset = train_dataset.filter(lambda example: len(example["input_ids"]) == args.max_length,
                                         num_proc=args.num_workers)
    if dev_dataset is not None:
        dev_dataset = dev_dataset.filter(lambda example: len(example["input_ids"]) == args.max_length,
                                         num_proc=args.num_workers)

    # Save the tokenized dataset to a file
    print("Saving tokenized dataset...")
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir()
    save_to_disk(train_dataset, Path(args.output_dir) / str(args.shard_index) / "train", num_proc=args.num_workers)
    if dev_dataset is not None:
        save_to_disk(dev_dataset, Path(args.output_dir) / str(args.shard_index) / "dev", num_proc=args.num_workers)


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