import copy
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers.models import BPE
from transformers import AutoTokenizer


def main(args):
    # load the source tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.hub_cache_dir,
    )
    vocab = tokenizer.get_vocab()
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    merges = tokenizer_json["model"]["merges"]
    if tokenizer_json["model"].get("byte_fallback") is not None:
        byte_fallback = tokenizer_json["model"]["byte_fallback"]
    else:
        byte_fallback = False
    if tokenizer_json["model"].get("fuse_unk") is not None:
        fuse_unk = tokenizer_json["model"]["fuse_unk"]
    else:
        fuse_unk = False

    # generate the new tokenizer
    for lang_idx, lang in enumerate(["cmn_Hani", "arb_Arab", "hin_Deva"]):
        curr_dataset = load_dataset(
            "text",
            data_files={"train": f"{args.corpus_path}_{lang}.txt"},
            cache_dir=args.datasets_cache_dir,
            split="train"
        )
        aux_tokenizer = tokenizer.train_new_from_iterator(
            curr_dataset["text"], args.vocab_size + ((lang_idx + 1) * args.num_new_tokens + 500),
        )
        aux_tokenizer_json = json.loads(aux_tokenizer._tokenizer.to_str())
        aux_merges = aux_tokenizer_json["model"]["merges"]
        if isinstance(aux_merges[0], str):
            aux_merges = [tuple(m.split(" ")) for m in aux_merges]
        else:
            pass

        # merge the tokenizers
        num_new_token = 0
        max_new_token = args.num_new_tokens
        ret_vocab = copy.copy(vocab)
        ret_merges = []
        old_merges = copy.copy(merges)
        for merge in aux_merges:
            # vocab
            token_1, token_2 = merge
            token = token_1 + token_2
            if num_new_token < max_new_token:
                if token_1 not in ret_vocab and token_2 not in ret_vocab:  # both are new
                    ret_vocab[token_1] = len(vocab) + num_new_token
                    ret_vocab[token_2] = len(vocab) + num_new_token + 1
                    num_new_token += 2
                elif token_1 not in ret_vocab and token_2 in ret_vocab:  # new + old
                    ret_vocab[token_1] = len(vocab) + num_new_token
                    num_new_token += 1
                elif token_1 in ret_vocab and token_2 not in ret_vocab:  # old + new
                    ret_vocab[token_2] = len(vocab) + num_new_token
                    num_new_token += 1
                else:  # both are old
                    pass
                if token not in ret_vocab:
                    ret_vocab[token] = len(vocab) + num_new_token
                    num_new_token += 1

            # merge
            if merge in merges:
                old_merges.remove(merge)
                ret_merges.append(merge)
            elif token in ret_vocab and token_1 in ret_vocab and token_2 in ret_vocab:
                ret_merges.append(merge)

        # setup a new BPE instance
        merges = ret_merges + old_merges
        vocab = ret_vocab
        if isinstance(merges[0], str):
            merges = [(merge.split(' ')[0], merge.split(' ')[1]) for merge in merges]
        else:
            pass

    # to avoid holes in the vocab
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    new_vocab = {token: i for i, (token, _) in enumerate(sorted_vocab)}

    tokenizer.backend_tokenizer.model = BPE(
        vocab=new_vocab,
        merges=[tuple(m) for m in merges],
        fuse_unk=fuse_unk,
        byte_fallback=byte_fallback,
    )

    # save
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name or path of the source tokenizer to use",
        required=True
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        help="Path to the corpus to train the aux tokenizer on",
        required=True
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="Vocabulary size of the aux tokenizer",
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory",
        required=True
    )
    parser.add_argument(
        "--num_new_tokens",
        type=int,
        help="Number of new tokens to add to the source tokenizer",
        default=100
    )
    parser.add_argument(
        "--datasets_cache_dir",
        type=str,
        help="Path to the datasets cache directory",
    )
    parser.add_argument(
        "--hub_cache_dir",
        type=str,
        help="Path to the Hugging Face hub cache directory",
    )
    args = parser.parse_args()
    main(args)

