from datasets import load_dataset
import re


URL_PREFIX_1 = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data"


LANG_FILES = {
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


def main(args):
    for lang in ["cmn_Hani", "arb_Arab", "hin_Deva"]:
        # Load the dataset
        print("Loading dataset...")
        # Load dataset...
        ## need to load dataset from different
        url_list = LANG_FILES[lang]
        curr_dataset = load_dataset("parquet", data_files=url_list, num_proc=args.num_workers)
        curr_dataset = curr_dataset["train"]
        curr_dataset = curr_dataset.shuffle(seed=42).select(range(250000))

        # Write the dataset to a .txt file
        print("Writing dataset to a .txt file...")
        if lang == "cmn_Hani":
            split_ptn = r'[！|？|。|，|；|：]'
        elif lang == "arb_Arab":
            split_ptn = r'[؟،؛!.:]'
        elif lang == "hin_Deva":
            split_ptn = r'[।॥!?.,:;]'
        else:
            raise NotImplementedError(f"language should be in `cmn_Hani`, `arb_Arab` or `hin_Deva`.")

        file_name = f"{args.output_file}_{lang}.txt"
        with open(file_name, "w") as f:
            lines = []
            for example in curr_dataset:
                if lines == []:
                    print([line.strip() for line in re.split(split_ptn, example["text"].replace("\\n", "\n")) if
                           line.strip()])
                lines.extend([line.strip() for line in re.split(split_ptn, example["text"].replace("\\n", "\n")) if
                              line.strip()])
            f.writelines([line + "\n" for line in lines])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes to use"
    )
    args = parser.parse_args()
    main(args)


