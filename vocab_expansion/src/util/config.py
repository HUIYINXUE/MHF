import argparse
from transformers import HfArgumentParser, TrainingArguments


class CustomNTPArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Tune a language model."
        )
        self.hf_parser = HfArgumentParser(TrainingArguments)

        # Define any custom arguments using argparse
        self.parser.add_argument(
            "--train_dataset_path",
            type=str,
            required=True,
            help="Path to the tokenized training dataset."
        )
        self.parser.add_argument(
            "--dev_dataset_path",
            type=str,
            default=None,
            help="Path to the tokenized validation dataset."
        )
        self.parser.add_argument(
            "--tokenizer_name_or_path",
            type=str,
            required=True,
            help="Path to the tokenizer."
        )
        self.parser.add_argument(
            "--model_name_or_path",
            type=str,
            required=True,
            help="Path to the model."
        )
        self.parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Path to the cache directory."
        )
        self.parser.add_argument(
            "--use_streaming",
            action="store_true",
            help="Use streaming data loader."
        )
        self.parser.add_argument(
            "--model_size",
            type=str,
            choices=["100m", "250m", "500m", "1b"],
            default="100m",
            help="Size of the model."
        )
        self.parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="Resume from checkpoint."
        )
        self.parser.add_argument(
            "--stop_on_checkpoint",
            action="store_true",
            help="Stop training after saving a checkpoint."
        )
        self.parser.add_argument(
            "--stop_after_n_checkpoints",
            type=int,
            default=4,
            help="Stop training after saving n checkpoints."
        )
        self.parser.add_argument(
            "--num_resumptions",
            type=int,
            default=0,
            help="Number of resumptions."
        )
        self.parser.add_argument(
            "--resumption_checkpoint_interval",
            type=int,
            default=20000,
            help="Interval for saving checkpoints."
        )
        self.parser.add_argument(
            "--num_failures",
            type=int,
            nargs="+",
            default=None,
            help="Number of failures for each checkpoint interval."
        )
        self.parser.add_argument(
            "--save_checkpoint_at_step",
            type=int,
            default=None,
            help="Save checkpoint at this step."
        )

        # TODO: add addtional params for HashFormers
        self.parser.add_argument(
            "--enable_hashformer",
            action="store_true",
            help="Enable HashFormers"
        )

        self.parser.add_argument(
            "--use_conditional_prob",
            action="store_true",
            help="Using conditional prob to convert the original vocabulary space"
        )

        self.parser.add_argument(
            "--lsh_only_embed",
            action="store_true",
            help="Only take embeddings from LSH index to reconstruct embeddings"
        )

        self.parser.add_argument(
            "--num_buckets",
            type=int,
            default=4196,
            help="Number of buckets for each hash function"
        )

        self.parser.add_argument(
            "--num_hashes",
            type=int,
            default=4,
            help="Number of total hash functions (LSH+RndHash)"
        )

        self.parser.add_argument(
            "--num_lsh",
            type=int,
            default=3,
            help="Number of LSH hash functions"
        )

        self.parser.add_argument(
            "--bottle_neck_dim",
            type=int,
            default=64,
            help="Bottleneck dim for Hash embeddings and lm head"
        )

        self.parser.add_argument(
            "--fused_decoding",
            action="store_true",
            help="Whether to use OOV hash indices for inference"
        )

        self.parser.add_argument(
            "--no_tie",
            action="store_false",
            help="Whether to forbid tie_word_embeddings"
        )

        # TODO: path to export external_dict and lsh_proj_mat
        self.parser.add_argument(
            "--external_file_path",
            type=str,
            default=None,
            help="path to save external_dict and lsh_proj_mat"
        )

    def parse_args(self):
        args, extras = self.parser.parse_known_args()
        training_args = self.hf_parser.parse_args_into_dataclasses(extras)[0]
        return args, training_args