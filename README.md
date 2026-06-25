# MultiHashFormer

## Core Dependencies

- transformers == 5.3.0
- mmh3 == 5.2.0
- flash_attn == 2.x
- tokenizers == 0.22.2
- lm_eval == 0.4.8


## Model Implementations
- folder: `hashformer/qwen3`
- standard: `hashformer/qwen3/qwen3_ori` (huggingface copy)
- MHF: `hashformer/qwen3/qwen3_hashformer` (ours)


## Pretraining English LM
### Preprocessing
- folder: `preprocessing`
- script: `preprocessing/generate_ve_data.sh`

### Training
- folder: `training`
- scripts:
  - standard: `training/ntp_standard.sh`
  - MHF: `training/ntp_mhf.sh`


## Vocabulary Expansion (continual pretraining)
- folder: `vocab_expansion`
### Tokenizer Training
- folder: `vocab_expansion/train_tokenizer`
- data preparation script: `vocab_expansion/train_tokenizer/gen_tokenizer_train_data.sh`
- tokenizer training script: `vocab_expansion/train_tokenizer/train_tokenizer.sh`

### Trained Tokenizer
- folder: `vocab_expansion/expanded_tokenizer`

### Continual Training
- folder: `vocab_expansion/src`
- scripts:
  - standard: `vocab_expansion/ve_mhf.sh`
  - MHF: `vocab_expansion/ve_standard.sh`