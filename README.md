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


## HuggingFace Models

### Pretrained Models
#### 100M, 1B and 3B on 100B tokens
##### 100M 
- `collections/klein9692/hf_decoder-100m`
- branch: `checkpoint-20000`

| Repo Name                                  | Abbr. in paper     |
|--------------------------------------------|--------------------|
| klein9692/hf_qwen3_100m-model_ori          | Standard [100M]    |
| klein9692/hf_qwen3_100m-model_add4L        | Standard+4L [100M] |
| klein9692/hf_qwen3_100m-model_10624_3_0_64_twe | MHF(H3B10K) [100M] |
| klein9692/hf_qwen3_100m-model_16384_4_0_64_twe | MHF(H4B16K) [100M] |

##### 1B
- `collections/klein9692/hf_decoder-1b`
- branch: `checkpoint-200000`


| Repo Name                                      | Abbr. in paper     |
|------------------------------------------------|--------------------|
| klein9692/hf_qwen3_1b-model_ori_twe     | Standard [1B]      |
| klein9692/raw-1b_add2L     | Standard+2L [1B]   |
| klein9692/hash-1b-h3b10624     | MHF(H3B10K) [1B]   |
| klein9692/hf_qwen3_1b-model_16384_4_0_64_twe     | MHF(H4B16K) [1B]   |

##### 3B
- `collections/klein9692/hf_decoder-3b`
- branch: `checkpoint-200000`

| Repo Name                                       | Abbr. in paper   |
|-------------------------------------------------|------------------|
| klein9692/raw-3b                                | Standard [3B]    |
| klein9692/hf_qwen3_3b-model_16384_4_0_64_twe_32 | MHF(H4B16K) [3B] |



#### 1B Ablations (on 20B tokens)
- `collections/klein9692/1b-abla`
- branch: `checkpoint-40000`


##### Varying Hash Functions and Bucket Size
| Repo Name                                    | Abbr. in paper |
|----------------------------------------------|----------------|
| klein9692/mhf_1b_4096_3_0_64                 | H3B4K          |
| klein9692/mhf_1b_4096_4_64                   | H4B4K          |
| klein9692/mhf_1b_8192_3_64                   | H3B8K          |
| klein9692/mhf_1b_8192_4_64                   | H4B8K          |
| klein9692/mhf_1b_16384_3_64                  | H3B16K         |
| klein9692/hf_qwen3_1b-model_16384_4_0_64_twe | H4B16K         |
| klein9692/mhf_1b_32768_4_64                  | H4B32K         |

##### Single Hash
| Repo Name                     | Abbr. in paper |
|-------------------------------|----------------|
| klein9692/shf_1b_4096_4_0_64  | H1B4K          |
| klein9692/shf_1b_8192_4_0_64  | H1B8K          |
| klein9692/shf_1b_16384_4_0_64 | H1B16K         |


##### LSH
| Repo Name                                    | LSH:MMH3 |
|----------------------------------------------|----------|
| klein9692/hf_qwen3_1b-model_16384_4_0_64_twe | 0:4      |
| klein9692/mhf_1b_16384_4_1_64                | 1:3      |
| klein9692/mhf_1b_16384_4_2_64                | 2:2      |
| klein9692/mhf_1b_16384_4_3_64                | 3:1      |
| klein9692/mhf_1b_16384_4_4_64                | 4:0      |


### Vocabulary Expanded Models
- `collections/klein9692/hf_ve`
- branch: `checkpoint-16000`

| Repo Name                       | Abbr. in paper     |
|---------------------------------|--------------------|
| klein9692/mistral_cve_1b_ori    | Standard      |
| klein9692/mistral_cve_1b_add2L  | Standard+2L   |
| klein9692/mistral_cve_1b_H3B10K | MHF(H3B10K)   |
| klein9692/mistral_cve_1b_H4B16K | MHF(H4B16K)   |

