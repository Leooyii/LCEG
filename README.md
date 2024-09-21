<h1 align="center">
<!-- <img src="./fig/.png" width="100" alt="" /> -->
<br>
A Controlled Study on Long Context Extension and Generalization in LLMs
</h1>



<p align="center">
  <a href="https://arxiv.org/pdf/2409.12181"><b>[📜 Paper]</b></a> •
  <a href="https://huggingface.co/Leooyii"><b>[🤗 HF HUB]</b></a> 
</p>


<p align="center">
Repo for "<a href="https://arxiv.org/pdf/2409.12181" target="_blank">A Controlled Study on Long Context Extension and Generalization in LLMs</a>"
</p>

<img src="./fig/needle.png" width="1000" alt="" />

## TABLE OF CONTENTS
1. [News](#news)
2. [Installation and Quick Guide](#installation-and-quick-guide)
3. [Long Context Methods Implementation](#long-context-methods-implementation)
   - [Training Data](#training-data)
   - [Models](#models)
   - [Continuous Training](#continuous-training)
4. [Evaluation](#evaluation)
   - [Perplexity Validation](#perplexity-validation)
   - [Needle in A Haystack](#needle-in-a-haystack)
   - [LongBench & Manyshots Trec](#longbench-manyshots-trec)
   - [Ruler](#ruler)
5. [Acknowledgement](#acknowledgement)
6. [Citation](#citation)
7. [License](#license)


## 🔥 News
- [2024/09/19] LCEG paper is available on arxiv.



## 🚀 Installation and Quick Guide

To install and run the evaluation:
1. Clone the repository on your local machine, using git clone and pasting the url of this project.
2. Run the following code:
```
conda create -n lceg python=3.10
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```
## Long Context Methods Inplementation

### Training Data
We follow [Long-Context-Data-Engineering](https://github.com/FranxYao/Long-Context-Data-Engineering) to create our training data.

| Data           | Tokens | Examples  | Length    | Download |
|:---------------|----------|----------|----------|----------|
| Slimpajama_downsample_32k_1B | 1B       | 30774       | 32k      | [Link](https://huggingface.co/datasets/Leooyii/Slimpajama_downsample_32k_1B) |
| Slimpajama_downsample_64k_1B | 1B       | 15386       | 64k      | [Link](https://huggingface.co/datasets/Leooyii/Slimpajama_downsample_32k_1B) |
| Slimpajama_downsample_64k_2B | 2B       | 30780       | 64k      | [Link](https://huggingface.co/datasets/Leooyii/Slimpajama_downsample_32k_1B) |

### Models
Models with continuous fine-tuning.

| Model                               | Size | Context | Training Tokens | Link                                                              |
|:------------------------------------|------|---------|-----------------|-------------------------------------------------------------------|
| Llama2-7b-hf-slimpajama1B-ntk-32k   | 7B   | 32768   | 1B          | [Model](https://huggingface.co/Leooyii/NTK_32k_Slimpajama_1B)  |
| Llama2-7b-hf-slimpajama1B-ntk-64k   | 7B   | 65536   | 1B          | [Model](https://huggingface.co/Leooyii/NTK_64k_Slimpajama_1B)  |
| Llama2-7b-hf-slimpajama2B-ntk-64k   | 7B   | 65536   | 2B          | [Model](https://huggingface.co/Leooyii/NTK_64k_Slimpajama_2B)  |
| Llama2-7b-hf-slimpajama1B-pi-32k    | 7B   | 32768   | 1B          | [Model](https://huggingface.co/Leooyii/PI_32k_Slimpajama_1B)  |
| Llama2-7b-hf-slimpajama1B-yarn-32k  | 7B   | 32768   | 1B          | [Model](https://huggingface.co/Leooyii/YaRN_32k_Slimpajama_1B)  |
| Llama2-7b-hf-slimpajama1B-longlora-32k | 7B   | 32768  | 1B          | [Model](https://huggingface.co/Leooyii/Longlora_32k_Slimpajama_1B) |
| Llama2-7b-hf-slimpajama1B-CLEX-32k | 7B   | 32768  | 1B          | [Model](TODO) |
| Llama2-7b-hf-slimpajama1B-landmark-512 | 7B   | -      | 1B          | [Model](https://huggingface.co/Leooyii/Landmark_512_Slimpajama_1B)  |


### Continuous Training
We provide our scripts for continuous fine-tuning on these long-context methods in `finetune.sh`.

To train the models, please enable DeepSpeed acceleration. continuous_finetuning/ds_configs/stage3_offload.json was the configuration file used for training.

**Setup `finetune.sh`**
```
cd continuous_finetuning
# set the methods and training config in finetune.sh
bash finetune.sh
```

In `finetune.sh`, we provide 3 scripts for continuous fine-tuning on 6 methods: `origin`, `pi`, `ntk`, `yarn`, `longlora`, and `landmark`. Here is an example:

```bash
torchrun  --nproc_per_node=8 fine-tune.py  \
        --model_name_or_path "meta-llama/Llama-2-7b-hf" \
        --bf16 True \
        --output_dir ckpts/llama2-7b-hf-slimpajama-pi-32k \
        --model_max_length 32768 \
        --use_flash_attn True \
        --low_rank_training False \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 32 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.0 \
        --warmup_steps 20 \
        --deepspeed ds_configs/stage3_offload.json \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 1     \
        --tf32 True \
        --report_to "wandb" \
        --use_wandb True \
        --dataset_dir Leooyii/Slimpajama_downsample_32k_1B \
        --method_name pi # option:[origin, pi, ntk, yarn]
```
- You can train different long-context methods by changing `--method_name`.
- You can change `--model_name_or_path`, `--output_dir` to your own directory. 
- Note that you can change `model_max_length` to other values.
- To train Longlora, please refer to the 'Scripts for Longlora' section in `finetune.sh` for training.
- To train Landmark Attention, please refer to the 'Scripts for Landmark Attention' section in `finetune.sh` for training.


## Evaluation
### Perplexity validation 
We provide our scripts for Perplexity validation on PG19 and Proof-pile in `eval_perplexity/scripts`. We use the tokenized test splits of PG19 and Proof-pile dataset processed by longlora. The raw data and tokenized data are in `eval_perplexity/data` folder.
```
cd eval_perplexity
python eval_pi.py \
        --seq_len 32768 \
        --batch_size 1 \
        --base_model path_to_checkpoints \
        --data_path data/pg19/test.bin \
        --output_dir results/pg19/pi_pg19.json 
```
- Please note that `--seq_len` is used to set the sequence length for evaluation. 
- Remember to change `--base_model`, `--output_dir` to your own directory. 

### Needle in A Haystack
**Setup `eval.sh`**
```
cd needle
bash eval.sh
```
- The evaluation on 64k context length requires 1 * 80G A100 and on 128k context requires 4 * 80G A100.
- Set the method name and sequence length in `eval.sh`.

### LongBench & ManyShots TREC
The data to evaluate LongBench and ManyShots TREC is available at [LongBench](https://huggingface.co/datasets/Leooyii/longbench) and [ManyShots TREC](https://huggingface.co/datasets/Leooyii/manyshots_trec).

We provide our scripts to evaluate LongBench and ManyShots TREC in `longbench/scripts/eval_llama2.sh`.

**Setup `eval_llama2.sh`**

To eval LongBench, set the datasets in `longbench/scripts/eval_llama2.sh`:
```
# longbench
datasets=("narrativeqa" "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "musique" \
          "gov_report" "qmsum" "multi_news" "trec" "triviaqa" "samsum" \
          "passage_count" "passage_retrieval_en" "lcc" "repobench-p")
```
To eval ManyShots TREC, , set the datasets in `longbench/scripts/eval_llama2.sh`:
```
datasets=("trec_1000shots" "trec_875shots" "trec_750shots" "trec_625shots" "trec_500shots" \
        "trec_400shots" "trec_300shots" "trec_200shots" "trec_100shots" "trec_75shots" \
        "trec_50shots" "trec_25shots" "trec_10shots" "trec_5shots" "trec_1shots")
```
After setting up the datasets and models, run `eval_llama2.sh`:
```
cd longbench
bash scripts/eval_llama2.sh
```
You can obtain the output of the model under the selected datasets under the `longbench/pred/` folder.

**Get the score using `score.sh`**

Run `longbench/scripts/score.sh` to evaluate all the long-context methods.
```
bash scripts/score.sh
```
### Ruler
**Requirements**
To evaluate RULER, please follow their guidance to create a new environment for evaluation. More details can be found at [RULER Requirements](https://github.com/hsiehjackson/RULER?tab=readme-ov-file#-requirements).

**Setup `run.sh`**

```
GPUS="" # number of GPUs
ROOT_DIR="" # the path that stores generated task samples and model predictions. 
MODEL_DIR="" # the path that contains individual model folders from Huggingface.
ENGINE_DIR="" # the path that contains individual engine folders from TensorRT-LLM.
```
- The evaluation on 32k context length requires 1 * 80G A100 and on 64k context requires 2 * 80G A100.

**Setup `config_models.sh`**

``` 
    case $MODEL_NAME in
        llama2-7b-hf-lminfinite)
            MODEL_PATH=YOUR_MODEL_FOLDER
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="hf"
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
            ;;
        llama-2-7b-hf-slimpajama-pi-32k)
            MODEL_PATH=YOUR_MODEL_FOLDER
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="vllm"
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
            ;;
        llama-2-7b-hf-slimpajama-ntk-32k)
            MODEL_PATH=YOUR_MODEL_FOLDER
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="hf"
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
            ;;
```
For NTK, LM-Infinite, and Landmark Attention methods, please set `MODEL_FRAMEWORK="hf"`.

**Start evaluation**
```
bash run.sh YOUR_MODEL_NAME synthetic
```

**Get the score using `eval.sh`**

```
eval_methods=("llama2-7b-hf" "llama2-7b-hf-lminfinite" "llama2-7b-hf-ntk-frozen" "llama-2-7b-hf-slimpajama-pi-32k" \
    "llama-2-7b-hf-slimpajama-ntk-32k" "llama2-7b-hf-slimpajama-ntk-64k" "llama2-7b-hf-slimpajama-ntk-64k-2B" \
    "llama2-7b-hf-slimpajama-yarn-32k" "llama2-7b-hf-slimpajama-longlora-32k" "llama2-7b-hf-slimpajama-landmark")
eval_length=(4096 8192 16384 32768 65536)

for method in "${eval_methods[@]}"; 
do
for length in "${eval_length[@]}"; 
do
python eval/evaluate.py \
  --data_dir /results/${method}/synthetic/${length}/pred \
  --benchmark synthetic
done
done
```

## Acknowledgements
We sincerely appreciate the assistance provided by the following people (works):
- We thank Yao Fu, Yue Yu, Tianyu Gao,  Celine Lee, Woojeong Kim, Jack Morris, Junxiong Wang, and Oscar Yin for their suggestions and feedback.
- Some evaluation code is modified upon [Landmark Attention](https://github.com/epfml/landmark-attention), [LongLora](https://github.com/dvlab-research/LongLoRA), [LongBench](https://github.com/THUDM/LongBench), [Long Context Data Engerneering](https://github.com/FranxYao/Long-Context-Data-Engineering) and [RULER](https://github.com/hsiehjackson/RULER).

## Citation
If you find it helpful, please kindly cite the paper.
<!-- ```

``` -->

<!-- ## License -->