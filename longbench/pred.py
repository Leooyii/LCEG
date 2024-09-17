import os
from datasets import load_dataset
import torch
import json
import transformers
from tqdm import tqdm
import numpy as np
import random
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
sys.path.append(models_dir)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)



def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    data = data['test']
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        kwargs = {}
        kwargs['use_cache'] = True
        if model_name == "llama2-7b-hf-slimpajama-landmark" or model_name == "llama2-7b-hf-slimpajama-landmark-test4k":  
            kwargs['offload_cache_to_cpu'] = False
            kwargs['use_flash'] = False
            kwargs['cache_top_k'] = 5
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                **kwargs,
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                **kwargs,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    print('testing:', model_name)
    print('model path:', path)
    if model_name == "llama2-7b-hf" or model_name == "llama2-7b-hf-slimpajama-pi-32k" or model_name == "llama2-7b-hf-slimpajama-longlora-32k":
        config = transformers.AutoConfig.from_pretrained(path)
        print('rope_scaling:', config.rope_scaling)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
        )
    elif model_name == "llama2-7b-hf-slimpajama-ntk-32k":
        config = transformers.AutoConfig.from_pretrained(path)
        print('rope_scaling:', config.rope_scaling)
        from models.llama_ntk_32k import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
        )
    elif model_name == "llama2-7b-hf-slimpajama-ntk-64k" or model_name == "llama-2-7b-hf-slimpajama-ntk-64k-2B":
        config = transformers.AutoConfig.from_pretrained(path)
        print('rope_scaling:', config.rope_scaling)
        from models.llama_ntk_64k import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
        )
    elif model_name == "llama2-7b-hf-lminfinite":
        from models.llama_infinite import LlamaForCausalLM
        from models.llama_infinite.llama import convert_llama_model
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = convert_llama_model(model, 4096, 10)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
        )
    elif model_name == "llama2-7b-hf-ntk-frozen":
            # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            path,
        )
        
        scaling_factor = 2.0
        print(config.rope_scaling)
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}

        # Load model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
        )
        
    elif model_name == "llama2-7b-hf-slimpajama-yarn-32k":
        from models.llama_yarn.modeling_llama_yarn import LlamaForCausalLM
        from models.llama_yarn.configuration_llama import LlamaConfig
        config_cls = LlamaConfig
        model_cls = LlamaForCausalLM
        config = config_cls.from_pretrained(path)
        print(config.rope_scaling)
        model = model_cls.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
        )
    elif model_name == "llama2-7b-hf-selfextend":
        from transformers import AutoModelForCausalLM
        from models.llama_selfextend import SelfExtend
        window_size = 1024
        group_size = 64
        use_flash = True
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2=use_flash)
        print(f'using group size {group_size} using window size {window_size}')
        SelfExtend.apply(model, group_size, window_size, enable_flash_attention=use_flash, flash_attention_impl="flash_attn") ## flash_attention_impl="triton" or "flash_attn"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
        )
    elif model_name == "llama2-7b-hf-slimpajama-clex-32k":
        print('eval clex')
        from models.llama_clex import LlamaForCausalLM, CLEXLlamaConfig
        config = CLEXLlamaConfig.from_pretrained(path)
        config.log_scale = True
        config.use_flashattn = True
        print(config.rope_scaling, flush=True)
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            config=config,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
        )

    elif model_name == "llama2-7b-hf-slimpajama-landmark":
        from models.llama_landmark.llama_mem import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token 
        
        mem_id = tokenizer.convert_tokens_to_ids("<landmark>")
        model.set_mem_id(mem_id)
        model = model.to(device)
    else:
        print('ERROR! Model_name not exist!')
        exit()
        

    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    max_length = model2maxlen[model_name]
    print('max_length:', max_length)

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred/"):
        os.makedirs("pred/")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    dataset_path = "data"
    dataset = args.dataset_name
    print('testing:', dataset)
    if args.e:
        # data = load_dataset(dataset_path, f"{dataset}_e", split='test')
        print('using longbench-e')
        data = load_dataset('json', data_files={'test': os.path.join(dataset_path, f'{dataset}_e.jsonl')})
        if not os.path.exists(f"pred_e/{model_name}"):
            os.makedirs(f"pred_e/{model_name}")
        out_path = f"pred_e/{model_name}/{dataset}.jsonl"
    elif "trec" in dataset and dataset != "trec":
        data = load_dataset('json', data_files={'test': os.path.join(dataset_path, f'{dataset}.jsonl')})
        if not os.path.exists(f"pred_trec/{model_name}"):
            os.makedirs(f"pred_trec/{model_name}")
        out_path = f"pred_trec/{model_name}/{dataset}.jsonl"
    else:
        # data = load_dataset(os.path.join(dataset_path, f'{dataset}.jsonl'))
        # data = load_dataset(dataset_path, dataset, split='test')
        data = load_dataset('json', data_files={'test': os.path.join(dataset_path, f'{dataset}.jsonl')})
        if not os.path.exists(f"pred/{model_name}"):
            os.makedirs(f"pred/{model_name}")
        out_path = f"pred/{model_name}/{dataset}.jsonl"
    if "trec" in dataset:
        dataset = "trec"
    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')