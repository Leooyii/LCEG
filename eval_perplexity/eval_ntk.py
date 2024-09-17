import os
import math
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
from peft import PeftModel
import json

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--seq_len', type=int, default=2048, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--data_path', type=str, default="./test.bin", help='')
    parser.add_argument('--base_factor', type=int, default=8, help='base factor to scale rope base')
    parser.add_argument('--output_dir', type=str, default='')
    args = parser.parse_args()
    return args

def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    all_ix.pop()

    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx+batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y

def iceildiv(x, y):
    return (x + y - 1) // y

def evaluate(model, data, batch_size, device, seq_length, sliding_window=256, use_cache=False):
    stats = {}

    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []


    # use first ten documents for pg19
    doc_start_idx = np.flatnonzero(data['val'] == data['val'][0])
    print('total tokens:',len(data['val']))
    print('selected 10 documents total tokens:', len(data['val'][:doc_start_idx[10]]))
    data['val'] = data['val'][:doc_start_idx[10]]

    with torch.no_grad():
        print(f"Using seq length {seq_length}")
        torch.set_printoptions(sci_mode=False)
        for idx, (x, y) in tqdm(
            enumerate(
                get_as_batch(
                    data['val'], 
                    seq_length, 
                    batch_size, 
                    device=device,
                    sliding_window=sliding_window
                )
            ),
            total=iceildiv(
                iceildiv(len(data['val']), sliding_window),
                batch_size
            )
        ):

            val_loss = 0.
            acc = 0.
            cnt = 0

            for part_idx, i in enumerate(range(0, x.shape[1], seq_length)):
                part_len = x[:, i:i + seq_length].shape[1]

                outputs = model(
                    input_ids=x[:, i:i + seq_length],
                    labels=x[:, i:i+seq_length].contiguous(),
                    use_cache=use_cache)

                val_loss = outputs.loss * part_len + val_loss
                acc = ((outputs.logits.argmax(-1) == y[:, i:i+seq_length]).float().sum()) + acc
                cnt += part_len
                while len(loss_step_list_val) <= part_idx:
                    loss_step_list_val.append([])
                loss_step_list_val[part_idx].append(outputs.loss.item())
            val_loss /= cnt
            acc /= cnt
            
            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())

    print('acc list:')
    print(acc_list)
    print('loss_list_val:')
    print(loss_list_val)
    print('loss_step_list_val:')
    print(loss_step_list_val)

    stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['val_perplexity_per_chunk'] = torch.exp(torch.as_tensor(loss_step_list_val).mean(dim=1)).tolist()

    return stats

def main(args):

    device = "cuda:0"
    seed = 2
    torch.cuda.set_device(device)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data = {'val': np.memmap(args.data_path, dtype=np.uint16, mode='r')}

    print(f"Num validation tokens: {len(data['val'])}")
    print("data path", args.data_path)
    print("base model", args.base_model)
    print("peft model", args.peft_model)


    if args.model_name == "llama-2-7b-hf-ntk-frozen":
        # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
        )
        
        scaling_factor = 2.0
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
        print(config.rope_scaling)

        # Load model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=config,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    elif args.model_name == "llama-2-7b-hf-slimpajama-ntk-32k":
        # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
        )
        print(config.rope_scaling)
         # Load model and tokenizer
        from models.llama_ntk_32k import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            config=config,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

    elif args.model_name == "llama-2-7b-hf-slimpajama-ntk-64k":
        # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
        )
        print(config.rope_scaling)
         # Load model and tokenizer
        from models.llama_ntk_64k import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            config=config,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )



    stats = evaluate(model, data, args.batch_size, device, args.seq_len, sliding_window=256)

    print(stats)
    with open(args.output_dir, 'w') as json_file:
        json.dump(stats, json_file, indent=4)

if __name__ == "__main__":
    args = parse_config()
    main(args)
