# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import torch
from typing import Dict, List, Optional
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(models_dir)


class HuggingFaceModel:
    def __init__(self, name_or_path: str, model_full_name: str, max_seq_length: int, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        model_kwargs = {"use_flash_attention_2": "True"}
        if model_full_name == "llama2-7b-hf" or model_full_name == "llama-2-7b-hf-slimpajama-pi-32k":
            print(f'testing {model_full_name}')
            print(f'load model:{name_or_path}')
            self.pipeline = None
                # Set RoPE scaling factor
            config = AutoConfig.from_pretrained(
                name_or_path
            )
            print(config.rope_scaling)
            print('testing length:', max_seq_length)
            
            # Load model and tokenizer
            import transformers
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                name_or_path,
                config=config,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            generation_kwargs['use_cache'] = True
        elif model_full_name == "llama2-7b-hf-ntk-frozen":
            print(f'testing {model_full_name}')
            print(f'load model:{name_or_path}')
            self.pipeline = None
                # Set RoPE scaling factor
            config = AutoConfig.from_pretrained(
                name_or_path
            )
            scaling_factor = 2.0
            config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
            print(config.rope_scaling)
            print('testing length:', max_seq_length)

            
            # Load model and tokenizer
            import transformers
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                name_or_path,
                config=config,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            generation_kwargs['use_cache'] = True

        elif model_full_name == "llama-2-7b-hf-slimpajama-ntk-32k":
            print(f'testing {model_full_name}')
            print(f'load model:{name_or_path}')
            self.pipeline = None
                # Set RoPE scaling factor
            config = AutoConfig.from_pretrained(
                name_or_path
            )
            print(config.rope_scaling)
            print('testing length:', max_seq_length)
            
            # Load model and tokenizer
            from models.llama_ntk_32k import LlamaForCausalLM
            self.model = LlamaForCausalLM.from_pretrained(
                name_or_path,
                config=config,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            generation_kwargs['use_cache'] = True
        elif model_full_name == "llama-2-7b-hf-slimpajama-ntk-64k":
            print(f'testing {model_full_name}')
            print(f'load model:{name_or_path}')
            self.pipeline = None
                # Set RoPE scaling factor
            config = AutoConfig.from_pretrained(
                name_or_path
            )
            print(config.rope_scaling)
            print('testing length:', max_seq_length)
            
            # Load model and tokenizer
            from models.llama_ntk_64k import LlamaForCausalLM
            self.model = LlamaForCausalLM.from_pretrained(
                name_or_path,
                config=config,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            generation_kwargs['use_cache'] = True

        elif model_full_name == "llama2-7b-hf-lminfinite":
            print('testing lminfinite')
            print(f'load model:{name_or_path}')
            self.pipeline = None
            from models.llama_infinite import LlamaForCausalLM
            from models.llama_infinite.llama import convert_llama_model
            self.model = LlamaForCausalLM.from_pretrained(
                name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = convert_llama_model(self.model, 4096, 10)
            generation_kwargs['use_cache'] = True

        elif model_full_name == "llama2-7b-hf-selfextend":
            print('testing selfextend')
            print(f'load model:{name_or_path}')
            self.pipeline = None

            from transformers import AutoModelForCausalLM
            from models.llama_selfextend import SelfExtend
            window_size = 1024
            group_size = 64
            use_flash = True
            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2=use_flash)
            print(f'using group size {group_size} using window size {window_size}')
            SelfExtend.apply(self.model, group_size, window_size, enable_flash_attention=use_flash, flash_attention_impl="flash_attn") ## flash_attention_impl="triton" or "flash_attn"
            
            generation_kwargs['use_cache'] = True

        elif model_full_name == "llama2-7b-hf-slimpajama-yarn-32k":
            print(f'testing {model_full_name}')
            print(f'load model:{name_or_path}')
            self.pipeline = None
            from models.llama_yarn.modeling_llama_yarn import LlamaForCausalLM
            from models.llama_yarn.configuration_llama import LlamaConfig
            config_cls = LlamaConfig
            model_cls = LlamaForCausalLM
            config = config_cls.from_pretrained(name_or_path)
            if model_full_name=="llama2-7b-hf-slimpajama-yarn-dynamic-32k":
                config.rope_scaling = {
                    "type": "dynamic-yarn",
                    "factor": 8.0,
                    "original_max_position_embeddings": 4096
                }

            print(config.rope_scaling)
            self.model = model_cls.from_pretrained(
                name_or_path,
                config=config,
                torch_dtype=torch.float16,
                use_flash_attention_2=True,
                device_map="auto",
            )
            generation_kwargs['use_cache'] = True

        elif model_full_name == "llama2-7b-hf-slimpajama-clex-32k":
            print(f'testing {model_full_name}')
            print(f'load model:{name_or_path}')
            self.pipeline = None
            from models.llama_clex import LlamaForCausalLM, CLEXLlamaConfig
            config = CLEXLlamaConfig.from_pretrained(name_or_path)
            config.log_scale = True
            config.use_flashattn = True
            print(config.rope_scaling, flush=True)
            self.model = LlamaForCausalLM.from_pretrained(
                name_or_path,
                config=config,
                torch_dtype=torch.float16,
                use_flash_attention_2=True,
                device_map="auto",
            )
            generation_kwargs['use_cache'] = True

        elif model_full_name == "llama2-7b-hf-slimpajama-landmark":
            print('testing landmark')
            print(f'load model:{name_or_path}')
            self.pipeline = None
            from models.llama_landmark.llama_mem import LlamaForCausalLM
            self.model = LlamaForCausalLM.from_pretrained(
                name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                )
            tokenizer = AutoTokenizer.from_pretrained(
                name_or_path,
                padding_side="right",
                use_fast=False,
            )
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token 
            
            mem_id = tokenizer.convert_tokens_to_ids("<landmark>")
            self.model.set_mem_id(mem_id)
            generation_kwargs['use_cache'] = True
            generation_kwargs['offload_cache_to_cpu'] = False
            generation_kwargs['use_flash'] = False
            generation_kwargs['cache_top_k'] = 5


        else:
            try:
                print(f'load model:{name_or_path}')
                print('use pipeline')
                self.pipeline = pipeline(
                    "text-generation",
                    model=name_or_path,
                    tokenizer=self.tokenizer,
                    # trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    model_kwargs=model_kwargs,
                )
                generation_kwargs['use_cache'] = True
            except Exception as e:
                print(f'{e}')
                print('load model except')
                self.pipeline = None
                self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,)
            
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompt, **self.generation_kwargs,)
            assert len(output) == 1
            generated_text = output[0]["generated_text"]
            
        # remove the input form the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]
                
        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]
        return {'text': [generated_text]}


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1] :])]}