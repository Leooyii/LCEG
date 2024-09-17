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


class HuggingFaceModel:
    def __init__(self, name_or_path: str, model_full_name: str, max_seq_length: int, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            # model_kwargs = {"attn_implementation": "flash_attention_2"}
            model_kwargs = {"use_flash_attention_2": "True"}

        if 'lminfinite' in model_full_name:
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
        elif 'ntk' in model_full_name:
            print(f'testing {model_full_name}')
            print(f'load model:{name_or_path}')
            self.pipeline = None
                # Set RoPE scaling factor
            config = AutoConfig.from_pretrained(
                name_or_path
            )
            maxlength2base_factor = {
                'llama2-7b-hf-ntk-frozen':{'4096':1,'8192':3,'16384':7,'32768':15,'65536':31},
                'llama-2-7b-hf-slimpajama-ntk-32k':{'4096':13,'8192':29,'16384':29,'32768':29,'65536':61},
                'llama2-7b-hf-slimpajama-ntk-64k':{'4096':25,'8192':25,'16384':57,'32768':57,'65536':57},
                'llama2-7b-hf-slimpajama-ntk-64k-2B':{'4096':25,'8192':25,'16384':57,'32768':57,'65536':57},
                }
            print(config.rope_scaling)
            config.base_factor = maxlength2base_factor[model_full_name][str(max_seq_length)]
            print('testing length:', max_seq_length)
            print('using base factor:', config.base_factor)
            base = 10000 * (
                    (config.base_factor)
                ) ** (128 / (128 - 2))
            print('setting base:', base)

            from models.llama_ntk import LlamaForCausalLM
            # Load model and tokenizer
            self.model = LlamaForCausalLM.from_pretrained(
                name_or_path,
                config=config,
                torch_dtype=torch.float16,
                use_flash_attention_2=True,
                device_map="auto",
            )
            generation_kwargs['use_cache'] = True
        elif 'landmark' in model_full_name:
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