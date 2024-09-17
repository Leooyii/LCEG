import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from opencompass.models.base import BaseModel
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
from opencompass.models.huggingface import HuggingFace
from .llama_infinite.modeling_llama import LlamaForCausalLM
from .llama_infinite.llama import convert_llama_model

PromptType = Union[PromptList, str]


@MODELS.register_module()
class HuggingFaceCausalLM_LM_Infinite(HuggingFace):

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):

        model_kwargs.setdefault('torch_dtype', torch.float16)
        self.model = LlamaForCausalLM.from_pretrained(path, **model_kwargs)
        self.model = convert_llama_model(self.model, 4096, 10)
        self.model.eval()

        if 'llama' in path:
            self.logger.warning('A patch for llama when batch_padding = True')  
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
    def generate(self, inputs: List[str], max_out_len: int,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        kwargs['use_cache'] = True
        if self.batch_padding and len(inputs) > 1:
            return self._batch_generate(inputs=inputs,
                                        max_out_len=max_out_len,
                                        **kwargs)
        else:
            return sum((self._single_generate(
                inputs=[input_], max_out_len=max_out_len, **kwargs)
                        for input_ in inputs), [])