import torch
from torch import Tensor, nn
from typing import Tuple, List
from .BaseModel import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM, Qwen2Tokenizer

__all__ = ["OLMo2"]


class OLMo2(BaseModel):
    def __init__(
        self,
        qwen_path="allenai/OLMo-2-0425-1B-Instruct",
        qwen_tokenizer_path="allenai/OLMo-2-0425-1B-Instruct",
        dtype=torch.bfloat16,
        *args,
        **kwargs,
    ):
        model = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=dtype, trust_remote_code=True).eval()
        tokenizer = Qwen2Tokenizer.from_pretrained(qwen_tokenizer_path, trust_remote_code=True, use_fast=False)
        super(OLMo2, self).__init__(model, tokenizer, *args, **kwargs)

    def get_prompt(self, usr_prompt, adv_suffix, target, *args, **kwargs) -> Tuple[Tensor, slice, slice, slice]:
        messages = [
            {"role": "user", "content": usr_prompt + " " + adv_suffix},
            {"role": "assistant", "content": target},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )[:-13]
        now_tokens = self.tokenizer([text], return_tensors="pt").input_ids.to(self.device)
        return now_tokens.squeeze(), None, None, None

