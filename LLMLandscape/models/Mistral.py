import torch
from torch import Tensor
from typing import Tuple, List
from .BaseModel import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralForCausalLM

__all__ = ["Mistral"]


class Mistral(BaseModel):
    def __init__(
        self,
        mistral_path="mistralai/Ministral-8B-Instruct-2410",
        mistral_tokenizer_path="mistralai/Ministral-8B-Instruct-2410",
        dtype=torch.bfloat16,
        *args,
        **kwargs,
    ):
        model = AutoModelForCausalLM.from_pretrained(mistral_path, torch_dtype=dtype, trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(mistral_tokenizer_path, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        super(Mistral, self).__init__(model, tokenizer, *args, **kwargs)

    def get_prompt(self, usr_prompt, adv_suffix, target, *args, **kwargs) -> Tuple[Tensor, slice, slice, slice]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": usr_prompt + " " + adv_suffix},
            {"role": "assistant", "content": target},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False if len(target) > 0 else True,
        )
        now_tokens = self.tokenizer([text], return_tensors="pt").input_ids.to(self.device)
        return now_tokens.squeeze(), None, None, None
