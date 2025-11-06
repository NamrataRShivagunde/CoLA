import torch
from torch import Tensor, nn
from typing import Tuple, List
from .BaseModel import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM, Qwen2Tokenizer

__all__ = ["Qwen2", "Qwen2DistilledByDeepSeekR1"]


class Qwen2(BaseModel):
    def __init__(
        self,
        qwen_path="Qwen/Qwen2.5-7B-Instruct",
        qwen_tokenizer_path="Qwen/Qwen2.5-7B-Instruct",
        dtype=torch.bfloat16,
        *args,
        **kwargs,
    ):
        model = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=dtype, trust_remote_code=True).eval()
        tokenizer = Qwen2Tokenizer.from_pretrained(qwen_tokenizer_path, trust_remote_code=True, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = None
        model.generation_config.top_k = None
        model.generation_config.do_sample = True
        # model.lm_head.weight.data = model.lm_head.weight.data[:tokenizer.vocab_size]
        # model.resize_token_embeddings(tokenizer.vocab_size)
        # 这个是为了fix qwen的tokenizer和embedding的vocab size不一样的bug。我不知道我这样fix是不是最标准的
        model.lm_head.bias = nn.Parameter(
            torch.zeros((model.lm_head.weight.shape[0],), dtype=torch.bfloat16),
            requires_grad=False,
        )
        # model.lm_head.bias[tokenizer.vocab_size :] = -10000
        model.lm_head.bias[151664:] = -10000
        super(Qwen2, self).__init__(model, tokenizer, *args, **kwargs)

    def get_prompt(self, usr_prompt, adv_suffix, target, *args, **kwargs) -> Tuple[Tensor, slice, slice, slice]:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "},
            {"role": "user", "content": usr_prompt + " " + adv_suffix},
            {"role": "assistant", "content": target},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )[:-11]
        now_tokens = self.tokenizer([text], return_tensors="pt").input_ids.to(self.device)
        return now_tokens.squeeze(), None, None, None


class Qwen2DistilledByDeepSeekR1(Qwen2):
    def __init__(
        self,
        qwen_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        generation_max_length: int = 2000,
        return_thinking: bool = True,
        *args,
        **kwargs,
    ):
        super(Qwen2DistilledByDeepSeekR1, self).__init__(
            qwen_path, generation_max_length=generation_max_length, *args, **kwargs
        )
        self.return_thinking = return_thinking

    def generate(self, *args, **kwargs):
        result = super(Qwen2DistilledByDeepSeekR1, self).generate(*args, **kwargs)
        if self.return_thinking:
            return result
        return result.split("<|box_end|>")[-1]

    def generate_by_input_ids(self, input_ids: Tensor, *args, **kwargs) -> Tensor:
        result = self.model.generate(input_ids, temperature=0.6, *args, **kwargs)
        return result


# def get_prompt(self, usr_prompt, adv_suffix, target, *args, **kwargs) -> Tuple[Tensor, slice, slice, slice]:
#     self.conv.messages.clear()
#     self.conv.append_message(self.conv.roles[0], f"{usr_prompt}")
#
#     now_tokens = self.tokenizer(self.conv.get_prompt()[:-11], *args, **kwargs).input_ids
#     usr_prompt_length = len(now_tokens)
#
#     self.conv.update_last_message(f"{usr_prompt} {adv_suffix} ")
#     now_tokens = self.tokenizer(self.conv.get_prompt()[:-11], *args, **kwargs).input_ids
#     total_prompt_length = len(now_tokens)
#     grad_slice = slice(usr_prompt_length, total_prompt_length - 1)
#
#     self.conv.append_message(self.conv.roles[1], "")
#     now_tokens = self.tokenizer(self.conv.get_prompt(), *args, **kwargs).input_ids
#     target_slice_left = len(now_tokens)
#
#     self.conv.update_last_message(f"{target}")
#     now_tokens = self.tokenizer(self.conv.get_prompt()[:-11], *args, **kwargs).input_ids
#     target_slice_right = len(now_tokens)
#     target_slice = slice(target_slice_left, target_slice_right)  # for output logits
#     loss_slice = slice(target_slice_left - 1, target_slice_right - 1)
#
#     return torch.tensor(now_tokens, device=self.device), grad_slice, target_slice, loss_slice
