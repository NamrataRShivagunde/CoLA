import torch
import math
from models import Llama2
from copy import deepcopy
from transformers import LlamaModel

def model_distance(model_a, model_b):
    l2_square = sum([((a - b) ** 2).sum().item() for a, b in zip(model_a.parameters(), model_b.parameters())])
    l2 = math.sqrt(l2_square)
    print("model distance is: ", l2)
    return l2


def llama2_to_hugging_face_string_convert(inputs: str) -> str:
    # attention part
    result = inputs.replace("attention", "self_attn").replace("tok_embeddings", "embed_tokens")
    result = result.replace("wq", "q_proj").replace("wk", "k_proj")
    result = result.replace("wo", "o_proj").replace("wv", "v_proj")
    # mlp part
    result = result.replace("feed_forward.w3", "mlp.gate_proj")
    result = result.replace("feed_forward.w1", "mlp.up_proj")
    result = result.replace("feed_forward.w2", "mlp.down_proj")
    # Norm
    result = result.replace("self_attn_norm", "input_layernorm")
    result = result.replace("ffn_norm", "post_attention_layernorm")
    # last
    result = result.replace("model.output.weight", "lm_head.weight")
    return result


model_hf = Llama2("meta-llama/Llama-2-7b-chat-hf", device="cpu").model
model_ori_ckpt = torch.load("./resources/checkpoints/llama2-7b-chat.pth")
model_ori = deepcopy(model_hf)
model_ori_ckpt = {"model." + k: v for k, v in model_ori_ckpt.items()}
model_ori_ckpt = {llama2_to_hugging_face_string_convert(k): v for k, v in model_ori_ckpt.items()}
del model_ori_ckpt["model.rope.freqs"]  # 我不太了解rope。我有疑问，难道huggingface版本里rope不需要freq吗？以及，freq为啥会是buffer？这玩意也不需要累计吧，
model_ori.load_state_dict(model_ori_ckpt)
model_distance(model_ori, model_hf)


