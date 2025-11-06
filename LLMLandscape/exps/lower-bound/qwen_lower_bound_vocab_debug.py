import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from models import Qwen2, Llama31, Mistral, Llama3


torch.set_grad_enabled(False)
llama2 = Qwen2()
model = llama2.model
tokenizer = llama2.tokenizer
# 因为有的模型tokenizer词数比embedding还多，这个是为了防止这个bug
vocab_size = tokenizer.vocab_size
embedding = model.model.embed_tokens.weight.cuda().to(torch.bfloat16)[:vocab_size,]  # N, D
norm = (embedding**2).sum(1)  # N
import pdb
pdb.set_trace()
batch_size = 4096
n_vocab = embedding.shape[0]
l2_values = []

for i in range(0, n_vocab, batch_size):
    i_end = min(i + batch_size, n_vocab)
    emb_i = embedding[i:i_end]  # B, D
    gram_batch = norm[i:i_end].view(-1, 1) + norm.view(1, -1)  # B, |V|
    mse_batch = gram_batch - 2 * emb_i @ embedding.T
    l2_values.append(torch.sqrt(mse_batch).cpu())

l2_values = torch.cat(l2_values)
l2_values.diagonal().zero_().add_(-100)   # discard diagonal
import pdb
pdb.set_trace()
reference = 0.1
certifiable_position = (0 < l2_values) & (l2_values <= reference)


# for debugging
# l2s = l2_values[l2_values > 0].sort()[0]
# import pdb
# pdb.set_trace()
l2_values = l2_values.cpu()
true_indices = torch.nonzero(certifiable_position)
certifiable_dict = dict()

for i in range(true_indices.shape[0]):
    k, v = true_indices[i][0], true_indices[i][1]
    k, v = tokenizer.decode(k), tokenizer.decode(v)
    if k not in certifiable_dict:
        certifiable_dict[k] = set()
    certifiable_dict[k].add(v)
print(certifiable_dict.keys())
import pdb

pdb.set_trace()

