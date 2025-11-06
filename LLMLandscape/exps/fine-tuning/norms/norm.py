from models import Qwen2, Qwen2DistilledByDeepSeekR1, Llama2
import math


def model_distance(model_a, model_b):
    l2_square = sum([((a - b) ** 2).sum().item() for a, b in zip(model_a.parameters(), model_b.parameters())])
    l2 = math.sqrt(l2_square)
    print("model distance is: ", l2)
    return l2


# ori_model = Qwen2().cuda()
# ori_model = Qwen2("Qwen/Qwen2.5-Math-7B-Instruct").cuda()
# worst_model = Qwen2("Qwen/Qwen2.5-7B-Instruct-1M").cuda()
# model_distance(ori_model, worst_model)
# worst_model = Qwen2("Qwen/Qwen2.5-7B").cuda()
# model_distance(ori_model, worst_model)
# worst_model = Qwen2(f"./results/benign-harmful-sft/benign/ultra-5e-05-qwen/checkpoint-200/").cuda()
# model_distance(ori_model, worst_model)
# worst_model = Qwen2("./results/benign-harmful-sft/benign/benign-5e-05-qwen/checkpoint-650/").cuda()
# model_distance(ori_model, worst_model)
# worst_model = Qwen2("./results/benign-harmful-sft/benign/benign-5e-05-qwen/checkpoint-50/").cuda()
# model_distance(ori_model, worst_model)
# worst_model = Qwen2("./results/benign-harmful-sft/benign/benign-2e-05-qwen/checkpoint-60/").cuda()
# model_distance(ori_model, worst_model)
# worst_model = Qwen2("./results/benign-harmful-sft/worst/worst-5e-05-qwen/checkpoint-5/").cuda()
# model_distance(ori_model, worst_model)
# worst_model=Qwen2DistilledByDeepSeekR1().cuda()
# worst_model = Qwen2("Qwen/Qwen2.5-Math-7B-Instruct").cuda()
# model_distance(ori_model, worst_model)
ori_model = Llama2()
worst_model = Llama2("meta-llama/Llama-2-7b-chat")
model_distance(ori_model, worst_model)
worst_model = Llama2("meta-llama/Llama-2-7b")
model_distance(ori_model, worst_model)
