import matplotlib

matplotlib.use("Agg")
import os
import torch
import numpy as np
import math
from models import Qwen2, Llama31, Mistral, Llama3
from data import get_adv_bench_behaviors_50
from tester import test_vanilla_harmful_output_rate, lm_eval_truthfulqa, lm_eval_gsm8k, lm_eval_humaneval, lm_eval_mmlu
from utils.plot import Landscape4Model


# model = Qwen2("Qwen/Qwen2.5-0.5B-Instruct")
# model=Qwen2()
model = Qwen2("./results/sft-alpaca-qwen/ultra-go-0.0002-region-0.02/checkpoint-5628/")


def generate_mmlu(input_ids, *args, **kwargs):
    result = model.generate_by_input_ids(input_ids, *args, **kwargs)
    result = model.tokenizer.decode(result[0])
    result = result.split("<|im_end|>\n<|im_start|>assistant\n")
    answer = result[1][0]
    result = model.tokenizer([answer], return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    # print(result, model.tokenizer.decode(result[0]))
    result = torch.cat([input_ids, result], dim=1)
    return result


model.cuda()

gen_kwargs = {
    "do_sample": True,
    "temperature": 0.6,
    "top_k": None,
    "top_p": None,
}
#
# system_humaneval = """
# You are Qwen, created by Alibaba Cloud. Implement only the Python function body for the given HumanEval problem. Rules:
# 1. Only write code inside the function, nothing else.
# 2. Use `return` for output, never `print`.
# 3. No extra functions, comments, or imports.
# 4. Use 4-space indentation.
# Extra code or new `def` will fail HumanEval.
# """

system_mmlu = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You will do a single choice question. You can only answer one token, A/B/C/D."


def get_benchmark(task_name: str):
    if task_name == "advbench":
        loader = get_adv_bench_behaviors_50()
        benchmark = lambda m, verbose=False: test_vanilla_harmful_output_rate(m, loader, verbose=verbose)
        model.generation_max_length = 100
    elif task_name == "truthfulqa":
        benchmark = lambda m, verbose=False: lm_eval_truthfulqa(m, verbose=verbose, limit=100)
        model.generate = model.generate_by_input_ids
    elif task_name == "humaneval":
        benchmark = lambda m, verbose=False: lm_eval_humaneval(
            m,
            verbose=verbose,
            limit=100,
            **gen_kwargs,
            # system_instruction=system_humaneval,
        )
        model.generate = model.generate_by_input_ids
    elif task_name == "mmlu":
        benchmark = lambda m, verbose=False: lm_eval_mmlu(
            m,
            verbose=verbose,
            limit=10,
            system_instruction=system_mmlu,
        )
        # model.generate = model.generate_by_input_ids
        model.generate = generate_mmlu
    else:
        benchmark = lambda m, verbose=False: lm_eval_gsm8k(m, verbose=verbose, **gen_kwargs)
        model.generate = model.generate_by_input_ids
    return benchmark


# Choice 1: test plain model
# benchmark(model, verbose=True)
# import pdb
#
# pdb.set_trace()
# Choice 2: Gaussian Perturbation with Weight Norm
# drawer = Landscape4Model(
#     model, lambda m: test_vanilla_harmful_output_rate(m, loader, verbose=True), direction="WeightNormGaussian"
# )
# coordinate_config = dict(x_min=-0.4, x_max=0.4, x_interval=0.1, y_min=-0.02, y_max=0.02, y_interval=1e-3)

# Choice 3: Naive Gaussian
os.makedirs("./landscape/enlarge/", exist_ok=True)
drawer = Landscape4Model(
    model,
    None,
    direction="Gaussian",
    device=torch.device("cuda"),
    save_path="./landscape/enlarge/",
)

coordinate_config = dict(x_min=-0.015, x_max=0.015, x_interval=1e-3, y_min=-5e-3, y_max=5e-3, y_interval=1e-3)
drawer.synthesize_coordinates(**coordinate_config)

os.makedirs("./landscape/data/enlarge/", exist_ok=True)
# for task in ["advbench", "humaneval", "gsm8k", "mmlu"]:
for task in ["mmlu"]:
    # saving_name = task + "-" + "gcwa=AA=0.0002=5e-05"
    saving_name = task + "-" + "go=2e-4=002=5628"
    # saving_name = task + "-" + "baseline"
    benchmark = get_benchmark(task)
    drawer.loss = benchmark
    # benchmark(model, verbose=True)
    result = drawer.draw(saving_name=saving_name)
    np.save(f"./landscape/data/enlarge/{saving_name}-high.npy", result)
    print(result)
