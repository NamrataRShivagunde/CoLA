import argparse
import os

import torch
import numpy as np
from models import Qwen2, Llama31, Mistral
from data import get_adv_bench_behaviors_50
from tester import test_vanilla_harmful_output_rate, lm_eval_truthfulqa, lm_eval_gsm8k, lm_eval_humaneval, lm_eval_mmlu
from utils.plot import Landscape4Model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["qwen", "llama", "mistral", "qwen05"], default="qwen")
# huangyao's suggestions: ["advbench", "truthfulqa", "livecode", "gpqa", "math500"]
# But, livecode&math500 is hard to implement
parser.add_argument("--task", type=str, choices=["advbench", "truthfulqa", "humaneval", "gsm8k", "mmlu"])
args = parser.parse_args()
if args.model == "qwen":
    model = Qwen2("Qwen/Qwen2.5-7B-Instruct")
elif args.model == "llama":
    model = Llama31()
elif args.model == "qwen05":
    model = Qwen2("Qwen/Qwen2.5-0.5B-Instruct")
else:
    model = Mistral()
model.cuda()
# print(model.generate("Who are you? "))
# print(model.generate("Who is the president of US"))
# print(model.generate("prove the lipschitz upper bound of g(x)=E[f(x)] when f(x) in [0,1]"))


if args.task == "advbench":
    loader = get_adv_bench_behaviors_50()
    benchmark = lambda m, verbose=False: test_vanilla_harmful_output_rate(m, loader, verbose=verbose)
    model.generation_max_length = 100
elif args.task == "truthfulqa":
    benchmark = lambda m, verbose=False: lm_eval_truthfulqa(m, verbose=verbose, limit=100)
    model.generate = model.generate_by_input_ids
elif args.task == "humaneval":
    benchmark = lambda m, verbose=False: lm_eval_humaneval(m, verbose=verbose)
    model.generate = model.generate_by_input_ids
elif args.task == "mmlu":
    benchmark = lambda m, verbose=False: lm_eval_mmlu(m, verbose=verbose)
    model.generate = model.generate_by_input_ids
else:
    benchmark = lambda m, verbose=False: lm_eval_gsm8k(m, verbose=verbose)
    model.generate = model.generate_by_input_ids


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
drawer = Landscape4Model(
    model,
    benchmark,
    direction="Gaussian",
    device=torch.device("cuda"),
)
if args.model == "qwen":
    coordinate_config = dict(x_min=-0.01, x_max=0.01, x_interval=0.5e-3, y_min=-5e-3, y_max=5e-3, y_interval=1e-3)
elif args.model == "qwen05":
    coordinate_config = dict(x_min=-0.005, x_max=0.005, x_interval=1e-3, y_min=-5e-3, y_max=5e-3, y_interval=1e-3)
else:
    coordinate_config = dict(x_min=-0.005, x_max=0.005, x_interval=0.25e-3, y_min=-5e-3, y_max=5e-3, y_interval=1e-3)
drawer.synthesize_coordinates(**coordinate_config)
saving_name = args.task + "-" + args.model
os.makedirs("./landscape/data/", exist_ok=True)
if args.task == "truthfulqa":
    result = 0
    for _ in range(5):
        drawer.find_direction()
        cur_result = drawer.compute_for_draw()
        result += cur_result
    result /= 5
    drawer.draw_figure(drawer.mesh_x, drawer.mesh_y, result, saving_name=saving_name)
    np.save(f"./landscape/data/{saving_name}-high.npy", result)
    print(result)
else:
    result = drawer.draw(saving_name=saving_name)
    np.save(f"./landscape/data/{saving_name}-high.npy", result)
    print(result)
