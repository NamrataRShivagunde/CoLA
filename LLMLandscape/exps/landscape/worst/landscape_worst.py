import argparse
import torch
import numpy as np
import math
from models import Qwen2, Llama31, Mistral
from data import get_adv_bench_behaviors_50
from tester import test_vanilla_harmful_output_rate, lm_eval_truthfulqa, lm_eval_gsm8k, lm_eval_humaneval, lm_eval_mmlu
from utils.plot import Landscape4Model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["qwen", "llama", "mistral"], default="qwen")
args = parser.parse_args()
if args.model == "qwen":
    model = Qwen2("Qwen/Qwen2.5-7B-Instruct")
    worst_model = Qwen2("./results/benign-harmful-sft/worst/worst-5e-05-qwen/checkpoint-10/")
elif args.model == "llama":
    model = Llama31()
    worst_model = Llama31("./results/benign-harmful-sft/worst/worst-5e-05-llama/checkpoint-10/")
else:
    model = Mistral()
    worst_model = Mistral("./results/benign-harmful-sft/worst/worst-5e-05-mistral/checkpoint-10/")
worst_model.cuda()
model.cuda()
num_params = sum([i.numel() for i in model.parameters()])
l2_square = sum([((a - b) ** 2).sum().item() for a, b in zip(model.parameters(), worst_model.parameters())])
l2 = math.sqrt(l2_square)
norm_gaussian = math.sqrt(num_params)
revise_factor = norm_gaussian / l2
print("norm gaussian is ", norm_gaussian)
print("norm perturbation is ", l2)
print("revise factor is ", revise_factor)


def get_benchmark(task_name: str):
    if task_name == "advbench":
        loader = get_adv_bench_behaviors_50()
        benchmark = lambda m, verbose=False: test_vanilla_harmful_output_rate(m, loader, verbose=verbose)
        model.generation_max_length = 100
    elif task_name == "truthfulqa":
        benchmark = lambda m, verbose=False: lm_eval_truthfulqa(m, verbose=verbose, limit=100)
        model.generate = model.generate_by_input_ids
    elif task_name == "humaneval":
        benchmark = lambda m, verbose=False: lm_eval_humaneval(m, verbose=verbose, limit=10)
        model.generate = model.generate_by_input_ids
    elif task_name == "mmlu":
        benchmark = lambda m, verbose=False: lm_eval_mmlu(m, verbose=verbose, limit=1)
        model.generate = model.generate_by_input_ids
    else:
        benchmark = lambda m, verbose=False: lm_eval_gsm8k(m, verbose=verbose)
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
drawer = Landscape4Model(
    model,
    None,
    direction="Gaussian",
    device=torch.device("cuda"),
    save_path="./landscape/worst/",
)


@torch.no_grad()
def find_worst_direction(target_drawer: Landscape4Model):
    target_drawer.x_unit_vector = {}
    target_drawer.y_unit_vector = {}
    for (name, param), (_, worst_param) in zip(model.named_parameters(), worst_model.named_parameters()):
        target_drawer.x_unit_vector[name] = (worst_param.data - param.data) * revise_factor
        target_drawer.y_unit_vector[name] = torch.tensor(0)


find_worst_direction(drawer)

if args.model == "qwen":
    coordinate_config = dict(x_min=0, x_max=0.01, x_interval=0.5e-3, y_min=-5e-3, y_max=5e-3, y_interval=1e-3)
else:
    coordinate_config = dict(x_min=0, x_max=0.005, x_interval=0.25e-3, y_min=-5e-3, y_max=5e-3, y_interval=1e-3)
drawer.synthesize_coordinates(**coordinate_config)
drawer.mesh_x = np.stack([drawer.mesh_x[:, 0], drawer.mesh_x[:, 1], drawer.mesh_x[:, -1]], axis=1)


for task in ["advbench", "humaneval", "gsm8k", "mmlu"]:
    saving_name = task + "-" + args.model
    benchmark = get_benchmark(task)
    drawer.loss = benchmark
    result = drawer.draw(saving_name=saving_name, random_direction=False)
    np.save(f"./landscape/data/worst/{saving_name}-high.npy", result)
    print(result)
