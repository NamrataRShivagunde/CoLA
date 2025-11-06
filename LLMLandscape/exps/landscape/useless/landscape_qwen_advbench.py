import torch
from models import Qwen2, Llama31, Llama31DistilledByDeepSeekR1, Qwen2DistilledByDeepSeekR1
from data import get_adv_bench_behaviors_50
from tester import test_vanilla_harmful_output_rate
from utils.plot import Landscape4Model

# model = Qwen2("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B").cuda()
model = Qwen2("Qwen/Qwen2.5-7B-Instruct").cuda()
# print(model.generate("Who are you? "))
# print(model.generate("Who is the president of US"))
# print(model.generate("prove the lipschitz upper bound of g(x)=E[f(x)] when f(x) in [0,1]"))
model.generation_max_length = 100
loader = get_adv_bench_behaviors_50()

# Choice 1: test ASR
test_vanilla_harmful_output_rate(model, loader, verbose=True)
# Choice 2: Gaussian Perturbation with Weight Norm
# drawer = Landscape4Model(
#     model, lambda m: test_vanilla_harmful_output_rate(m, loader, verbose=True), direction="WeightNormGaussian"
# )
# coordinate_config = dict(x_min=-0.4, x_max=0.4, x_interval=0.1, y_min=-0.02, y_max=0.02, y_interval=1e-3)

# Choice 3: Naive Gaussian
drawer = Landscape4Model(
    model,
    lambda m: test_vanilla_harmful_output_rate(m, loader, verbose=False),
    direction="Gaussian",
    device=torch.device("cpu"),
)
coordinate_config = dict(x_min=-0.009, x_max=0.009, x_interval=1e-3, y_min=-5e-3, y_max=5e-3, y_interval=1e-3)
drawer.synthesize_coordinates(**coordinate_config)
drawer.draw(saving_name="advbench")
