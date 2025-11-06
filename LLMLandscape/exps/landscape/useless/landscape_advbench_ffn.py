import torch
from models import Vicuna15, Llama3, Llama2
from data import get_adv_bench_behaviors_50
from tester import test_vanilla_harmful_output_rate
from utils.plot import Landscape4Model


# model = Llama3("/home/vip/.cache/huggingface/Meta-Llama-3-8B-Instruct/")
# model = Llama2("lmsys/vicuna-7b-v1.5")  # Vicuna 1.5 here
model = Llama2().cuda()
loader = get_adv_bench_behaviors_50()
loader = list(loader)[:2]

# Choice 1: test ASR
# test_vanilla_advbench_harmful_output_rate(model, loader)

# Choice 2: Gaussian Perturbation with Weight Norm
# drawer = Landscape4Model(
#     model, lambda m: test_vanilla_harmful_output_rate(m, loader, verbose=True), direction="WeightNormGaussian"
# )
# coordinate_config = dict(x_min=-0.4, x_max=0.4, x_interval=0.1, y_min=-0.02, y_max=0.02, y_interval=1e-3)

# Choice 3: Naive Gaussian
drawer = Landscape4Model(
    model,
    lambda m: test_vanilla_harmful_output_rate(m, loader, verbose=True),
    direction="Gaussian",
    device=torch.device("cpu")
)
# coordinate_config = dict(x_min=-0.009, x_max=0.009, x_interval=1e-3, y_min=-5e-3, y_max=5e-3, y_interval=1e-3)
coordinate_config = dict(x_min=-0.09, x_max=0.09, x_interval=5e-3, y_min=-5e-3, y_max=5e-3, y_interval=1e-3)
drawer.synthesize_coordinates(**coordinate_config)
param_names = [name for name, _ in model.named_parameters()]
for n, p in model.named_parameters():
    print(n, p.shape)
ffn_exclusion = [i for i in param_names if "mlp" not in i]
attn_exclusion = [i for i in param_names if "attn" not in i]
exclusion = [i for i in param_names if "embed" not in i]
# exclusion = [i for i in param_names if "mlp" not in i and "attn" not in i]
print("excluded parameters: ", exclusion)  # 必须是真正的parameter name，丝毫不差
drawer.find_direction(exclusion)
drawer.draw(random_direction=False, saving_name="embedding")
