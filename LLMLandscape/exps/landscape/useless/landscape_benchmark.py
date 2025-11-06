import torch
from models import Vicuna15, Llama3, Llama2
from tester import lm_eval_gsm8k, lm_eval_mmlu, lm_eval_truthfulqa
from utils.plot import Landscape4Model


# model = Llama3()
# model = Llama2("lmsys/vicuna-7b-v1.5")  # Vicuna 1.5 here
# model = Llama3()  # llama2 GSM8k只有0.29?
# model = Llama2("./results/fine-tuning-attack/default-4e-5-bf16/checkpoint-4/")
model = Llama2("./results/sft-alpaca/baseline/checkpoint-390/")
model.to(torch.device("cuda"))
model.generate = model.generate_by_input_ids  # if GSM8k such generation benchmark, enable this

lm_eval_truthfulqa(model)
# Choice 1: Gaussian Perturbation with Weight Norm
# drawer = Landscape4Model(
#     model, lambda m: test_vanilla_harmful_output_rate(m, loader, verbose=True), direction="WeightNormGaussian"
# )
# coordinate_config = dict(x_min=-0.4, x_max=0.4, x_interval=0.1, y_min=-0.02, y_max=0.02, y_interval=1e-3)

# Choice 2: Naive Gaussian
drawer = Landscape4Model(
    model,
    lambda m: lm_eval_truthfulqa(m),
    direction="Gaussian",
    # device=torch.device("cpu"),  # 那个perturbation是放在cpu上
)
#  coordinate_config = dict(x_min=-0.007, x_max=0.007, x_interval=2e-4, y_min=-5e-3, y_max=5e-3, y_interval=1e-3)
coordinate_config = dict(x_min=-0.07, x_max=0.07, x_interval=1e-3, y_min=-5e-3, y_max=5e-3, y_interval=1e-3)
drawer.synthesize_coordinates(**coordinate_config)
drawer.draw(saving_name="baseline-truthqa.png")
