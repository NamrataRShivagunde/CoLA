import numpy as np
import os
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["qwen", "llama", "mistral", "qwen-32b"])
args = parser.parse_args()
model_name = args.model
# plt.rcParams["text.usetex"] = True  # 全局启用 LaTeX 渲染


def benchmark_to_loss_landscape(ys: np.array):
    # loss越小最好
    ys = 1 - ys
    # 最小值设置为0
    ys = ys - np.min(ys)
    # normalize到0-1区间
    ys = ys / np.max(ys)
    return ys


os.makedirs("./landscape/aggregated/", exist_ok=True)
if "qwen" in model_name:
    x = np.arange(-0.01, 0.01, 0.5e-3)
else:
    x = np.arange(-0.005, 0.005, 0.25e-3)
model_name = model_name + "-high"
advbench = np.load(f"./landscape/data/advbench-{model_name}.npy")
gsm8k = np.load(f"./landscape/data/gsm8k-{model_name}.npy")
truthfulqa = np.load(f"./landscape/data/mmlu-{model_name}.npy")
humaneval = np.load(f"./landscape/data/humaneval-{model_name}.npy")


gsm8k = benchmark_to_loss_landscape(gsm8k)
truthfulqa = benchmark_to_loss_landscape(truthfulqa)
humaneval = benchmark_to_loss_landscape(humaneval)

# plt.xlim(-0.075, 0.075)
plt.plot(x, advbench, label="Safety")
plt.plot(x, gsm8k, label="Math")
plt.plot(x, truthfulqa, label="Basic")
plt.plot(x, humaneval, label="Coding")
plt.xlabel(r"$\alpha$")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"./landscape/aggregated/{model_name}.pdf", bbox_inches="tight")
# plt.show()
plt.close()
