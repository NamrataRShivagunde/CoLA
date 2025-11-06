import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["useless", "llama", "mistral"])
args = parser.parse_args()
model_name = args.model
# plt.rcParams["text.usetex"] = True  # 全局启用 LaTeX 渲染


def benchmark_to_loss_landscape(ys: np.array):
    ys = np.concatenate([ys[-2:][::-1], ys])  # Note, 横轴都是正的，另一半只是对称过去的
    # loss越小最好
    ys = 1 - ys
    # 最小值设置为0
    ys = ys - np.min(ys)
    # normalize到0-1区间
    ys = ys / np.max(ys)
    return ys


os.makedirs("./landscape/aggregated/", exist_ok=True)
if model_name == "useless":
    x = np.array([-0.025, -0.5e-3, 0, 0.5e-3, 0.025])
else:
    x = np.array([-0.005, -0.25e-3, 0, 0.25e-3, 0.005])
model_name = model_name + "-high"
advbench = np.load(f"./landscape/data/worst/advbench-{model_name}.npy")

gsm8k = np.load(f"./landscape/data/worst/gsm8k-{model_name}.npy")
mmlu = np.load(f"./landscape/data/worst/mmlu-{model_name}.npy")
humaneval = np.load(f"./landscape/data/worst/humaneval-{model_name}.npy")

advbench = np.concatenate([advbench[-2:][::-1], advbench])
gsm8k = benchmark_to_loss_landscape(gsm8k)
mmlu = benchmark_to_loss_landscape(mmlu)
humaneval = benchmark_to_loss_landscape(humaneval)

alpha = 0.5
# plt.xlim(-0.075, 0.075)
plt.plot(x, advbench, label="Safety", alpha=alpha)
plt.plot(x, gsm8k, label="Math", alpha=alpha)
plt.plot(x, mmlu, label="Basic", alpha=alpha)
plt.plot(x, humaneval, label="Coding", alpha=alpha)
plt.xlabel(r"$\alpha$")
plt.ylabel("Loss")

xticks = plt.gca().get_xticks()
xtick_labels = [f"{abs(val):.4f}" if val != 0 else "0" for val in xticks]
plt.xticks(xticks, xtick_labels)

plt.legend()
plt.savefig(f"./landscape/worst/aggregated/{model_name}.pdf", bbox_inches="tight")
# plt.show()
plt.close()
