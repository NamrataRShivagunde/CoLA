import numpy as np
import os
from matplotlib import pyplot as plt



# plt.rcParams["text.usetex"] = True  # 全局启用 LaTeX 渲染


def benchmark_to_loss_landscape(ys: np.array):
    ys = np.concatenate([ys[1:][::-1], ys])  # Note, 横轴都是正的，另一半只是对称过去的
    # loss越小最好
    ys = 1 - ys
    # 最小值设置为0
    ys = ys - np.min(ys)
    # normalize到0-1区间
    ys = ys / np.max(ys)
    return ys


os.makedirs("./landscape/aggregated/", exist_ok=True)

model_name = "qwen-to-1m"
x = np.arange(-0.01, 0.01, 1e-3)[1:]

model_name = model_name + "-high"
advbench = np.load(f"./landscape/data/benign_origin/advbench-{model_name}.npy")
gsm8k = np.load(f"./landscape/data/benign_origin/gsm8k-{model_name}.npy")
truthfulqa = np.load(f"./landscape/data/benign_origin/mmlu-{model_name}.npy")
humaneval = np.load(f"./landscape/data/benign_origin/humaneval-{model_name}.npy")

advbench = np.concatenate([advbench[1:][::-1], advbench])
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

xticks = plt.gca().get_xticks()
xtick_labels = [f"{abs(val):.3f}" if val != 0 else "0" for val in xticks]
plt.xticks(xticks, xtick_labels)

plt.legend()
os.makedirs(f"./landscape/aggregated/benign_origin/", exist_ok=True)
plt.savefig(f"./landscape/aggregated/benign_origin/{model_name}.pdf", bbox_inches="tight")
# plt.show()
plt.close()
