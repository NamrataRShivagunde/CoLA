import numpy as np
import os
from matplotlib import pyplot as plt


def benchmark_to_loss_landscape(ys: np.array):
    # loss越小最好
    ys = 1 - ys
    # 最小值设置为0
    ys = ys - np.min(ys)
    # normalize到0-1区间
    ys = ys / np.max(ys)
    return ys


os.makedirs("./landscape/aggregated/", exist_ok=True)

x = np.arange(-0.01, 0.01, 1e-3)
model_name = "baseline" + "-high"
# model_name = "region0=938" + "-high"
# model_name = "go4685t" + "-high"
advbench = np.load(f"./landscape/data/enlarge/advbench-{model_name}.npy")
gsm8k = np.load(f"./landscape/data/enlarge/gsm8k-{model_name}.npy")
truthfulqa = np.load(f"./landscape/data/enlarge/mmlu-{model_name}.npy")[5:-5]
humaneval = np.load(f"./landscape/data/enlarge/humaneval-{model_name}.npy")

advbench = np.concatenate([np.array([1] * 5), advbench, np.array([1] * 5)])
gsm8k = np.concatenate([np.array([0] * 5), gsm8k, np.array([0] * 5)])
gsm8k = benchmark_to_loss_landscape(gsm8k)
truthfulqa = benchmark_to_loss_landscape(truthfulqa)
humaneval = np.concatenate([np.array([0] * 5), humaneval, np.array([0] * 5)])
humaneval = benchmark_to_loss_landscape(humaneval)


plt.plot(x, advbench, label="Safety")
plt.plot(x, gsm8k, label="Math")
plt.plot(x, truthfulqa, label="Basic")
plt.plot(x, humaneval, label="Coding")
plt.xlabel(r"$\alpha$")
plt.ylabel("Loss")
plt.legend()
os.makedirs("./landscape/aggregated/enlarge/", exist_ok=True)
plt.savefig(f"./landscape/aggregated/enlarge/{model_name}.pdf", bbox_inches="tight")
# plt.show()
plt.close()
