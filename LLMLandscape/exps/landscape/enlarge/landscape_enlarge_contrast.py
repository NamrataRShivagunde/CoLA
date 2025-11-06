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

x = np.arange(-0.015, 0.015, 1e-3)

baseline = np.load(f"./landscape/data/enlarge/mmlu-baseline-high.npy")
mmlu5e3 = np.load(f"./landscape/data/enlarge/mmlu-go=2e-4=5628-high.npy")
mmlu1e2 = np.load(f"./landscape/data/enlarge/mmlu-go=2e-4=001=5628-high.npy")
mmlu2e2 = np.load(f"./landscape/data/enlarge/mmlu-go=2e-4=002=5628-high.npy")

baseline = benchmark_to_loss_landscape(baseline)
mmlu5e3 = benchmark_to_loss_landscape(mmlu5e3)
mmlu1e2 = benchmark_to_loss_landscape(mmlu1e2)
mmlu2e2 = benchmark_to_loss_landscape(mmlu2e2)

# plt.xlim(-0.075, 0.075)
plt.plot(x, baseline, label="origin")
plt.plot(x, mmlu5e3, label="5e-3")
plt.plot(x, mmlu1e2, label="1e-2")
# plt.plot(x, mmlu2e2, label="2e-2")
plt.xlabel(r"$\alpha$")
plt.ylabel("Loss")
plt.legend()
os.makedirs("./landscape/aggregated/enlarge/", exist_ok=True)
plt.savefig(f"./landscape/aggregated/enlarge/contrast.pdf", bbox_inches="tight")
# plt.show()
plt.close()
