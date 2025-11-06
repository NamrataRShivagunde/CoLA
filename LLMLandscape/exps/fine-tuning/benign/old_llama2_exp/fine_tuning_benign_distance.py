from utils.model_info import model_distance
from models import Llama2
from data import get_adv_bench_behaviors_50
from matplotlib import pyplot as plt


ori_llama2 = Llama2()
l2s, l1s, linfs, avgs, l2_squares = [], [], [], [], []
advbench_50 = list(get_adv_bench_behaviors_50())
ckpts = [50, 100, 150, 200, 250, 300, 350, 375]
for ckpt in ckpts:
    llama2 = Llama2(f"./results/fine-tuning-attack/benign-2e-5/checkpoint-{ckpt}/").cuda()
    for prompt, _ in advbench_50[10:11]:
        llama2.generate(prompt, verbose=True)
    llama2.cpu()
    l2, l1, linf, avg, l2_square = model_distance(ori_llama2, llama2)
    l2s.append(l2)
    l1s.append(l1)
    linfs.append(linf)
    avgs.append(avg)
    l2_squares.append(l2_square)
    del llama2
    print("-" * 10)

plt.plot(ckpts, linfs, label="linf")
plt.plot(ckpts, avgs, label="avg")
plt.legend()
plt.savefig("./linf_avg.png")
plt.clf()
plt.plot(ckpts, l2s, label="l2")
plt.legend()
plt.savefig("./l2.png")
