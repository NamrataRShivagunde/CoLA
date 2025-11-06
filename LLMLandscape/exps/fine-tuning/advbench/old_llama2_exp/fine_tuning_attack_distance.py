from utils.model_info import model_distance
from models import Llama2
from data import get_adv_bench_behaviors_50
from matplotlib import pyplot as plt


ori_llama2 = Llama2().cuda()
l2s, l1s, linfs, avgs, l2_squares = [], [], [], [], []
advbench_50 = list(get_adv_bench_behaviors_50())
ckpts = list(range(1, 6))
for ckpt in ckpts:
    llama2 = Llama2(f"./results/fine-tuning-attack/default-4e-5/checkpoint-{ckpt}/").cuda()
    for prompt, _ in advbench_50[10:12]:
        llama2.generate(prompt, verbose=True)
    l2, l1, linf, avg, l2_square = model_distance(ori_llama2, llama2)
    l2s.append(l2)
    l1s.append(l1)
    linfs.append(linf)
    avgs.append(avg)
    l2_squares.append(l2_square)
    del llama2
    print("()" * 100)
# plt.plot(ckpts, l2s, label="l2")
# plt.plot(ckpts, l1s, label="l1")
plt.plot(ckpts, linfs, label="linf")
plt.plot(ckpts, avgs, label="avg")
# plt.plot(ckpts, l2_squares, label="l2_square")
plt.legend()
plt.savefig("./linf_avg.png")
plt.clf()
plt.plot(ckpts, l2s, label="l2")
plt.legend()
plt.savefig("./l2.png")
