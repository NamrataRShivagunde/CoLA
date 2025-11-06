"""
The core principle of visualizing the loss landscape is to first select a direction \(\delta\),
and then visualize the two-dimensional function \(L(\theta_0 + \alpha \delta))\

Loss Landscape是一个高维函数，人类无法直接理解高维函数。因此可视化的原理是预定一个方向\(\delta\)，
可视化大模型在这个方向上的loss变化，因此是可视化的实际是二维函数\(L(\theta_0 + \alpha \delta))\
"""
import torch
import math
import numpy as np
from utils.plot import Landscape4Model

"""
The core code is class "Landscape4Model". Jump into the code for implementation details. 
We only present how to use this class here.

画Loss Landscape用的是"Landscape4Model"这个类。您可以跳转到相应代码看看具体实现。本文件只讲如何调用。
"""

from models import Qwen2
from tester import lm_eval_mmlu, test_vanilla_harmful_output_rate, lm_eval_gsm8k, lm_eval_humaneval
from data import get_adv_bench_behaviors_50

"""
Landscape4Model has two important arguments.
1. Models, which must be nn.Module. (Huggingface modules are all inherited from PretrainModel, 
which is also a subclass of nn.Module.
2. A function that takes the model (weights) as input, and return a real value in [0, 1]

Landscape4Model有两个重要参数
1. 模型。因为我们需要修改模型的weight才能画Loss Landscape，因此这里传参必须是nn.Module类型的，这样我们Landscape4Model这个类才能修改weight。
所有huggingface 模型都继承于PretrainModel。因此都是nn.Module的子类。
2. loss函数。即input一个模型，output一个[0, 1]区间的值作为Loss。
"""

# Model. Note, you don't necessarily encapsulate models like me.  This is just for convenience.
# 模型。您完全不需要把模型像我这样封装一层。只要满足上面参数要求都能做的通。我这样写只是个人的preference而已。
model = Qwen2().cuda()

# loss calculator. Input a model (with weight), return a loss
# loss函数。即input一个模型，output一个[0, 1]区间的值作为Loss。
loss_calculator = lm_eval_humaneval  # Humaneval benchmark for code
loss_calculator = lm_eval_mmlu  # MMLU benchmark for basic capacity
loss_calculator = lm_eval_gsm8k  # GSM8k Benchmark for math
loss_calculator = lambda m: test_vanilla_harmful_output_rate(m, get_adv_bench_behaviors_50())  # AdvBench for safety

"""
There are 3 types of Loss Landscape.
Most-case landscape: delta is standard Gaussian.
Worst-case landscape: delta is an adversarial direction.
SFT-landscape: Landscape along SFT.


Most-case landscape一定是**Standard Gaussian**。我们希望basin越大，在basin内部表达能力越强，越不容易遗忘。
如果加的是Standard Gaussian，各个方向有均匀概率取到，因此可视化的是“大多数方向”。也满足basin越大，表达能力越大，越不容易遗忘
（无论是从rademacher complexity的角度，还是本论文使用的RS技术，都是如此。

个人观点，不认为应该使用Normalized Gaussian。否则Basin大小毫无意义，无法得到表达能力或者certified bound。

"""

drawer = Landscape4Model(
    model,
    loss_calculator,
    direction="Gaussian",
    save_path="./landscape",
    # device=torch.device("cpu"), uncomment this line if you don't have enough memory.
    # Note, this does NOT mean the calculation is on CPU. It just offloads the parameters to CPU.
)

"""
Defining the range of alpha to make your plot beautiful.

为你的模型选择合适的landscape尺寸。不同模型landscape大小不同，我不可能预制一个统一尺寸。你可以选择最适合的尺寸。

Adjusting x_interval for better resolution.

你可以调整最小刻度x_interval，获得更高分辨率。当然也会导致更大的计算量。

results will be save to save_path+saving_name.
"""

coordinate_config = dict(x_min=-0.01, x_max=0.01, x_interval=0.5e-3, y_min=-5e-3, y_max=5e-3, y_interval=1e-3)
drawer.synthesize_coordinates(**coordinate_config)
result = drawer.draw(saving_name="qwen-advbench-most-case")


"""
Now you finished the part of most-case basin. We can also visualizing along worst-direction or SFT direction.

恭喜你已经完成了most-case landscape的绘制。我们还可以画worst-case landscape.

To find the worst_direction or other directions you want, you need a target model.

"""


worst_model = Qwen2("./results/benign-harmful-sft/worst/worst-5e-05-qwen/checkpoint-10/")

num_params = sum([i.numel() for i in model.parameters()])
l2_square = sum([((a - b) ** 2).sum().item() for a, b in zip(model.parameters(), worst_model.parameters())])
l2 = math.sqrt(l2_square)
norm_gaussian = math.sqrt(num_params)
revise_factor = norm_gaussian / l2  # normalize the delta to make it have same scale as previous.


def find_worst_direction(target_drawer: Landscape4Model):
    target_drawer.x_unit_vector = {}
    target_drawer.y_unit_vector = {}
    for (name, param), (_, worst_param) in zip(model.named_parameters(), worst_model.named_parameters()):
        target_drawer.x_unit_vector[name] = (worst_param.data - param.data) * revise_factor
        target_drawer.y_unit_vector[name] = torch.tensor(0)


find_worst_direction(drawer)
result = drawer.draw(saving_name="qwen-advbench-worst-case")

"""
You can also save the data.
你也可以保存各个点的loss，避免图画的不好看想重画时的重复计算。
"""
np.save(f"./landscape/data/worst/qwen-high.npy", result)
print(result)


