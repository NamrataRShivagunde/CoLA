import math
import torch
import os
import datetime
import torch.distributed as dist

"""
deepspeed --include localhost:1,2 fine_tune_benign.py
"""

dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))

os.environ["WANDB_PROJECT"] = "benign-harmful-sft"
os.environ["NCCL_WATCHDOG_TIMEOUT"] = "7200000"  # 7200 秒（2 小时）
os.environ["NCCL_TIMEOUT"] = "7200000"  # 兼容旧版本
os.environ["TORCH_NCCL_TIMEOUT"] = "7200"  # 单位：秒
os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1000000"  # 启用 FlightRecorder
from models import Qwen2, Llama31, Mistral
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from data import get_alpaca_no_safety_50k, get_alpaca_original_52k, get_adv_bench_behaviors_50
from torch.utils.data import Subset
from tester import lm_eval_humaneval, lm_eval_truthfulqa, lm_eval_gsm8k, test_vanilla_harmful_output_rate
from utils import model_distance


class BenchmarkEvalCallback(TrainerCallback):
    def __init__(self, log_frequency=10):
        self.log_frequency = log_frequency

    # def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # if state.is_world_process_zero:
        #     model.generate = model.generate_by_input_ids
        #     human_eval_score = lm_eval_humaneval(model)
        #     truthfulqa_score = lm_eval_truthfulqa(model)
        #     gsm8k_score = lm_eval_gsm8k(model)
        #     model.generate = ori_generate_function
        #     advbench_score = test_vanilla_harmful_output_rate(model, get_adv_bench_behaviors_50())
        #     l2 = model_distance(model, ori_model)[0]
        #     landscape_d = l2 / math.sqrt(num_params)
        #     trainer.log(
        #         {
        #             "HumanEval": human_eval_score,
        #             "TruthfulQA": truthfulqa_score,
        #             "GSM8k": gsm8k_score,
        #             "AdvBench": advbench_score,
        #             "l2": l2,
        #             "landscape_d": landscape_d,
        #         }
        #     )
        # torch.distributed.barrier()


model_name = "qwen"
assert model_name in ["qwen", "llama", "mistral"]
if model_name == "qwen":
    model = Qwen2("Qwen/Qwen2.5-7B-Instruct")
    ori_model = Qwen2("Qwen/Qwen2.5-7B-Instruct")
elif model_name == "llama":
    model = Llama31()
    ori_model = Llama31()
else:
    model = Mistral()
    ori_model = Mistral()
model.cuda().requires_grad_(True)
ori_generate_function = model.generate


learning_rate = 5e-4
exp_name = f"benign-{learning_rate}-{model_name}"

training_args = TrainingArguments(
    output_dir=f"./results/benign-harmful-sft/benign/{exp_name}",
    eval_strategy="steps",
    logging_dir=f"./logs/benign-harmful-sft/benign/{exp_name}",
    eval_steps=50,
    save_steps=50,
    logging_steps=10,
    learning_rate=learning_rate,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,  # TODO：保证batchsize是128
    bf16=True,
    deepspeed="./exps/fine-tuning/configs/sft-alpaca.json",
    save_only_model=True,
    lr_scheduler_type="constant",
    run_name=exp_name,
)


# dataset = get_alpaca_no_safety_50k(model)
# train_dataset = Subset(dataset, list(range(48000)))
# eval_dataset = Subset(dataset, list(range(48000, 49971)))
dataset = get_alpaca_original_52k(model)
train_dataset = Subset(dataset, list(range(50000)))
eval_dataset = Subset(dataset, list(range(50000, 52002)))

lengths = [len(item["input_ids"]) for item in train_dataset]
print("Max length:", max(lengths), "Avg length:", sum(lengths) / len(lengths))

trainer = Trainer(
    model=model.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(model.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    callbacks=[BenchmarkEvalCallback()],
)

num_params = trainer.get_num_trainable_parameters()
print("num of trainable parameters: ", num_params)

trainer.train()
