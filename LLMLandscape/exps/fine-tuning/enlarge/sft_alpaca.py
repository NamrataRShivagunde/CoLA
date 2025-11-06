import torch
from models import Qwen2, Llama2, Llama3
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from data import get_alpaca_original_52k
from torch.utils.data import Subset
import os

os.environ["WANDB_PROJECT"] = "alpaca-useless"
exp_name = "real-baseline"


class GradientLoggingCallback(TrainerCallback):
    def __init__(self, log_frequency=10):
        self.log_frequency = log_frequency  # 每隔多少步记录一次


training_args = TrainingArguments(
    output_dir=f"./results/sft-alpaca-useless/{exp_name}",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"./logs/sft-alpaca-useless/{exp_name}",
    logging_steps=10,
    learning_rate=2e-5,
    num_train_epochs=100,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,  # Note：保证batchsize是128
    bf16=True,
    save_only_model=True,  # 不然optimizer会占巨多
    lr_scheduler_type="constant",
    run_name=exp_name,
)


llama2 = Qwen2("Qwen/Qwen2.5-0.5B-Instruct")
model = llama2.model
model.requires_grad_(True)

dataset = get_alpaca_original_52k(llama2)

train_dataset = Subset(dataset, list(range(50000)))
eval_dataset = Subset(dataset, list(range(50000, 52002)))


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(llama2.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    callbacks=[GradientLoggingCallback()],
)

print("num of trainable parameters: ", trainer.get_num_trainable_parameters())

trainer.train()
