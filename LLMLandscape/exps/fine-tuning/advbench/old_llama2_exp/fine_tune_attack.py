import torch
import os
from models import Vicuna15, Llama2, Llama3
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from data.general import GeneralQADataset
from data import get_adv_bench_behaviors_50


"""
8x3090:
deepspeed --include='localhost:0,1,2,3,4,5,6,7' fine_tune_attack.py

2xA100:
deepspeed --include='localhost:0' fine_tune_attack.py
"""
os.environ["WANDB_PROJECT"] = "SFT-attack"

exp_name = "default-4e-5-bf16"
llama2 = Llama2(dtype=torch.float32)
# llama2 = Llama3("meta-llama/Llama-3.1-8B-Instruct")
model = llama2.model
model.requires_grad_(True)

model.gradient_checkpointing_enable()
training_args = TrainingArguments(
    output_dir=f"./results/fine-tuning-attack/{exp_name}",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"./logs/fine-tuning-attack/{exp_name}",
    learning_rate=4e-5,
    num_train_epochs=5,
    save_total_limit=5,
    # weight_decay=0.01,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    bf16=True,
    deepspeed="./exps/fine-tuning/configs/sftattack-zero1.json",
    save_only_model=True,  # 不然optimizer会占巨多
    lr_scheduler_type="constant",
)

advbench_50 = list(get_adv_bench_behaviors_50())
train_dataset = GeneralQADataset(llama2, advbench_50[:16])
eval_dataset = GeneralQADataset(llama2, advbench_50[16:])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(llama2.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)

print("num of trainable parameters: ", trainer.get_num_trainable_parameters())

trainer.train()
