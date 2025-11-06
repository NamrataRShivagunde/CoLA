import torch
from models import Vicuna15, Llama2, Llama3
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from data import get_alpaca_no_safety_50k
from torch.utils.data import Subset
import os


"""
On 3090, may require
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
"""

os.environ["WANDB_PROJECT"] = "RCWA"


exp_name = "benign-2e-5"

training_args = TrainingArguments(
    output_dir=f"./results/fine-tuning-attack/{exp_name}",
    evaluation_strategy="steps",
    logging_dir=f"./logs/fine-tuning-attack/{exp_name}",
    eval_steps=50,
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-5,
    num_train_epochs=1,
    # weight_decay=0.01,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # TODO：保证batchsize是128
    fp16=True,  # True的话前18个step一定没梯度。可能是nan了？
    deepspeed="./exps/fine-tuning/configs/benign-zero3.json",  # TODO:做实验的时候改一下，这里bs不对，也不是zero3，新建一个
    save_only_model=True,  # 不然optimizer会占巨多
    lr_scheduler_type="constant",
)


llama2 = Llama2(dtype=torch.float16, device=torch.device("cpu"))
# llama2 = Llama3("meta-llama/Llama-3.1-8B-Instruct")
model = llama2.model
model.requires_grad_(True)

dataset = get_alpaca_no_safety_50k(llama2)
train_dataset = Subset(dataset, list(range(48000)))
eval_dataset = Subset(dataset, list(range(48000, 49971)))

lengths = [len(item["input_ids"]) for item in train_dataset]
print("Max length:", max(lengths), "Avg length:", sum(lengths) / len(lengths))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(llama2.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)

print("num of trainable parameters: ", trainer.get_num_trainable_parameters())

trainer.train()
