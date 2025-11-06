import torch
import os
from models import Llama2, Llama3, Qwen2
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from data import get_alpaca_original_52k
from torch.utils.data import Subset
from optimizers import GaussianAugmentedOptimizer
from torch.optim import AdamW


os.environ["WANDB_PROJECT"] = "alpaca-qwen"
lr = 5e-4
region = 0
exp_name = f"go-{lr}-region-{region}"


class GradientLoggingCallback(TrainerCallback):
    def __init__(self, log_frequency=10):
        self.log_frequency = log_frequency

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        optimizer.before_training()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        optimizer.after_training()

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        optimizer.recover_model()  # each substep use new gaussian
        optimizer.perturb_model()


class SaveOriTrainer(Trainer):
    def _save_checkpoint(self, *args, **kwargs):
        print("saving checkpoints.")
        optimizer.after_training()
        super(SaveOriTrainer, self)._save_checkpoint(*args, **kwargs)
        optimizer.before_training()
        print("checkpoints saved.")

    def create_optimizer(self):
        print("subclass of Trainer, create optimizer function was called")
        self.optimizer = optimizer
        return optimizer


training_args = TrainingArguments(
    output_dir=f"./results/sft-alpaca-qwen/{exp_name}",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"./logs/sft-alpaca-qwen/{exp_name}",
    logging_steps=1,
    learning_rate=lr,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,  # Note：保证batchsize是128
    bf16=True,
    save_only_model=True,  # 不然optimizer会占巨多
    lr_scheduler_type="constant",
    run_name=exp_name,
)


llama2 = Qwen2("Qwen/Qwen2.5-0.5B-Instruct")
llama2.cuda()
model = llama2.model
model.requires_grad_(True)

dataset = get_alpaca_original_52k(llama2)

train_dataset = Subset(dataset, list(range(50000)))
eval_dataset = Subset(dataset, list(range(50000, 52002)))


optimizer = GaussianAugmentedOptimizer(
    AdamW(model.parameters(), lr=lr, weight_decay=0),
    model,
    gaussian_std=region,
    backup_device=torch.device("cpu"),
)

trainer = SaveOriTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(llama2.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    callbacks=[GradientLoggingCallback()],
    optimizers=(optimizer, None),
)

print("num of trainable parameters: ", trainer.get_num_trainable_parameters())


trainer.train()
