import torch
from torch.utils.data import Dataset
from models import BaseModel
from torch.utils.data import Subset
from datasets import load_dataset


__all__ = ["UltraFeedBackBest", "get_ultra_feedback_train_eval"]


class UltraFeedBackBest(Dataset):
    def __init__(self, model: BaseModel, dataset: Dataset):
        self.model = model
        self.data = dataset

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, index):
        # Note: We do not add EOT in this dataset class.
        # Because some dataset should not include EoT.
        # If you need EoT, please add EoT string to your data.
        # ***Note***: Please check whether the labels are correct whenever you change the model!!!
        now = self.data[index]
        q = now["instruction"]
        a = [(i["overall_score"], i["response"]) for i in now["completions"]]
        if len(a) == 0:
            return self.__getitem__(index - 1)
        max_score_tuple = max(a, key=lambda x: x[0])
        a = max_score_tuple[1]
        input_ids = self.model.get_prompt(q, "", a + self.model.tokenizer.eos_token)[0]
        input_ids = input_ids.cpu()
        labels = input_ids.clone()
        item = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }
        return item


def get_ultra_feedback_train_eval(model: BaseModel):
    dataset = load_dataset("openbmb/UltraFeedback")["train"]
    dataset = dataset.with_format("torch")
    train_dataset = Subset(dataset, list(range(60000)))
    eval_dataset = Subset(dataset, list(range(60000, 63967)))
    return UltraFeedBackBest(model, train_dataset), UltraFeedBackBest(model, eval_dataset)
