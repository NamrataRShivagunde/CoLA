import torch
from torch.utils.data import Dataset
from models import BaseModel
from typing import Iterable, Tuple, List, Any


__all__ = ["GeneralQADataset"]


class GeneralQADataset(Dataset):
    def __init__(self, model: BaseModel, data: List[Tuple[str, Any]]):
        self.model = model
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Note: We do not add EOT in this dataset class.
        # Because some dataset should not include EoT.
        # If you need EoT, please add EoT string to your data.
        # ***Note***: Please check whether the labels are correct whenever you change the model!!!
        q, a = self.data[index]
        input_ids, _, _, _ = self.model.get_prompt(q, "", a + self.model.tokenizer.eos_token)
        input_ids = input_ids.cpu()
        labels = input_ids.clone()
        # labels = torch.zeros_like(input_ids) - 100
        # labels[target_slice] = input_ids[target_slice]
        item = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }
        return item
