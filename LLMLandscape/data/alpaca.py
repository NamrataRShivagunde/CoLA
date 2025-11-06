from .general import GeneralQADataset
from models import BaseModel
from datasets import load_from_disk, load_dataset


__all__ = ["get_alpaca_no_safety_50k", "get_alpaca_original_52k"]


def get_alpaca_no_safety_50k(
    model: BaseModel,
    path: str = "./resources/safety-tuned-llama-data/training/alpaca-no-safe-50k",
):
    loaded_ds = load_from_disk(path)
    data = [(item["instruction"], item["output"]) for item in loaded_ds["train"]]
    dataset = GeneralQADataset(model, data)
    return dataset


def get_alpaca_original_52k(model: BaseModel, path: str = "tatsu-lab/alpaca"):
    loaded_ds = load_dataset(path)
    data = [(item["instruction"], item["output"]) for item in loaded_ds["train"]]
    dataset = GeneralQADataset(model, data)
    return dataset
