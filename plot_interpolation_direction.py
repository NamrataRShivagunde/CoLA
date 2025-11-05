import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from cola import ColaForCausalLM
from datasets import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR_COLA = "checkpoints/cola_60m-2025-11-03-16-14-50"
BASE_DIR_BASELINE = "../GaLore/checkpoints/llama_60m-2025-11-04-16-37-42"
CHECKPOINT = "model_10000"  # pick the checkpoint to compare


def compute_loss(model, dataloader):
    model.eval()
    total, count = 0, 0
    with torch.no_grad():
        for i, (input_ids, mask) in enumerate(dataloader):
            if i >= 50:
                break
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            loss = model(input_ids=input_ids, attention_mask=mask, labels=input_ids).loss
            total += loss.item()
            count += 1
    return total / count


def get_dataloader(tokenizer):
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = [ex["text"] for _, ex in zip(range(20000), dataset)]
    tokens = tokenizer(texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    ds = torch.utils.data.TensorDataset(tokens["input_ids"], tokens["attention_mask"])
    return torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)


def main():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    dataloader = get_dataloader(tokenizer)

    # ✅ Load models on a SINGLE device — important!
    cola = ColaForCausalLM.from_pretrained(f"{BASE_DIR_COLA}/{CHECKPOINT}", torch_dtype=torch.bfloat16).to(DEVICE)
    base = AutoModelForCausalLM.from_pretrained(f"{BASE_DIR_BASELINE}/{CHECKPOINT}", torch_dtype=torch.bfloat16).to(DEVICE)

    # ✅ Convert named params to dicts
    cola_params = dict(cola.named_parameters())
    base_params = dict(base.named_parameters())

    # ✅ Only interpolate over shared parameters
    shared_keys = set(cola_params.keys()) & set(base_params.keys())
    print(f"Interpolating over {len(shared_keys)} shared parameters")

    # ✅ Store copies of original weights
    base_init = {name: p.detach().clone() for name, p in base_params.items()}
    cola_init = {name: p.detach().clone() for name, p in cola_params.items()}

    # α ∈ [-0.2, 1.2] → beyond endpoints to see basin shape
    alphas = np.linspace(-0.2, 1.2, 25)
    losses = []

    with torch.no_grad():
        for alpha in alphas:
            for name, p in base.named_parameters():
                if name in shared_keys:
                    p.copy_((1 - alpha) * base_init[name] + alpha * cola_init[name])
                else:
                    p.copy_(base_init[name])  # restore untouched params
            losses.append(compute_loss(base, dataloader))

    # ✅ Plot
    plt.figure(figsize=(7,5))
    plt.plot(alphas, losses, linewidth=2)
    plt.axvline(0, color='gray', linestyle='--')
    plt.axvline(1, color='gray', linestyle='--')
    plt.text(0, min(losses), "Baseline", ha="center", va="bottom")
    plt.text(1, min(losses), "CoLA", ha="center", va="bottom")
    plt.xlabel("Interpolation α  (0 = Baseline, 1 = CoLA)")
    plt.ylabel("NLL Loss")
    plt.title(f"Interpolation Loss Curve ({CHECKPOINT})")
    plt.grid(True)
    plt.savefig("interpolation_curve_fixed.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
