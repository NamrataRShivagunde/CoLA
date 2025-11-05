import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from cola import ColaForCausalLM, ColaConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR_COLA = "checkpoints/cola_60m-2025-11-03-16-14-50"
BASE_DIR_BASELINE = "../GaLore/checkpoints/llama_60m-2025-11-04-16-37-42"
CHECKPOINT = "model_10000"  # choose one checkpoint to compare directly

ALPHA_MIN = -0.01
ALPHA_MAX = 0.01
ALPHA_INTERVAL = 0.0005


def get_dataloader(tokenizer):
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for i, ex in enumerate(dataset):
        if i >= 20000:
            break
        texts.append(ex["text"])

    tokens = tokenizer(texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    ds = torch.utils.data.TensorDataset(tokens["input_ids"], tokens["attention_mask"])
    return torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)


def compute_loss(model, dataloader):
    model.eval()
    total, count = 0, 0
    with torch.no_grad():
        for i, (input_ids, mask) in enumerate(dataloader):
            if i >= 50: break
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            loss = model(input_ids=input_ids, attention_mask=mask, labels=input_ids).loss
            total += loss.item()
            count += 1
    return total / count


def generate_shared_direction(model):
    direction = {}
    for name, p in model.named_parameters():
        d = torch.randn_like(p)
        d = d / (d.norm() + 1e-9) * (p.norm() + 1e-9)
        direction[name] = d
    return direction


def apply_direction(model, base_params, direction, alpha):
    with torch.no_grad():
        for (name, p), p0 in zip(model.named_parameters(), base_params.values()):
            p.copy_(p0 + alpha * direction[name])


def main():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    dataloader = get_dataloader(tokenizer)

    cola = ColaForCausalLM.from_pretrained(f"{BASE_DIR_COLA}/{CHECKPOINT}", torch_dtype=torch.bfloat16, device_map="auto")
    base = AutoModelForCausalLM.from_pretrained(f"{BASE_DIR_BASELINE}/{CHECKPOINT}", torch_dtype=torch.bfloat16, device_map="auto")

    cola_params0 = {n: p.detach().clone() for n, p in cola.named_parameters()}
    base_params0 = {n: p.detach().clone() for n, p in base.named_parameters()}

    direction = generate_shared_direction(base)

    alphas = np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_INTERVAL, ALPHA_INTERVAL)
    cola_losses, base_losses = [], []

    for alpha in alphas:
        apply_direction(cola, cola_params0, direction, alpha)
        cola_losses.append(compute_loss(cola, dataloader))

        apply_direction(base, base_params0, direction, alpha)
        base_losses.append(compute_loss(base, dataloader))

    # restore
    apply_direction(cola, cola_params0, direction, 0)
    apply_direction(base, base_params0, direction, 0)

    plt.figure(figsize=(7,5))
    plt.plot(alphas*1000, cola_losses, label="CoLA", linewidth=2)
    plt.plot(alphas*1000, base_losses, label="Baseline", linewidth=2, linestyle="--")
    plt.xlabel(r"$\alpha \times 10^{-3}$")
    plt.ylabel("NLL Loss")
    plt.title(f"Shared Direction Loss Landscape ({CHECKPOINT})")
    plt.legend()
    plt.grid(True)
    plt.savefig("shared_direction_landscape.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
