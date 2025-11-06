import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from cola import ColaForCausalLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR_COLA = "checkpoints/cola_60m-2025-11-03-16-14-50"
BASE_DIR_BASELINE = "../GaLore/checkpoints/llama_60m-2025-11-04-16-37-42"
CHECKPOINTS = [f"model_{i}" for i in range(1000, 11000, 1000)]

ALPHA_MIN = -0.01
ALPHA_MAX = 0.01
ALPHA_INTERVAL = 0.0005


def get_dataloader(tokenizer):
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = [ex["text"] for _, ex in zip(range(20000), dataset)]
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


def generate_shared_direction(model_cola, model_base):
    direction = {}
    cola_params = dict(model_cola.named_parameters())
    base_params = dict(model_base.named_parameters())

    shared_keys = set(cola_params.keys()) & set(base_params.keys())

    for name in shared_keys:
        p = base_params[name]
        d = torch.randn_like(p, device=p.device)
        d = d / (d.norm() + 1e-9) * (p.norm() + 1e-9)
        direction[name] = d
    return direction


def apply_direction(model, params0, direction, alpha):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in direction:
                p.copy_(params0[name] + alpha * direction[name])
            else:
                p.copy_(params0[name])


def main():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    dataloader = get_dataloader(tokenizer)

    # === Step 1: generate shared direction from final checkpoint ===
    base_ref = AutoModelForCausalLM.from_pretrained(f"{BASE_DIR_BASELINE}/model_10000", torch_dtype=torch.bfloat16).to(DEVICE)
    cola_ref = ColaForCausalLM.from_pretrained(f"{BASE_DIR_COLA}/model_10000", torch_dtype=torch.bfloat16).to(DEVICE)
    direction = generate_shared_direction(cola_ref, base_ref)
    del base_ref, cola_ref
    torch.cuda.empty_cache()

    # === Step 2: sweep checkpoints ===
    results = {}

    alphas = np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_INTERVAL, ALPHA_INTERVAL)

    for ckpt in CHECKPOINTS:
        print(f"\nProcessing {ckpt} ...")

        cola = ColaForCausalLM.from_pretrained(f"{BASE_DIR_COLA}/{ckpt}", torch_dtype=torch.bfloat16).to(DEVICE)
        base = AutoModelForCausalLM.from_pretrained(f"{BASE_DIR_BASELINE}/{ckpt}", torch_dtype=torch.bfloat16).to(DEVICE)

        cola_params0 = {n: p.detach().clone() for n, p in cola.named_parameters()}
        base_params0 = {n: p.detach().clone() for n, p in base.named_parameters()}

        cola_losses, base_losses = [], []

        for alpha in alphas:
            apply_direction(cola, cola_params0, direction, alpha)
            cola_losses.append(compute_loss(cola, dataloader))

            apply_direction(base, base_params0, direction, alpha)
            base_losses.append(compute_loss(base, dataloader))

        # restore
        apply_direction(cola, cola_params0, direction, 0)
        apply_direction(base, base_params0, direction, 0)

        results[ckpt] = (cola_losses, base_losses)

        del cola, base
        torch.cuda.empty_cache()

    # === Step 3: plotting grid ===
    rows = 2
    cols = (len(CHECKPOINTS) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), sharey=True, sharex=True)
    axes = axes.flatten()

    for i, ckpt in enumerate(CHECKPOINTS):
        cola_losses, base_losses = results[ckpt]
        ax = axes[i]
        ax.plot(alphas * 1000, cola_losses, label="CoLA", linewidth=2)
        ax.plot(alphas * 1000, base_losses, label="Baseline", linewidth=2, linestyle="--")
        ax.set_title(ckpt)
        ax.grid(True)

    axes[0].legend()
    fig.suptitle("Shared-Direction Loss Landscape Evolution", fontsize=18)
    plt.xlabel(r"$\alpha \times 10^{-3}$")
    plt.ylabel("NLL Loss")
    plt.tight_layout()
    plt.savefig("shared_direction_landscape_evolution.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
