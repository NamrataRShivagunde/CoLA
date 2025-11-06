import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import List, Dict
from cola import ColaConfig, ColaForCausalLM


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA_MIN = -0.01
ALPHA_MAX = 0.01
ALPHA_INTERVAL = 0.002
BASE_DIR = "checkpoints/cola_60m-2025-11-03-16-14-50"
BASE_DIR = "../GaLore/checkpoints/llama_60m-2025-11-04-16-37-42"
CHECKPOINTS = [f"model_{i}" for i in range(1000, 11000, 1000)]


# =====================================================================
# Compute NLL Loss
# =====================================================================
def compute_nll_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    MAX_BATCHES = 5
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= MAX_BATCHES:
                break
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            else:
                input_ids, attention_mask = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


# =====================================================================
# Load Data
# =====================================================================
def get_dataloader(tokenizer):
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    MAX_EXAMPLES = 20000  # smaller subset for quick contour eval

    samples = []
    for i, ex in enumerate(dataset):
        if i >= MAX_EXAMPLES:
            break
        samples.append(ex)

    tokenized = tokenizer(
        [ex["text"] for ex in samples],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    torch_dataset = torch.utils.data.TensorDataset(
        tokenized["input_ids"], tokenized["attention_mask"]
    )
    return torch.utils.data.DataLoader(torch_dataset, batch_size=4, shuffle=False)


# =====================================================================
# 2D Loss Landscape
# =====================================================================
class LossLandscape2D:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.params_0 = [p.data.clone() for p in model.parameters()]
        # random directions
        self.dir_x = [torch.randn_like(p.data).to(p.device) for p in model.parameters()]
        self.dir_y = [torch.randn_like(p.data).to(p.device) for p in model.parameters()]
        # orthogonalize dir_y wrt dir_x
        dot = sum((dx * dy).sum() for dx, dy in zip(self.dir_x, self.dir_y))
        norm_x = sum((dx * dx).sum() for dx in self.dir_x)
        for dy, dx in zip(self.dir_y, self.dir_x):
            dy -= (dot / norm_x) * dx

    def synthesize_and_compute(self, alpha_min, alpha_max, alpha_interval):
        alphas = np.arange(alpha_min, alpha_max + alpha_interval / 2, alpha_interval)
        losses = np.zeros((len(alphas), len(alphas)))

        for i, ax in enumerate(alphas):
            for j, ay in enumerate(alphas):
                with torch.no_grad():
                    for p, p0, dx, dy in zip(self.model.parameters(), self.params_0, self.dir_x, self.dir_y):
                        p.data.copy_(p0 + ax * dx + ay * dy)
                loss = compute_nll_loss(self.model, self.dataloader)
                losses[i, j] = loss

        # restore weights
        with torch.no_grad():
            for p, p0 in zip(self.model.parameters(), self.params_0):
                p.data.copy_(p0)

        return alphas, losses


# =====================================================================
# Main
# =====================================================================
def main():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    dataloader = get_dataloader(tokenizer)

    all_results = {}

    for m in CHECKPOINTS:
        model_path = f"{BASE_DIR}/{m}"
        print(f"\nLoading Model: {model_path}")
        config = ColaConfig.from_pretrained(model_path)
        model = ColaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

        drawer = LossLandscape2D(model, dataloader)
        alphas, losses = drawer.synthesize_and_compute(ALPHA_MIN, ALPHA_MAX, ALPHA_INTERVAL)
        all_results[m] = (alphas, losses)

    # Plot contour grids
    cols = 3
    rows = int(np.ceil(len(all_results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), sharex=True, sharey=True)
    axes = axes.flatten()

    # Find global color range
    all_min, all_max = (
        min(losses.min() for _, losses in all_results.values()),
        max(losses.max() for _, losses in all_results.values())
    )

    for i, (name, (alphas, losses)) in enumerate(all_results.items()):
        ax = axes[i]
        X, Y = np.meshgrid(alphas * 1000, alphas * 1000)
        contour = ax.contourf(X, Y, losses, levels=20, cmap="viridis", vmin=all_min, vmax=all_max)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(r"$\alpha_x \times 10^{-3}$")
        ax.set_ylabel(r"$\alpha_y \times 10^{-3}$")
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("2D Loss Contours Across CoLA Checkpoints", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    cbar = fig.colorbar(contour, ax=axes, shrink=0.95)
    cbar.set_label("NLL Loss", rotation=270, labelpad=15)
    plt.savefig("loss_landscape_baseline_2d.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
