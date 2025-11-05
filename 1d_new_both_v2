import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from typing import List, Dict
from cola import ColaConfig, ColaForCausalLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sweep settings
ALPHA_MIN = -0.010
ALPHA_MAX = 0.010
ALPHA_INTERVAL = 0.0005

BETA_MIN = -0.010
BETA_MAX = 0.010
BETA_INTERVAL = 0.002   # Slightly coarser for 2D

# Model checkpoint directories
BASE_DIR_COLA = "checkpoints/cola_60m-2025-11-03-16-14-50"
BASE_DIR_BASELINE = "../GaLore/checkpoints/llama_60m-2025-11-04-16-37-42"
CHECKPOINTS = [f"model_{i}" for i in range(1000, 11000, 1000)]


# ============================================================
# Compute NLL Loss
# ============================================================
def compute_nll_loss(model, dataloader) -> float:
    model.eval()
    total_loss, total_batches = 0.0, 0
    MAX_BATCHES = 80
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= MAX_BATCHES:
                break
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(model_device)
                attention_mask = batch["attention_mask"].to(model_device)
            else:
                input_ids, attention_mask = [x.to(model_device) for x in batch]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


# ============================================================
# Load Data
# ============================================================
def get_dataloader(tokenizer):
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    MAX_EXAMPLES = 40000  

    texts = []
    for i, ex in enumerate(dataset):
        if i >= MAX_EXAMPLES:
            break
        texts.append(ex["text"])

    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    ds = torch.utils.data.TensorDataset(tokenized["input_ids"], tokenized["attention_mask"])
    return torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)


# ============================================================
# Loss Landscape Drawer (Updated for Normalize + 2D)
# ============================================================
class LossLandscapeDrawer:
    def __init__(self, model, dataloader, mode="1D"):
        self.model = model
        self.dataloader = dataloader
        self.mode = mode
        self.original_params = {name: p.data.clone() for name, p in model.named_parameters()}

        self.x_direction = {}
        self.y_direction = {}

        for name, p in model.named_parameters():
            # X direction
            d = torch.randn_like(p.data)
            d = d / (d.norm() + 1e-9) * (p.data.norm() + 1e-9)
            self.x_direction[name] = d.to(p.device)

            # Y direction only for 2D
            if mode == "2D":
                d2 = torch.randn_like(p.data)
                d2 = d2 / (d2.norm() + 1e-9) * (p.data.norm() + 1e-9)
                self.y_direction[name] = d2.to(p.device)

    def set_params(self, alpha, beta=0):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                update = alpha * self.x_direction[name]
                if self.mode == "2D":
                    update += beta * self.y_direction[name]
                p.data.copy_(self.original_params[name] + update)

    def restore(self):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.data.copy_(self.original_params[name])

    def compute_1d(self):
        alphas = np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_INTERVAL / 2, ALPHA_INTERVAL)
        losses = []
        for alpha in alphas:
            self.set_params(alpha)
            losses.append(compute_nll_loss(self.model, self.dataloader))
        self.restore()
        return alphas, np.array(losses)

    def compute_2d(self):
        alphas = np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_INTERVAL / 2, ALPHA_INTERVAL)
        betas = np.arange(BETA_MIN, BETA_MAX + BETA_INTERVAL / 2, BETA_INTERVAL)
        Z = np.zeros((len(alphas), len(betas)))
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                self.set_params(a, b)
                Z[i, j] = compute_nll_loss(self.model, self.dataloader)
        self.restore()
        return alphas, betas, Z


# ============================================================
# MAIN
# ============================================================
def main(mode="1D"):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    dataloader = get_dataloader(tokenizer)

    results_cola = {}
    results_base = {}

    for ckpt in CHECKPOINTS:
        print(f"\n--- Processing {ckpt} ---")

        # Load COLA
        model_cola = ColaForCausalLM.from_pretrained(f"{BASE_DIR_COLA}/{ckpt}", torch_dtype=torch.bfloat16, device_map="auto")
        drawer_cola = LossLandscapeDrawer(model_cola, dataloader, mode)
        results_cola[ckpt] = drawer_cola.compute_1d() if mode=="1D" else drawer_cola.compute_2d()
        del model_cola; torch.cuda.empty_cache()

        # Load Baseline
        model_base = AutoModelForCausalLM.from_pretrained(f"{BASE_DIR_BASELINE}/{ckpt}", torch_dtype=torch.bfloat16, device_map="auto")
        drawer_base = LossLandscapeDrawer(model_base, dataloader, mode)
        results_base[ckpt] = drawer_base.compute_1d() if mode=="1D" else drawer_base.compute_2d()
        del model_base; torch.cuda.empty_cache()

    # ---------------- PLOT ----------------
    if mode == "1D":
        plt.figure(figsize=(12, 6))
        for ckpt in CHECKPOINTS:
            a1, l1 = results_cola[ckpt]
            a2, l2 = results_base[ckpt]
            plt.plot(a1 * 1000, l1, label=f"{ckpt} COLA")
            plt.plot(a2 * 1000, l2, linestyle='--', label=f"{ckpt} BASELINE")
        plt.xlabel(r"$\alpha \times 10^{-3}$")
        plt.ylabel("NLL Loss")
        plt.legend()
        plt.title("1D Loss Landscape Comparison")
        plt.grid(True)
        plt.show()

    else:
        ckpt = CHECKPOINTS[-1]
        alphas, betas, Z = results_cola[ckpt]
        A, B = np.meshgrid(alphas*1000, betas*1000, indexing="ij")
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(A, B, Z, cmap="viridis")
        ax.set_xlabel(r"$\alpha \times 10^{-3}$")
        ax.set_ylabel(r"$\beta \times 10^{-3}$")
        ax.set_zlabel("NLL Loss")
        ax.set_title(f"2D Loss Landscape: {ckpt} (COLA)")
        plt.show()


if __name__ == "__main__":
    # mode="1D" or mode="2D"
    main(mode="1D")
