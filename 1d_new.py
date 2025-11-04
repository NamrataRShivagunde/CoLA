import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import List, Dict
from cola import ColaConfig, ColaForCausalLM

# Configuration
CUDA_VISIBLE_DEVICES = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALPHA_MIN = -0.010
ALPHA_MAX = 0.010
ALPHA_INTERVAL = 0.0005

BASE_DIR = "checkpoints/cola_60m-2025-11-03-16-14-50"
CHECKPOINTS = [f"model_{i}" for i in range(1000, 11000, 1000)]


# =====================================================================
# Compute NLL Loss
# =====================================================================
def compute_nll_loss(model: AutoModelForCausalLM, dataloader: torch.utils.data.DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    MAX_BATCHES = 5
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= MAX_BATCHES:
                break
            # input_ids = batch["input_ids"].to(model_device)
            # attention_mask = batch["attention_mask"].to(model_device)
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(model_device)
                attention_mask = batch["attention_mask"].to(model_device)
            else:
                input_ids, attention_mask = [x.to(model_device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


# =====================================================================
# Load Data
# =====================================================================
def get_dataloader(tokenizer: AutoTokenizer):
    """
    Loads ~1% of the C4 validation split in streaming mode.
    This avoids downloading the entire dataset — only streams what is used.
    """
    print("Loading C4 validation split (1%) in streaming mode...")

    # 1️⃣ Stream the dataset — nothing big gets downloaded upfront
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    # 2️⃣ Compute how many samples ~1% is (C4 val ~364M examples total → 1% ≈ 3.6M)
    # But we don’t need *that* many for a loss landscape — we’ll just take first N streamed examples.
    MAX_EXAMPLES = 10000  # adjust this smaller/larger to control size
    print(f"Sampling first {MAX_EXAMPLES} examples from streaming dataset...")

    # 3️⃣ Take the first N examples from the stream
    streamed_samples = []
    for i, example in enumerate(dataset):
        if i >= MAX_EXAMPLES:
            break
        streamed_samples.append(example)

    # 4️⃣ Tokenize them
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized = tokenizer(
        [ex["text"] for ex in streamed_samples],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    # 5️⃣ Build a simple PyTorch DataLoader
    dataset_torch = torch.utils.data.TensorDataset(
        tokenized["input_ids"],
        tokenized["attention_mask"]
    )
    dataloader = torch.utils.data.DataLoader(dataset_torch, batch_size=4, shuffle=False)

    print(f"DataLoader ready with {len(dataset_torch)} examples.")
    return dataloader



# =====================================================================
# Loss Landscape Drawer
# =====================================================================
class LossLandscapeDrawer:
    def __init__(self, model: AutoModelForCausalLM, dataloader: torch.utils.data.DataLoader):
        self.model = model
        self.dataloader = dataloader
        self.original_params = [p.data.clone() for p in model.parameters()]
        self.direction = [torch.randn_like(p.data).to(p.device) for p in model.parameters()]

    def synthesize_and_compute(self, x_min: float, x_max: float, x_interval: float) -> Dict[str, np.ndarray]:
        alphas = np.arange(x_min, x_max + x_interval / 2, x_interval)
        losses = []

        for i, alpha in enumerate(alphas):
            print(f"  α={alpha:.5f} ({i+1}/{len(alphas)})", end="\r")
            with torch.no_grad():
                for p, p0, d in zip(self.model.parameters(), self.original_params, self.direction):
                    p.data.copy_(p0 + alpha * d)
            loss = compute_nll_loss(self.model, self.dataloader)
            losses.append(loss)

        # Restore parameters
        with torch.no_grad():
            for p, p0 in zip(self.model.parameters(), self.original_params):
                p.data.copy_(p0)

        return {"alphas": alphas, "loss_values": np.array(losses)}


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

        drawer = LossLandscapeDrawer(model, dataloader)
        result = drawer.synthesize_and_compute(ALPHA_MIN, ALPHA_MAX, ALPHA_INTERVAL)
        all_results[m] = result

    # -----------------------------------------------------------------
    # Combined Plot with Subplots
    # -----------------------------------------------------------------
    num_models = len(all_results)
    cols = 3
    rows = int(np.ceil(num_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (name, res) in enumerate(all_results.items()):
        ax = axes[i]
        alphas, losses = res["alphas"], res["loss_values"]
        ax.plot(alphas * 1000, losses, lw=2)
        ax.set_title(name, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        if i % cols == 0:
            ax.set_ylabel("NLL Loss")
        if i >= (rows - 1) * cols:
            ax.set_xlabel(r"$\alpha \ (\times 10^{-3})$")

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("1D Loss Landscapes Across CoLA Checkpoints", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("loss_landscape_cola_all.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
