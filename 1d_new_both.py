import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from typing import List, Dict
from cola import ColaConfig, ColaForCausalLM # Assuming you have a custom cola module

# Configuration
# Note: DEVICE and CUDA_VISIBLE_DEVICES are often set outside the script
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALPHA_MIN = -0.010
ALPHA_MAX = 0.010
ALPHA_INTERVAL = 0.0005

# --- New BASE_DIR Definitions ---
BASE_DIR_COLA = "checkpoints/cola_60m-2025-11-03-16-14-50"
BASE_DIR_BASELINE = "../GaLore/checkpoints/llama_60m-2025-11-04-16-37-42"
CHECKPOINTS = [f"model_{i}" for i in range(10000, 11000, 1000)]
# --------------------------------

# =====================================================================
# Compute NLL Loss (No Change Needed)
# =====================================================================
# def compute_nll_loss(model: AutoModelForCausalLM, dataloader: torch.utils.data.DataLoader) -> float:
#     model.eval()
#     total_loss = 0.0
#     total_batches = 0
#     MAX_BATCHES = 100
#     model_device = next(model.parameters()).device

#     with torch.no_grad():
#         for i, batch in enumerate(dataloader):
#             if i >= MAX_BATCHES:
#                 break
#             if isinstance(batch, dict):
#                 input_ids = batch["input_ids"].to(model_device)
#                 attention_mask = batch["attention_mask"].to(model_device)
#             else:
#                 input_ids, attention_mask = [x.to(model_device) for x in batch]
#             # Use labels=input_ids for standard Causal Language Modeling loss
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
#             total_loss += outputs.loss.item()
#             total_batches += 1

#     return total_loss / max(total_batches, 1)

def compute_nll_loss(model, dataloader, pad_idx, target_tokens=1_000_000):
    """
    Compute token-normalized NLL that matches training evaluation.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for (input_ids, attention_mask) in dataloader:
            # Stop after reaching same token count as baseline eval
            if total_tokens >= target_tokens:
                break

            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)

            # Mask out padding
            labels = input_ids.clone()
            labels[labels == pad_idx] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            batch_loss = outputs.loss.item()  # mean loss over non-masked tokens
            batch_tokens = (labels != -100).sum().item()

            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens

    return total_loss / max(total_tokens, 1)



# =====================================================================
# Load Data (No Change Needed)
# =====================================================================
def get_dataloader(tokenizer: AutoTokenizer):
    """
    Loads ~1% of the C4 validation split in streaming mode.
    """
    print("Loading C4 validation split (1%) in streaming mode...")
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    MAX_EXAMPLES = 100000 
    print(f"Sampling first {MAX_EXAMPLES} examples from streaming dataset...")

    streamed_samples = []
    for i, example in enumerate(dataset):
        if i >= MAX_EXAMPLES:
            break
        streamed_samples.append(example)

    tokenized = tokenizer(
        [ex["text"] for ex in streamed_samples],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    dataset_torch = torch.utils.data.TensorDataset(
        tokenized["input_ids"],
        tokenized["attention_mask"]
    )
    dataloader = torch.utils.data.DataLoader(dataset_torch, batch_size=4, shuffle=False)

    print(f"DataLoader ready with {len(dataset_torch)} examples.")
    return dataloader


# =====================================================================
# Loss Landscape Drawer (No Change Needed)
# =====================================================================
class LossLandscapeDrawer:
    def __init__(self, model: AutoModelForCausalLM, dataloader: torch.utils.data.DataLoader):
        self.model = model
        self.dataloader = dataloader
        self.original_params = [p.data.clone() for p in model.parameters()]
        # Use a consistent direction for all checkpoints
        # For fair comparison, we could normalize or use the same random seed, 
        # but sticking to the original implementation which uses a random direction per model.
        self.direction = [torch.randn_like(p.data).to(p.device) for p in model.parameters()] 


    def synthesize_and_compute(self, x_min: float, x_max: float, x_interval: float, pad_idx) -> Dict[str, np.ndarray]:
        alphas = np.arange(x_min, x_max + x_interval / 2, x_interval)
        losses = []

        for i, alpha in enumerate(alphas):
            print(f"  α={alpha:.5f} ({i+1}/{len(alphas)})", end="\r")
            with torch.no_grad():
                for p, p0, d in zip(self.model.parameters(), self.original_params, self.direction):
                    # p.data = p0 + alpha * d
                    p.data.copy_(p0 + alpha * d)
            loss = compute_nll_loss(self.model, self.dataloader, pad_idx, target_tokens=100000)
            losses.append(loss)

        # Restore parameters
        with torch.no_grad():
            for p, p0 in zip(self.model.parameters(), self.original_params):
                p.data.copy_(p0)

        return {"alphas": alphas, "loss_values": np.array(losses)}


# =====================================================================
# Main (Modified for Combined Plotting)
# =====================================================================
def main():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    pad_idx = tokenizer.pad_token_id
    dataloader = get_dataloader(tokenizer)

    all_results_cola = {}
    all_results_baseline = {}
    
    # 1. Compute Results for both methods
    for m in CHECKPOINTS:
        # --- COLA Model ---
        model_path_cola = f"{BASE_DIR_COLA}/{m}"
        print(f"\nLoading COLA Model: {model_path_cola}")
        try:
            # Assuming you want to load ColaConfig/ColaForCausalLM for the COLA checkpoints
            config_cola = ColaConfig.from_pretrained(model_path_cola)
            model_cola = ColaForCausalLM.from_pretrained(model_path_cola, torch_dtype=torch.bfloat16, device_map="auto")
        except:
            # Fallback/Default for loading the model if custom Cola classes aren't available/correct
            print("Using AutoModel/AutoConfig for COLA model.")
            config_cola = AutoConfig.from_pretrained(model_path_cola)
            model_cola = AutoModelForCausalLM.from_pretrained(model_path_cola, torch_dtype=torch.bfloat16, device_map="auto")

        drawer_cola = LossLandscapeDrawer(model_cola, dataloader)
        result_cola = drawer_cola.synthesize_and_compute(ALPHA_MIN, ALPHA_MAX, ALPHA_INTERVAL, pad_idx)
        all_results_cola[m] = result_cola
        del model_cola # Free memory
        torch.cuda.empty_cache()

        # --- BASELINE Model ---
        model_path_baseline = f"{BASE_DIR_BASELINE}/{m}"
        print(f"\nLoading BASELINE Model: {model_path_baseline}")
        config_baseline = AutoConfig.from_pretrained(model_path_baseline)
        model_baseline = AutoModelForCausalLM.from_pretrained(model_path_baseline, torch_dtype=torch.bfloat16, device_map="auto")
        
        drawer_baseline = LossLandscapeDrawer(model_baseline, dataloader)
        result_baseline = drawer_baseline.synthesize_and_compute(ALPHA_MIN, ALPHA_MAX, ALPHA_INTERVAL, pad_idx)
        all_results_baseline[m] = result_baseline
        del model_baseline # Free memory
        torch.cuda.empty_cache()


    # 2. Combined Plot with Subplots
    num_models = len(CHECKPOINTS)
    cols = 3
    rows = int(np.ceil(num_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Determine common y-axis limits (optional, but ensures same scale for comparison)
    all_losses = np.concatenate([res["loss_values"] for res in all_results_cola.values()] + 
                                [res["loss_values"] for res in all_results_baseline.values()])
    y_min = np.floor(all_losses.min() * 10) / 10 - 0.1 # Example: round down to nearest 0.1 and subtract 0.1
    y_max = np.ceil(all_losses.max() * 10) / 10 + 0.1  # Example: round up to nearest 0.1 and add 0.1


    for i, name in enumerate(CHECKPOINTS):
        ax = axes[i]
        
        # Plot COLA results
        res_cola = all_results_cola[name]
        alphas, losses = res_cola["alphas"], res_cola["loss_values"]
        # Convert alphas to x 10^-3 for better x-axis readability
        ax.plot(alphas * 1000, losses, lw=2, label="COLA", color='tab:blue') 
        
        # Plot BASELINE results
        res_baseline = all_results_baseline[name]
        alphas, losses = res_baseline["alphas"], res_baseline["loss_values"]
        ax.plot(alphas * 1000, losses, lw=2, label="BASELINE", color='tab:orange', linestyle='--')
        
        ax.set_title(name, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Add labels to the edge subplots
        if i % cols == 0:
            ax.set_ylabel("NLL Loss")
        if i >= (rows - 1) * cols:
            ax.set_xlabel(r"$\alpha \ (\times 10^{-3})$")
            
        # Add legend to the first subplot only
        if i == 0:
            ax.legend()
            
        # Manually set y-limits for consistency (if common scale is desired)
        # ax.set_ylim(y_min, y_max)


    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("1D Loss Landscapes: COLA vs. BASELINE Across Checkpoints", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("loss_landscape_cola_vs_baseline_combined.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()