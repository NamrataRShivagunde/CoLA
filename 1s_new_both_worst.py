import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from typing import List, Dict
from cola import ColaConfig, ColaForCausalLM # Assuming you have a custom cola module

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALPHA_MIN = -0.010
ALPHA_MAX = 0.010
ALPHA_INTERVAL = 0.0005

# --- BASE_DIR Definitions (Set these to your actual paths) ---
BASE_DIR_COLA = "checkpoints/cola_60m-2025-11-03-16-14-50"
BASE_DIR_BASELINE = "../GaLore/checkpoints/llama_60m-2025-11-04-16-37-42"
CHECKPOINTS = [f"model_{i}" for i in range(1000, 11000, 1000)]
FINAL_CHECKPOINT = "model_10000" # Define the checkpoint used for the direction vector
# --------------------------------

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
# Loss Landscape Drawer (Modified for Worst-Case/DoC Direction)
# =====================================================================
class LossLandscapeDrawer:
    def __init__(self, model: AutoModelForCausalLM, dataloader: torch.utils.data.DataLoader, direction_model: AutoModelForCausalLM = None):
        self.model = model
        self.dataloader = dataloader
        self.original_params = [p.data.clone() for p in model.parameters()]
        
        if direction_model is not None:
            # 1. Define direction as the difference between the current model and a target model (direction_model)
            print("Using Difference-of-Checkpoints (DoC) direction.")
            self.direction = []
            
            # Ensure parameters are aligned (though usually guaranteed in HF models)
            current_params = list(model.parameters())
            target_params = list(direction_model.parameters())

            for p_current, p_target in zip(current_params, target_params):
                # Direction: Target (Final) - Current (Checkpoint)
                d = (p_target.data.clone() - p_current.data.clone()).to(p_current.device)
                self.direction.append(d)
            
            # Optional: Normalize the direction vector
            norm = torch.sqrt(sum([d.norm(2)**2 for d in self.direction]))
            if norm > 1e-6: # Avoid division by zero for the final checkpoint (m=10000)
                self.direction = [d / norm for d in self.direction]
        else:
            # Fallback/Original: Random Gaussian direction
            print("Using Random Gaussian direction.")
            self.direction = [torch.randn_like(p.data).to(p.device) for p in model.parameters()] 

    def synthesize_and_compute(self, x_min: float, x_max: float, x_interval: float) -> Dict[str, np.ndarray]:
        alphas = np.arange(x_min, x_max + x_interval / 2, x_interval)
        losses = []

        for i, alpha in enumerate(alphas):
            print(f"  α={alpha:.5f} ({i+1}/{len(alphas)})", end="\r")
            with torch.no_grad():
                for p, p0, d in zip(self.model.parameters(), self.original_params, self.direction):
                    # p.data = p0 + alpha * d
                    p.data.copy_(p0 + alpha * d)
            loss = compute_nll_loss(self.model, self.dataloader)
            losses.append(loss)

        # Restore parameters
        with torch.no_grad():
            for p, p0 in zip(self.model.parameters(), self.original_params):
                p.data.copy_(p0)

        return {"alphas": alphas, "loss_values": np.array(losses)}


# =====================================================================
# Main (Modified for DoC Direction)
# =====================================================================
def main():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    dataloader = get_dataloader(tokenizer)

    # 1. Load the FINAL checkpoint model to define the DoC direction
    print(f"\n--- Loading Final Checkpoints to Define Direction ({FINAL_CHECKPOINT}) ---")
    
    # COLA Final Model
    final_path_cola = f"{BASE_DIR_COLA}/{FINAL_CHECKPOINT}"
    try:
        final_model_cola = ColaForCausalLM.from_pretrained(final_path_cola, torch_dtype=torch.bfloat16, device_map="auto")
    except:
        final_model_cola = AutoModelForCausalLM.from_pretrained(final_path_cola, torch_dtype=torch.bfloat16, device_map="auto")
    
    # BASELINE Final Model
    final_path_baseline = f"{BASE_DIR_BASELINE}/{FINAL_CHECKPOINT}"
    final_model_baseline = AutoModelForCausalLM.from_pretrained(final_path_baseline, torch_dtype=torch.bfloat16, device_map="auto")

    # 2. Compute Results for both methods using the DoC direction
    all_results_cola = {}
    all_results_baseline = {}
    
    for m in CHECKPOINTS:
        # --- COLA Model ---
        model_path_cola = f"{BASE_DIR_COLA}/{m}"
        print(f"\nLoading COLA Model: {model_path_cola}")
        try:
            model_cola = ColaForCausalLM.from_pretrained(model_path_cola, torch_dtype=torch.bfloat16, device_map="auto")
        except:
            model_cola = AutoModelForCausalLM.from_pretrained(model_path_cola, torch_dtype=torch.bfloat16, device_map="auto")

        # Pass the final model as the direction_model
        drawer_cola = LossLandscapeDrawer(model_cola, dataloader, direction_model=final_model_cola)
        result_cola = drawer_cola.synthesize_and_compute(ALPHA_MIN, ALPHA_MAX, ALPHA_INTERVAL)
        all_results_cola[m] = result_cola
        del model_cola 
        torch.cuda.empty_cache()

        # --- BASELINE Model ---
        model_path_baseline = f"{BASE_DIR_BASELINE}/{m}"
        print(f"\nLoading BASELINE Model: {model_path_baseline}")
        config_baseline = AutoConfig.from_pretrained(model_path_baseline)
        model_baseline = AutoModelForCausalLM.from_pretrained(model_path_baseline, torch_dtype=torch.bfloat16, device_map="auto")
        
        # Pass the final model as the direction_model
        drawer_baseline = LossLandscapeDrawer(model_baseline, dataloader, direction_model=final_model_baseline)
        result_baseline = drawer_baseline.synthesize_and_compute(ALPHA_MIN, ALPHA_MAX, ALPHA_INTERVAL)
        all_results_baseline[m] = result_baseline
        del model_baseline
        torch.cuda.empty_cache()
    
    # Free up the direction models
    del final_model_cola, final_model_baseline
    torch.cuda.empty_cache()


    # 3. Combined Plot with Subplots (Same plotting logic as before)
    num_models = len(CHECKPOINTS)
    cols = 3
    rows = int(np.ceil(num_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Determine common y-axis limits (optional, but ensures same scale for comparison)
    all_losses = np.concatenate([res["loss_values"] for res in all_results_cola.values()] + 
                                [res["loss_values"] for res in all_results_baseline.values()])
    y_min = np.floor(all_losses.min() * 10) / 10 - 0.1
    y_max = np.ceil(all_losses.max() * 10) / 10 + 0.1

    for i, name in enumerate(CHECKPOINTS):
        ax = axes[i]
        
        # Plot COLA results
        res_cola = all_results_cola[name]
        alphas, losses = res_cola["alphas"], res_cola["loss_values"]
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
            # Note: X-axis label is still alpha, but direction is now DoC
            ax.set_xlabel(r"$\alpha \cdot (\mathbf{w}_{final} - \mathbf{w}_{curr}) \ (\times 10^{-3})$")
            
        if i == 0:
            ax.legend()
            
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"1D Loss Landscapes: COLA vs. BASELINE (Along Optimization Path to {FINAL_CHECKPOINT})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("loss_landscape_cola_vs_baseline_worst.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()