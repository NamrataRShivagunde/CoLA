import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer # Explicitly import Qwen2Tokenizer
from datasets import load_dataset
from typing import List, Callable, Dict

from cola import ColaConfig, ColaForCausalLM, ColaMForCausalLM

# Model Configuration
CUDA_VISIBLE_DEVICES = "0"  # Set to specific GPU ID or use "cuda" for automatic selection
MODEL_NAME = "checkpoints/cola_60m-2025-11-03-16-14-50/model_1000" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss Landscape Configuration (Alpha points for X-axis)
ALPHA_MIN = -0.010
ALPHA_MAX = 0.010
ALPHA_INTERVAL = 0.0005

# =========================================================================
# === 3. BENCHMARK FUNCTION: NLL LOSS ===
# =========================================================================

def compute_nll_loss(model: AutoModelForCausalLM, dataloader: torch.utils.data.DataLoader) -> float:
    """
    Computes the Negative Log-Likelihood (NLL) loss on a given dataloader.

    NLL loss is commonly used for likelihood-based benchmarks, which typically
    yield a smooth and continuous loss landscape (U-shaped).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # We limit to a small number of batches to speed up the plotting
    MAX_BATCHES = 5

    # Get the device from the first parameter of the model
    model_device = next(model.parameters()).device


    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= MAX_BATCHES:
                break

            # Move input tensors to the model's device
            input_ids = batch['input_ids'].to(model_device)
            attention_mask = batch['attention_mask'].to(model_device)

            # Forward pass: labels are shifted inside the model for Causal LM
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # Use input_ids as labels for LM
            )

            # Loss is averaged over the batch and sequence length
            loss = outputs.loss.item()
            total_loss += loss
            total_tokens += 1

    if total_tokens == 0:
        return 100.0 # Return high loss if no data

    # Return average loss per batch
    return total_loss / total_tokens

# =========================================================================
# === 4. DATA PREPARATION ===
# =========================================================================

def get_dataloader(tokenizer: AutoTokenizer) -> torch.utils.data.DataLoader:
    """Loads and processes the Wikitext-2 dataset."""
    print("Loading Wikitext-2 dataset...")
    # Using Wikitext-2-raw-v1 (small, common benchmark)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    # Tokenize the dataset
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Convert to PyTorch Tensors
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Create DataLoader
    return torch.utils.data.DataLoader(tokenized_datasets, batch_size=4)

# =========================================================================
# === 5. LOSS LANDSCAPE VISUALIZATION CLASS ===
# =========================================================================

class LossLandscapeDrawer:
    """
    Handles parameter perturbation, benchmark evaluation, and plotting for 1D loss landscapes.
    """
    def __init__(self, model: AutoModelForCausalLM, dataloader: torch.utils.data.DataLoader):
        self.model = model
        self.dataloader = dataloader
        self.device = model.device # Model is already on the correct device due to device_map="auto"
        self.original_params: List[torch.Tensor] = []
        self.direction: List[torch.Tensor] = []

        self._cache_original_params()
        self._generate_gaussian_direction()

    def _cache_original_params(self):
        # Store the original, trained weights (theta_0) for restoration
        print("Caching original parameters...")
        # Remove .to(self.device) as device_map="auto" handles device placement
        self.original_params = [p.data.clone() for p in self.model.parameters()]


    def _generate_gaussian_direction(self):
        # Create a random Gaussian direction vector (delta) for the "most-case" landscape [cite: 126]
        print("Generating Gaussian direction (delta) for perturbation...")
        with torch.no_grad():
            for param in self.model.parameters():
                # Ensure delta is on the same device as the parameter
                delta = torch.randn_like(param.data).to(param.device)
                self.direction.append(delta)

    def synthesize_and_compute(self, x_min: float, x_max: float, x_interval: float) -> Dict[str, np.ndarray]:
        """Calculates loss L(alpha) = J(theta_0 + alpha * delta) over alpha range."""

        # 1. Synthesize alpha coordinates (X-axis)
        alphas = np.arange(x_min, x_max + x_interval / 2, x_interval)
        loss_values = []

        # 2. Compute Loss for each alpha
        print(f"Computing loss for {len(alphas)} alpha points...")
        for i, alpha in enumerate(alphas):
            print(f"  Evaluating point {i+1}/{len(alphas)}: alpha={alpha:.5f}", end='\r')

            # Perturb the parameters: theta_alpha = theta_0 + alpha * delta
            with torch.no_grad():
                for param, original_param, delta in zip(
                    self.model.parameters(),
                    self.original_params,
                    self.direction
                ):
                     # Ensure all tensors are on the same device before the operation
                    param.data.copy_(original_param + alpha * delta)


            # Evaluate the NLL benchmark (J(theta_alpha))
            avg_loss = compute_nll_loss(self.model, self.dataloader)
            loss_values.append(avg_loss)

        # 3. Restore original parameters (CRITICAL)
        with torch.no_grad():
            for param, original_param in zip(self.model.parameters(), self.original_params):
                 # Ensure tensors are on the same device for restoration
                param.data.copy_(original_param)


        print("\nComputation complete. Original parameters restored.")

        return {"alphas": alphas, "loss_values": np.array(loss_values)}

    def plot_1d(self, results: Dict[str, np.ndarray]):
        """Plots the 1D loss landscape."""
        alphas = results["alphas"]
        loss_values = results["loss_values"]

        plt.figure(figsize=(9, 5))
        # Scale alpha by 1000 for cleaner x-axis labeling (as seen in the paper's figures)
        plt.plot(alphas * 1000, loss_values, linewidth=2, label="NLL Loss")

        plt.xlabel(r'$\alpha \ (\times 10^{-3})$ (Perturbation Scale)')
        plt.ylabel('Negative Log-Likelihood (NLL)')
        plt.title(f"1D Loss Landscape of {MODEL_NAME} (Most-Case Direction)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(f"loss_landscape_1d_cola_1000.png", dpi=300)

# =========================================================================
# === 6. MAIN EXECUTION ===
# =========================================================================

def main():
    # 1. Model and Tokenizer Setup
    print(f"Loading Model: {MODEL_NAME} on {DEVICE}")

    model_config = ColaConfig.from_pretrained(MODEL_NAME)

    # Load model without bitsandbytes quantization, potentially in bfloat16
    model = ColaForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16, # Specify bfloat16 dtype
    )
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    # 2. Data Setup
    dataloader = get_dataloader(tokenizer)

    # 3. Initialize Drawer and Compute Landscape
    drawer = LossLandscapeDrawer(model, dataloader)

    results = drawer.synthesize_and_compute(ALPHA_MIN, ALPHA_MAX, ALPHA_INTERVAL)

    # 4. Plot Results
    drawer.plot_1d(results)

if __name__ == "__main__":
    main()