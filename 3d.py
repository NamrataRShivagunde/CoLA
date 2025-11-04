import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# --------------------------
# Config
# --------------------------
CHECKPOINTS = [
    'model_1000.pt', 'model_2000.pt', 'model_3000.pt',
    'model_4000.pt', 'model_5000.pt', 'model_6000.pt',
    'model_7000.pt', 'model_8000.pt', 'model_9000.pt', 'model_10000.pt'
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Suppose you already have a working function that returns loss
def compute_val_loss(model):
    # You can replace this with your actual validation loss computation
    # For example:
    #   outputs = model(inputs)
    #   loss = criterion(outputs, labels)
    #   return loss.item()
    return np.random.uniform(4.6, 5.3)  # dummy placeholder for structure

# --------------------------
# Generate direction vectors
# --------------------------
def get_random_directions(model):
    directions = []
    for _ in range(2):
        direction = [torch.randn_like(p) for p in model.parameters()]
        norm = torch.sqrt(sum(torch.sum(d**2) for d in direction))
        directions.append([d / norm for d in direction])
    return directions[0], directions[1]

# --------------------------
# Evaluate loss over grid
# --------------------------
def evaluate_landscape(model, theta_star, d1, d2, alphas, betas):
    Z = np.zeros((len(alphas), len(betas)))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            # Perturb weights
            new_state = {k: theta_star[k] + a * d1[k] + b * d2[k] for k in theta_star.keys()}
            model.load_state_dict(new_state)
            Z[i, j] = compute_val_loss(model)
    return Z

# --------------------------
# Visualization
# --------------------------
def plot_3d_surface(X, Y, Z, title):
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_xlabel(r'$\alpha_1$')
    ax.set_ylabel(r'$\alpha_2$')
    ax.set_zlabel('NLL Loss')
    ax.set_title(title)
    plt.tight_layout()
    return fig

# --------------------------
# Generate & Animate
# --------------------------
alphas = np.linspace(-10e-3, 10e-3, 25)
betas = np.linspace(-10e-3, 10e-3, 25)
X, Y = np.meshgrid(alphas, betas)

all_Z = []

for ckpt in CHECKPOINTS:
    print(f"Processing {ckpt}...")
    model = torch.load(ckpt, map_location=DEVICE)
    theta_star = model.state_dict()
    d1, d2 = get_random_directions(model)
    d1 = {k: v for k, v in zip(theta_star.keys(), d1)}
    d2 = {k: v for k, v in zip(theta_star.keys(), d2)}
    Z = evaluate_landscape(model, theta_star, d1, d2, alphas, betas)
    all_Z.append(Z)

# --------------------------
# Create Animation
# --------------------------
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.imshow(all_Z[0], origin='lower', cmap='plasma', extent=[alphas.min(), alphas.max(), betas.min(), betas.max()])
ax.set_xlabel(r'$\alpha_1$')
ax.set_ylabel(r'$\alpha_2$')
ax.set_title('2D Loss Contours Evolution')
fig.colorbar(cax, ax=ax, label='Loss')

def update(frame):
    ax.clear()
    ax.set_title(f"Loss Landscape at {CHECKPOINTS[frame]}")
    cax = ax.imshow(all_Z[frame], origin='lower', cmap='plasma',
                    extent=[alphas.min(), alphas.max(), betas.min(), betas.max()])
    return [cax]

ani = animation.FuncAnimation(fig, update, frames=len(all_Z), interval=800, blit=False)
ani.save("loss_landscape_evolution.gif", writer='pillow')
plt.close(fig)

print("✅ Animation saved as loss_landscape_evolution.gif")
