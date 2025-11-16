import torch
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# ===================================== #
# PINN for Damped Harmonic Oscillator
# ===================================== #

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z_xi):
        out = self.layer(z_xi)
        return out

def pde_residual(model, z, xi, device):
    z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=device)
    xi_tensor = torch.tensor(xi, dtype=torch.float32, requires_grad=True, device=device)
    input_tensor = torch.cat((z_tensor, xi_tensor), dim=1)

    x_pred = model(input_tensor)

    dx_dz = torch.autograd.grad(x_pred, z_tensor, grad_outputs=torch.ones_like(x_pred), create_graph=True)[0]
    d2x_dz2 = torch.autograd.grad(dx_dz, z_tensor, grad_outputs=torch.ones_like(dx_dz), create_graph=True)[0]

    # PDE: x'' + 2*xi*x' + x = 0
    residual = d2x_dz2 + 2 * xi_tensor * dx_dz + x_pred
    loss = torch.mean(residual**2)
    return loss

def ic_loss(model, z0, xi0, x0, v0, device):
    z0_tensor = torch.tensor(z0, dtype=torch.float32, requires_grad=True, device=device)
    xi0_tensor = torch.tensor(xi0, dtype=torch.float32, requires_grad=True, device=device)
    input0 = torch.cat((z0_tensor, xi0_tensor), dim=1)

    x0_pred = model(input0)
    dx0_dz = torch.autograd.grad(x0_pred, z0_tensor, grad_outputs=torch.ones_like(x0_pred), create_graph=True)[0]

    loss_ic = (x0_pred - x0)**2 + (dx0_dz - v0)**2
    return loss_ic.mean()

# --------------------------------------
# Training Function
# --------------------------------------

def pinn_train(z, xi, x0, v0, epochs=12000, device="cpu"):
    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    tic = time.time()
    loss_history = []

    for epoch in range(epochs):
        model.train()
        
        # Physics Loss
        loss_eqn = pde_residual(model, z, xi, device)

        # IC Loss (Randomized xi sample for robustness)
        xi_ic_sample = np.random.uniform(0.1, 0.4, size=(10, 1))
        z0_sample = np.zeros((10, 1))
        loss_ic = ic_loss(model, z0_sample, xi_ic_sample, x0, v0, device)

        loss = loss_eqn + 10 * loss_ic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6e}")

    toc = time.time()
    elapsed_time = toc - tic
    print(f"Elapsed Time = {elapsed_time:.2f} seconds")
    return model, loss_history, elapsed_time

def damped_oscillator(z, x0, v0, xi):
    # Vectorized solution for the damped harmonic oscillator
    # This handles scalar, 1D, and 2D array inputs for xi
    
    # Ensure z and xi are numpy arrays for broadcasting
    z = np.asarray(z)
    xi = np.asarray(xi)

    # Add a tiny epsilon to xi to avoid division by zero or sqrt(0) at xi=1.0
    # These "wrong" values will be overwritten by np.where,
    # but they prevent NaN errors during calculation.
    xi_under_safe = np.where(np.isclose(xi, 1.0), 1.0 - 1e-8, xi)
    xi_over_safe = np.where(np.isclose(xi, 1.0), 1.0 + 1e-8, xi)

    # --- Calculate Underdamped solution (xi < 1) for ALL points ---
    omega_d_under = np.sqrt(1 - xi_under_safe**2)
    term1_under = x0 * np.cos(omega_d_under * z)
    term2_under = (v0 + xi_under_safe * x0) / omega_d_under * np.sin(omega_d_under * z)
    out_under = np.exp(-xi_under_safe * z) * (term1_under + term2_under)
    
    # --- Calculate Critically Damped solution (xi == 1) for ALL points ---
    out_crit = np.exp(-z) * (x0 + (v0 + x0) * z)
    
    # --- Calculate Overdamped solution (xi > 1) for ALL points ---
    omega_d_over = np.sqrt(xi_over_safe**2 - 1)
    # Avoid 0/0 if omega_d_over is zero (which it shouldn't be with xi_over_safe)
    omega_d_over = np.where(np.isclose(omega_d_over, 0), 1e-8, omega_d_over)
    
    c1_over = (v0 + (xi_over_safe + omega_d_over) * x0) / (2 * omega_d_over)
    c2_over = x0 - c1_over
    out_over = np.exp(-xi_over_safe * z) * (c1_over * np.exp(omega_d_over * z) + c2_over * np.exp(-omega_d_over * z))

    # --- Use np.where to pick the correct solution based on the original xi ---
    
    # Start with underdamped solution by default
    out = np.where(xi < 1.0, out_under, 0.0)
    
    # Where xi is overdamped, use the overdamped solution
    out = np.where(xi > 1.0, out_over, out)
    
    # Where xi is critically damped, use the critically damped solution
    # np.isclose is the safest way to check for floating point equality
    out = np.where(np.isclose(xi, 1.0), out_crit, out)
    
    return out


# --------------------------------------
# Main Execution & Plotting
# --------------------------------------

if __name__ == "__main__":
    # Data Prep
    nPt = 1000
    z_train = np.linspace(0, 20, nPt).reshape(-1, 1)
    xi_train = np.random.uniform(0.1, 0.4, size=(nPt, 1))
    x0, v0 = 0.7, 1.2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    num_epochs = 12000

    # Train
    model, loss_hist, training_time = pinn_train(z_train, xi_train, x0, v0, epochs=num_epochs, device=device)

    # ==========================================================
    # GRAPH 1: Analytical vs PINN Comparison
    # ==========================================================
    plt.figure(figsize=(10, 6))
    
    z_plot = np.linspace(0, 20, 200).reshape(-1, 1)
    test_xis = [0.25]
    colors = ['green']

    for i, xi_val in enumerate(test_xis):
        # Analytic
        y_true = damped_oscillator(z_plot, x0, v0, xi_val)

        # Prediction
        xi_eval = xi_val * np.ones_like(z_plot)
        z_t = torch.tensor(z_plot, dtype=torch.float32, device=device)
        xi_t = torch.tensor(xi_eval, dtype=torch.float32, device=device)
        in_eval = torch.cat((z_t, xi_t), dim=1)
        y_pred = model(in_eval).detach().cpu().numpy()

        plt.plot(z_plot, y_true, linestyle='--', color=colors[i], alpha=0.5)
        plt.plot(z_plot, y_pred, linestyle='-', color=colors[i], linewidth=1.5, label=f'ξ={xi_val}')

    plt.plot([], [], 'k--', label='Analytical')
    plt.plot([], [], 'k-', label='PINN Prediction')
    plt.title("Graph 1: PINN Predictions vs Analytical Solution")
    plt.xlabel("Time (z)")
    plt.ylabel("Displacement (x)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

    # ==========================================================
    # GRAPH 2: Training Efficiency (Loss Convergence)
    # ==========================================================
    plt.figure(figsize=(10, 5))
    plt.plot(loss_hist, 'k-', lw=1.5)
    plt.yscale('log')
    plt.title("Graph 2: Training Loss Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Total Loss (MSE)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

    # ==========================================================
    # GRAPH 3: Global Efficiency (Error Heatmap)
    # ==========================================================
    # Generate a meshgrid over the entire domain of Z and Xi
    z_mesh = np.linspace(0, 20, 200)
    xi_mesh = np.linspace(0.1, 0.4, 100)
    Z_grid, XI_grid = np.meshgrid(z_mesh, xi_mesh)

    # Prepare tensors
    z_flat = torch.tensor(Z_grid.flatten()[:, None], dtype=torch.float32, device=device)
    xi_flat = torch.tensor(XI_grid.flatten()[:, None], dtype=torch.float32, device=device)
    input_mesh = torch.cat((z_flat, xi_flat), dim=1)

    # Get predictions and true values
    pred_mesh = model(input_mesh).detach().cpu().numpy().reshape(Z_grid.shape)
    true_mesh = damped_oscillator(Z_grid, x0, v0, XI_grid)
    
    # Calculate absolute error
    abs_error = np.abs(pred_mesh - true_mesh)

    plt.figure(figsize=(12, 6))
    cp = plt.pcolormesh(Z_grid, XI_grid, abs_error, cmap='inferno', shading='auto')
    cbar = plt.colorbar(cp)
    cbar.set_label('Absolute Error |True - Pred|')
    
    plt.title("Graph 3: Error Heatmap across Time (z) and Damping (ξ)")
    plt.xlabel("Time (z)")
    plt.ylabel("Damping Ratio (ξ)")
    plt.show()

    # ==========================================================

    # ==========================================================
    
    # Calculate error metrics from the heatmap data
    mean_abs_error = np.mean(abs_error)
    max_abs_error = np.max(abs_error)
    final_loss = loss_hist[-1]

    print("\n" + "="*50)
    print("      PINN PERFORMANCE SUMMARY (Copy to Post)")
    print("="*50 + "\n")
    
    print("--- Problem Setup ---")
    print(f"Equation:     x'' + 2*ξ*x' + x = 0")
    print(f"Parameters:   x(0) = {x0}, x'(0) = {v0}")
    print(f"Domain:       z ∈ [0, 20], ξ ∈ [0.1, 0.4]")
    print("\n--- Model & Training ---")
    print(f"Network:      4-Layer MLP (2-32-128-64-1) with SiLU")
    print(f"Device:       {device}")
    print(f"Epochs:       {len(loss_hist)}")
    print(f"Train Time:   {training_time:.2f} seconds")
    
    print("\n--- Performance Metrics ---")
    print(f"Final Loss:   {final_loss:.4e}")
    print(f"Mean Error:   {mean_abs_error:.4e} (Mean Abs Error on Test Grid)")
    print(f"Max Error:    {max_abs_error:.4e} (Max Abs Error on Test Grid)")
    print("\n" + "="*50 + "\n")
