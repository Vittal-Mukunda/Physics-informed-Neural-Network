# Physics-Informed Neural Network (PINN) for Damped Harmonic Oscillator

This project implements a **Physics-Informed Neural Network (PINN)** to approximate the solution of the **damped harmonic oscillator** differential equation:

$$
\frac{d^2x}{dt^2} + 2 \xi \frac{dx}{dt} + x = 0
$$

across a range of damping ratios \(\xi \in [0.1, 0.4]\).

Unlike traditional neural networks, the PINN **incorporates the governing physical law directly** into its loss function, allowing it to learn solutions that satisfy both the **differential equation** and the **initial conditions**:

- \(x(0) = x_0 = 0.7\)  
- \(\frac{dx}{dt}(0) = v_0 = 1.2\)


## Key Features

- **Physics-constrained learning:** Minimizes the **PDE residual** and initial condition losses to ensure physically consistent predictions.  
- **Handles varying damping ratios:** Takes damping ratio \(\xi\) as an input, allowing a single network to generalize across multiple damping scenarios.  
- **No large datasets required:** Learns directly from the **physics of the system** rather than from extensive simulation or experimental data.  
- **Visualization & verification:** Compares predicted solutions with the **analytical solution** for selected damping ratios to validate accuracy.  

---

## Implementation Details

### Neural Network Architecture

- Fully-connected feedforward network using **PyTorch**.  
- Architecture: `[Input (t, xi) → 32 → 128 → 64 → 1 Output (x)]`  
- Activation: **SiLU (Sigmoid Linear Unit)**  

### Physics-Informed Loss Function

The PINN enforces the physics by defining a custom loss:

1. **PDE Residual Loss**  
   Using PyTorch's **automatic differentiation**, derivatives of the network output with respect to time \(t\) are computed:

   \[
   \text{Residual} = \frac{d^2x}{dt^2} + 2 \xi \frac{dx}{dt} + x
   \]

   Loss = Mean Squared Error (MSE) of the residual.  

2. **Initial Condition Loss**  
   Ensures the network satisfies:

   \[
   x(0) = x_0, \quad \frac{dx}{dt}(0) = v_0
   \]

   Loss = MSE between predicted and true initial conditions.  

3. **Total Loss** = PDE Residual Loss + Initial Condition Loss  

The network is trained by minimizing this **physics-informed loss** using the **Adam optimizer**.

---

### Inputs & Outputs

- **Inputs:**  
  - Dimensionless time \(t\)  
  - Damping ratio \(\xi\)  

- **Output:**  
  - Displacement \(x(t)\) of the damped harmonic oscillator  

---

## Usage

1. Clone this repository:

```bash
git clone https://github.com/Vittal-Mukunda/Physics-informed-Neural-Network.git
cd Physics-informed-Neural-Network
