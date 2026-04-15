import numpy as np
import matplotlib.pyplot as plt
from ddft_solver import DDFT_Inertia_1D

# Define physical parameters
optsPhys = {
    'kBT': 1.0,
    'mS': [1.0],               # Mass of your MD particles
    'gammaS': [5.0],           # Friction coefficient
    'nParticlesS': [5.0]       # Total number of particles integrated over domain
}

# Define numerical parameters
yMax = 13.29
optsNum = {
    'yMin': 0.0,
    'yMax': 10.0,
    'N': 100                   # Number of spectral grid points
}

# Initialize solver with ZERO potential (Ideal Gas Bulk)
solver = DDFT_Inertia_1D(optsPhys, optsNum)
solver.Vext = np.zeros(solver.N)
solver.Vext_grad = np.zeros(solver.N)

# Get the flat initial condition
# Because Vext is 0, xv_ic_flat[:N] is exactly kBT * ln(rho_flat)
xv_ic_flat = solver.compute_equilibrium()

print("Initial flat equilibrium found. Turning on Gaussian potential...")

# Turn on the Gaussian External Potential
A = 5.0      # Amplitude
mu = 5.0     # Mean
sigma = 2.0  # Standard deviation

solver.Vext = A * np.exp(-0.5 * ((solver.y - mu) / sigma)**2)
solver.Vext_grad = solver.Dy.dot(solver.Vext)

# Since x = kBT * ln(rho) + Vext, and we just increased Vext, 
# we must add the new Vext to our initial state vector 'x' 
# so that the physical density rho remains flat at t=0!
xv_ic_flat[:solver.N] += solver.Vext

# Run time integration
tMax = 10.0
t_eval = np.linspace(0, tMax, 101)

print("Starting Spectral DDFT Simulation with Gaussian V_ext...")
sol = solver.solve_dynamics([0, tMax], t_eval, xv_ic=xv_ic_flat)

# 5. Extract and Plot Results
# Unpack state vector
xv_t = sol.y.T.reshape(len(t_eval), 2 * solver.N, solver.nSpecies)
x_t = xv_t[:, :solver.N, :]
v_t = xv_t[:, solver.N:, :]

# Transform back to physical density
rho_t = np.exp((x_t - solver.Vext[None, :, None]) / solver.kBT)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("Density Evolution")
for i in range(0, len(t_eval), 20):
    plt.plot(solver.y, rho_t[i, :, 0], label=f"t={t_eval[i]:.1f}")
plt.xlabel("z-axis")
plt.ylabel(r"$\rho(z)$")
plt.legend()

plt.subplot(122)
plt.title("Velocity Evolution")
for i in range(0, len(t_eval), 20):
    plt.plot(solver.y, v_t[i, :, 0])
plt.xlabel("z-axis")
plt.ylabel("v(z)")
plt.savefig(f'test_plot.png')
