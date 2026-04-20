import torch
import numpy as np
import h5py
import glob
from datetime import datetime
import os
import argparse

from phi.torch.flow import *
import matplotlib.pyplot as plt
from tqdm import tqdm

from ddft_solver import DDFT_Inertia_1D_Neural
from scipy.interpolate import interp1d

import sys
sys.path.append('../neural-ddft-pcm/ddft/')
from neural_free_energy import NeuralFreeEnergyFine, NeuralFreeEnergy
from functions import plot_movie

parser = argparse.ArgumentParser(description='neural DDFT')
parser.add_argument('--run_id', type=str, default='g1-closed')
parser.add_argument('--md_path', type=str, default='../neural-ddft-pcm/data/g1-closed-dz100/avg_5000')
parser.add_argument('--ref_md_path', type=str, default=None)
parser.add_argument('--type', type=str, default='c2')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--k', type=float, default=1)
parser.add_argument('--T', type=float, default=2)
parser.add_argument('--m', type=float, default=1)
parser.add_argument('--gamma', type=float, default=2)
parser.add_argument('--eta', type=float, default=0.01)
#parser.add_argument('--dt', type=float, default=1e-4)

args = parser.parse_args()

current_time = datetime.now().strftime("%Y-%m-%d")
#results_path = f"results/{current_time}_{args.run_id}_k{args.k}_T{args.T}_m{args.m}_g{args.gamma}_eta{args.eta}_dt{args.dt}_{args.type}"
results_path = f"results/{current_time}_{args.run_id}_k{args.k}_T{args.T}_m{args.m}_g{args.gamma}_eta{args.eta}_{args.type}"
os.makedirs(results_path, exist_ok=True)

# Load MD data

file_pattern = f"{args.md_path}/collection.*.h5"
md_file_list = sorted(glob.glob(file_pattern))

if len(md_file_list) == 0:
    raise FileNotFoundError(f"No MD files found matching pattern: {file_pattern}")

with h5py.File(md_file_list[0], 'r') as f:
    init_rho = np.array(f['n']).flatten()
    V = np.array(f['V']).flatten()
    r = np.array(f['r']).flatten()

dz = r[1] - r[0]
print(f'Number of particles: {np.sum(init_rho) * dz * 10 * 10} (dz: {dz})')

# Load F model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

if int(1/dz) == 100:
    model = NeuralFreeEnergyFine.load_from_checkpoint(checkpoint_path=f"../neural-ddft-pcm/ddft/models/{args.type}_dz100.ckpt", map_location=torch.device('cpu'))
    
else:
    model = NeuralFreeEnergy.load_from_checkpoint(checkpoint_path=f"../neural-ddft-pcm/ddft/models/{args.type}_dz{int(1/dz)}.ckpt", map_location=torch.device('cpu'))

if(args.model is not None):
    # 2. Now, load your FINE-TUNED pure-PyTorch weights
    finetuned_ckpt = f"../neural-ddft-pcm/ddft/models/{args.model}"
    finetuned_data = torch.load(finetuned_ckpt, map_location=device)

    # 3. Overwrite the base model's weights with the fine-tuned weights
    model.load_state_dict(finetuned_data['state_dict'])

model.eval()

# Assuming you loaded init_rho, V, r, and the model as in your script

optsPhys = {
    'kBT': args.T * args.k, # using your K and T
    'mS': [args.m],         # Mass 
    'gammaS': [args.gamma], # Friction
    'eta': args.eta,        # Viscosity
    'nParticlesS': [np.sum(init_rho) * dz] # Total mass integral
}

optsNum = {
    'yMin': r[0],
    'yMax': r[-1],
    'N': 100  # Number of Chebyshev points (you can adjust this for speed/accuracy)
}

# 1. Initialize solver with ZERO potential (Ideal Gas Bulk)
solver = DDFT_Inertia_1D_Neural(optsPhys, optsNum, model, r)
solver.Vext = np.zeros(solver.N)
solver.Vext_grad = np.zeros(solver.N)

# 2. Get the flat initial condition
# Because Vext is 0, xv_ic_flat[:N] is exactly kBT * ln(rho_flat)
xv_ic_flat = solver.compute_equilibrium()

# Assign your external potential to the solver grid
# (Interpolating your uniform V to the Chebyshev grid)
#interp_V = interp1d(r, V, kind='cubic', fill_value="extrapolate")
#solver.Vext = interp_V(solver.y)
#solver.Vext_grad = solver.Dy.dot(solver.Vext) # Spectral derivative of potential!
A = 5.0      
mu = 5.0     
sigma = 2.0  
L = solver.line.L # Box length (10)

# Periodized Gaussian (Summing the main hill + immediate neighbors)
solver.Vext = (
    A * np.exp(-0.5 * ((solver.y - mu) / sigma)**2) +
    A * np.exp(-0.5 * ((solver.y - mu + L) / sigma)**2) +
    A * np.exp(-0.5 * ((solver.y - mu - L) / sigma)**2)
)
solver.Vext_grad = solver.Dy.dot(solver.Vext)

# Since x = kBT * ln(rho) + Vext, and we just increased Vext, 
# we must add the new Vext to our initial state vector 'x' 
# so that the physical density rho remains flat at t=0!
xv_ic_flat[:solver.N] += solver.Vext

# Run time integration (SciPy handles the adaptive dt under the hood)
t_range = 4.0
save_time_interval = 1e-2
t_eval = np.arange(0, t_range, save_time_interval)

print("Starting Spectral DDFT Simulation...")
# Because spectral methods are stiff, we use 'BDF' which is ultra-stable
sol = solver.solve_dynamics([0, t_range], t_eval, xv_ic=xv_ic_flat) 

# Extract and save results
xv_t = sol.y.T.reshape(len(t_eval), 2 * solver.N, solver.nSpecies)
x_t = xv_t[:, :solver.N, :]
v_t = xv_t[:, solver.N:, :]

# Transform x back to physical density
rho_t = np.exp((x_t - solver.Vext[None, :, None]) / solver.kBT)

print("Saving results...")
for i, t in enumerate(t_eval):
    # Get current state on Chebyshev grid
    rho_current_cheb = rho_t[i, :, 0]
    v_current_cheb = v_t[i, :, 0]
    
    # Calculate dFdrho using our new method
    dFdrho_uniform, dFdrho_cheb = solver.predict_dFdrho(rho_current_cheb)
    
    # Optional: If you want to save everything on the UNIFORM grid so it 
    # perfectly matches your old PhiFlow plot_movie function expectations:
    interp_to_uniform = interp1d(solver.y, rho_current_cheb, kind='cubic', fill_value="extrapolate")
    rho_uniform = interp_to_uniform(r)
    
    interp_v_to_uniform = interp1d(solver.y, v_current_cheb, kind='cubic', fill_value="extrapolate")
    v_uniform = interp_v_to_uniform(r)

    interp_Vext_to_uniform = interp1d(solver.y, solver.Vext, kind='cubic', fill_value="extrapolate")
    Vext_uniform = interp_Vext_to_uniform(r)

    # Apply alpha if you are saving a modified version like in your script
    current_alpha = solver.get_alpha(t, t_range)
    dFdrho_mod_uniform = dFdrho_uniform * current_alpha

    # Save exactly matching the PhiFlow format!
    np.save(f"{results_path}/rho{t:06.3f}.npy", rho_uniform)
    np.save(f"{results_path}/v{t:06.3f}.npy", v_uniform)
    np.save(f"{results_path}/V_ext{t:06.3f}.npy", Vext_uniform)
    np.save(f"{results_path}/dFdrho{t:06.3f}.npy", dFdrho_uniform)
    np.save(f"{results_path}/dFdrho_mod{t:06.3f}.npy", dFdrho_mod_uniform)


print("Simulation complete! Generating movie...")
plot_movie(results_path, args.md_path, args.ref_md_path, True)
