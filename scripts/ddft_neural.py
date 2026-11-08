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

from goddard_psm.ddft_solver import DDFT_Inertia_1D_Neural
from scipy.interpolate import interp1d

from neural_ddft.models.neural_free_energy import NeuralFreeEnergyFine, NeuralFreeEnergy
from neural_ddft.utils.functions import plot_movie
from neural_ddft.utils.config import load_config, get_next_result_path, save_config

PARAM_SCHEME = {
    "run_id": (str, "psm_eval"),
    "md_path": (str, "../neural-ddft-pcm/data/g1-closed-dz100/avg_5000"),
    "ref_md_path": (str, None),
    "type": (str, "c2"),
    "base_model": (str, None),
    "k": (float, 1.0),
    "T": (float, 2.0),
    "m": (float, 1.0),
    "gamma": (float, 2.0),
    "eta": (float, 0.01),
    "N": (int, 100),  # Number of spectral (Chebyshev/Fourier) grid points
    #"dt": (float, 1e-4),
}

args = load_config(PARAM_SCHEME, description="Modular Neural DDFT with spectral solver (PSM)")

current_time = datetime.now().strftime("%Y-%m-%d")
base_path = f"results/{current_time}_{args['run_id']}"
results_path = get_next_result_path(base_path)
os.makedirs(results_path, exist_ok=True)
save_config(args, results_path)

# Load MD data

file_pattern = f"{args['md_path']}/collection.*.h5"
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
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(f"Using compute device: {device}")

if int(1/dz) == 100:
    model = NeuralFreeEnergyFine.load_from_checkpoint(checkpoint_path=f"../neural-ddft-pcm/ddft/models/{args['type']}_dz100.ckpt", map_location=torch.device('cpu'))
    
else:
    model = NeuralFreeEnergy.load_from_checkpoint(checkpoint_path=f"../neural-ddft-pcm/ddft/models/{args['type']}_dz{int(1/dz)}.ckpt", map_location=torch.device('cpu'))

if(args['base_model'] is not None):
    # 2. Now, load your FINE-TUNED pure-PyTorch weights
    finetuned_ckpt = f"../neural-ddft-pcm/ddft/models/{args['base_model']}"
    finetuned_data = torch.load(finetuned_ckpt, map_location=torch.device('cpu'))

    # 3. Overwrite the base model's weights with the fine-tuned weights
    model.load_state_dict(finetuned_data['state_dict'])

model.eval()

# Assuming you loaded init_rho, V, r, and the model as in your script

optsPhys = {
    'kBT': args['T'] * args['k'], # using your K and T
    'mS': [args['m']],         # Mass 
    'gammaS': [args['gamma']], # Friction
    'eta': args['eta'],        # Viscosity
    'nParticlesS': [np.sum(init_rho) * dz] # Total mass integral
}

optsNum = {
    'yMin': r[0],
    'yMax': r[-1],
    'N': args['N']  # Number of Chebyshev/Fourier points (adjust for speed/accuracy)
}

# 1. Initialize solver with ZERO potential (Ideal Gas Bulk)
solver = DDFT_Inertia_1D_Neural(optsPhys, optsNum, model, r)
solver.Vext = np.zeros(solver.N)
solver.Vext_grad = np.zeros(solver.N)

# 2. Get the flat initial condition
# Because Vext is 0, xv_ic_flat[:N] is exactly kBT * ln(rho_flat)
xv_ic_flat = solver.compute_equilibrium()

# Assign the REAL external potential (loaded from the MD file above, variable
# `V` on the uniform grid `r`) to the solver's own Fourier/Chebyshev grid, via
# a periodic cubic spline - same periodic-padding convention already used by
# DDFT_Inertia_1D_Neural.predict_dFdrho() for density, so the interpolation
# is consistent across the whole solver. Previously this block ignored the
# loaded V entirely and hardcoded a Gaussian-potential shape (A=5,mu=5,
# sigma=2) instead - only coincidentally correct for the "gaussian" MD
# dataset, silently wrong for every other potential (3a/3b/3c/3d).
from scipy.interpolate import CubicSpline
# `r` is a 320-point cell-center periodic grid, so its TRUE period is
# r[-1]-r[0]+dz (not solver.line.L=r[-1]-r[0], which is one cell short -
# that quantity belongs to the solver's own, separately-resolution'd
# internal grid and is not the right pad distance for `r` itself).
dz_uniform = r[1] - r[0]
r_padded = np.append(r, r[-1] + dz_uniform)
V_padded = np.append(V, V[0])
spline_V = CubicSpline(r_padded, V_padded, bc_type='periodic')
solver.Vext = spline_V(solver.y)
solver.Vext_grad = solver.Dy.dot(solver.Vext)  # Spectral derivative of potential!

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

# On an early blow-up abort (sol.status==1), solve_ivp's t_eval only contains
# the points actually reached - sol.y has fewer columns than the full
# t_eval requested, so reshaping against the full array crashes. Trim
# t_eval to match what was actually returned, so a blow-up still saves its
# partial trajectory up to the failure point instead of losing everything.
t_eval = t_eval[:sol.y.shape[1]]

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
plot_movie(results_path, args['md_path'], args['ref_md_path'], True)
