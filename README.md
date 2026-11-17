# goddard-ddft

A pseudo-spectral solver for **underdamped (inertial) dynamical density functional theory**
in 1D periodic systems. It is a Python port of the MATLAB pseudo-spectral DDFT code by Ben
Goddard (2DChebClass, [Nold et al. 2017](https://doi.org/10.1016/j.jcp.2016.12.023)), extended
with:
- a periodic Fourier line;
- the neural PCM excess free-energy functional of
  [Ram et al. (2025)](https://arxiv.org/abs/2505.09543);
- an optional learned superadiabatic correction force, as in
  [neural-ddft-pcm](https://github.com/Sidlertime/neural-ddft-pcm).

Its purpose in the project is **numerical validation**. It solves the same underdamped DDFT
equations as the finite-volume PhiFlow solver of neural-ddft-pcm, but with a completely
different discretization:
- space: spectral differentiation on a Fourier grid;
- time: adaptive implicit integration (`scipy.integrate.solve_ivp`, BDF).

If both solvers agree, the PhiFlow results are not artifacts of its discretization.

## Installation

The three project repositories share one conda environment and are expected next to each
other. The solver imports `neural_ddft` and loads the PCM checkpoints and MD data from
`../neural-ddft-pcm`.

```
workspace/
├── neural-ddft-pcm/   git clone https://github.com/Sidlertime/neural-ddft-pcm.git
├── goddard/           git clone https://github.com/Sidlertime/goddard-ddft.git goddard
└── mdext-ddft/        git clone https://github.com/Sidlertime/mdext-ddft.git
```

Clone this repository into a folder named `goddard`: the solver-comparison scripts of
neural-ddft-pcm look for it there. You can also set the environment variable `GODDARD_ROOT`.

```bash
cd neural-ddft-pcm
mamba env create -f environment.yml      # or: conda env create -f environment.yml
mamba activate neural-ddft
pip install -e ddft                      # neural_ddft package
cd ../goddard
pip install -e .                         # goddard_psm package
```

### Tests

```bash
pytest tests
```

The tests check that the Chebyshev differentiation matrix is exact for polynomials and that
the Clenshaw-Curtis weights integrate `sin` exactly. `python scripts/ddft_test.py` runs a
quick ideal-gas demo: no neural functional, Gaussian potential switched on at t = 0.

## Reproducing the results

All runs start from `goddard/` and write to
`results/<date>_<run_id>_NN/`. Each run writes `rho*.npy`, `v*.npy`, `V_ext*.npy` and
`dFdrho*.npy` per saved frame (every 0.01τ up to 4τ), plus `config.json` and a movie.

The physics matches the PhiFlow solver: reduced LJ units, k_BT = 2, m = 1, γ = 0.1,
η = 0.6944, PCM functional `c2`. The MD data must exist in `neural-ddft-pcm/data/lj_data/`;
see the [mdext-ddft](https://github.com/Sidlertime/mdext-ddft) README.

**PCM-only model** (equilibrium functional, no learned correction):

```bash
python scripts/ddft_neural.py --run_id psm_3a --type c2 --gamma 0.1 --eta 0.6944 \
    --md_path ../neural-ddft-pcm/data/lj_data/ref_3a_langevin_01/avg_2000
python scripts/ddft_neural.py --run_id psm_3d --type c2 --gamma 0.1 --eta 0.6944 \
    --md_path ../neural-ddft-pcm/data/lj_data/ref_3d_langevin_01/avg_2000
```

- On 3a, the result agrees closely with PhiFlow: peak density RMSE vs. MD about 0.057 for both.
- On 3d, with its steep walls and near-empty regions, the solver blows up at t ≈ 2.44τ.
  This confirms that 3d is numerically hard for the PDE itself, not only for PhiFlow.

**Hybrid model** (PCM + learned superadiabatic force): pass a SuperadiabaticNet checkpoint
trained with neural-ddft-pcm, with its path relative to `neural-ddft-pcm/ddft/`, and the
same input-channel flags it was trained with:

```bash
python scripts/ddft_neural.py --run_id psm_3a_hybrid --type c2 --gamma 0.1 --eta 0.6944 \
    --md_path ../neural-ddft-pcm/data/lj_data/ref_3a_langevin_01/avg_2000 \
    --sup_model results/<training_run>/checkpoints/<checkpoint>.ckpt --sup_use_vext
```

The comparison plots of PhiFlow, this solver and py-pde are made with neural-ddft-pcm's
`scripts/evaluation/plot_3solver_error.py` and `compare_3solvers_movie.py`.

## Additional information

- **Runtime** for 3a up to 4τ, single CPU process:
  - PCM-only: about 2.5 min;
  - hybrid with only the `V_ext` input channel: about 10 min.
- **Known limitation:** the history inputs `rho_prev`, `v_prev` and `accel` are not
  well-defined for the adaptive BDF integrator.
  - The integrator evaluates the right-hand side at many trial times per step, and the
    solver approximates these inputs with the last evaluated state.
  - With the full 6-channel correction model, the integration did not converge (stopped after
    1 h 44 min without output).
  - Use correction models without history channels, e.g. `--sup_use_vext` only.
- The initial state is the flat bulk equilibrium without external potential. The potential
  (read from the MD data) is switched on at t = 0, as in the MD simulations.
- Parameters can be given on the command line or as a JSON config (`--config`); every run
  saves its full configuration to `config.json`.

## License and attribution

The original MATLAB code has no public license. See [NOTICE.md](NOTICE.md). The port is
provided for academic reference; please contact the original authors before reuse.
