import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, CubicSpline
from tqdm import tqdm

from goddard_psm.spectral_line import SpectralLine, PeriodicLine

class DDFT_Inertia_1D:
    def __init__(self, optsPhys, optsNum):
        self.kBT = optsPhys['kBT']
        self.m = np.array(optsPhys['mS'])
        self.gamma = np.array(optsPhys['gammaS'])
        self.eta = optsPhys.get('eta', 0.0)
        self.nParticlesS = np.array(optsPhys['nParticlesS'])
        self.nSpecies = len(self.m)
        self.mInv = 1.0 / self.m
        
        self.N = optsNum['N']
        #self.line = SpectralLine(optsNum['yMin'], optsNum['yMax'], self.N)
        self.line = PeriodicLine(optsNum['yMin'], optsNum['yMax'], self.N)
        self.y = self.line.y
        self.Dy = self.line.Dy
        self.D2y = np.dot(self.Dy, self.Dy)
        
        # Define external potential (User-defined or Gaussian as you mentioned)
        self.Vext, self.Vext_grad = self.get_Vext(self.y)

        self.t_range = 1.0

    def get_Vext(self, y):
        """
        Define your Gaussian external potential here.
        Returns: Vext (array), gradient of Vext (array)
        """
        # Example dummy potential; replace with your Gaussian!
        Vext = np.zeros_like(y)
        Vext_grad = np.zeros_like(y)
        return Vext, Vext_grad

    def get_chemical_potential(self, x, rho, t=0.0):
        """
        Calculates the total chemical potential gradient minus the ideal part.
        x is the transformation variable: x = kBT * ln(rho) + Vext
        """
        # ===================================================================
        # NEURAL NETWORK INTEGRATION HERE
        # In MATLAB: mu_s  = getFex(rho_s) + x + ConvRho_s - mu + VAdd;
        # You need to return: NN_Prediction(rho) + x
        # ===================================================================

        # Placeholder for NN: Assuming 0 excess free energy for now (ideal gas)
        dFexc_drho = np.zeros_like(rho)

        # mu_total = F_exc_deriv + x
        mu_total = dFexc_drho + x
        return mu_total

    def get_sup_force(self, rho, v, t=0.0):
        """
        Optional learned superadiabatic correction force, added additively
        to the momentum equation (Equation 2 in dxv_dt) - the Goddard
        analogue of neural-ddft-pcm's hybrid f_sup. Base class has none;
        override in DDFT_Inertia_1D_Neural when a sup_model is provided.
        """
        return 0.0

    def equilibrium_residual(self, x0mu0_flat):
        """ Translates f(x) from the MATLAB fsolve step. """
        mu = x0mu0_flat[:self.nSpecies]
        x = x0mu0_flat[self.nSpecies:].reshape(self.N, self.nSpecies)
        
        rho = np.exp((x - self.Vext[:, None]) / self.kBT)
        
        mu_s = self.get_chemical_potential(x, rho)
        res_mu = mu_s - mu[None, :]
        
        res_norm = np.dot(self.line.Int, rho) - self.nParticlesS
        
        return np.concatenate((res_norm, res_mu.flatten()))

    def compute_equilibrium(self):
        """ Find initial state before turning on dynamics. """
        print("Computing initial condition...")
        # Initial guess (Ideal gas assumption)
        rho_init = np.exp(-self.Vext / self.kBT)[:, None] * np.ones((self.N, self.nSpecies))

        # Calculate the normalization factor (shape: (nSpecies,))
        norm_factor = np.dot(self.line.Int, rho_init) / self.nParticlesS

        # Broadcast the scalar/1D norm_factor to the full (N, nSpecies) grid!
        x_guess = -self.kBT * np.log(norm_factor * np.ones((self.N, self.nSpecies)))
        mu_guess = np.zeros(self.nSpecies)
        
        x0mu0_guess = np.concatenate((mu_guess, x_guess.flatten()))
        
        sol = fsolve(self.equilibrium_residual, x0mu0_guess)
        
        self.mu_eq = sol[:self.nSpecies]
        self.x_eq = sol[self.nSpecies:].reshape(self.N, self.nSpecies)
        self.v_eq = np.zeros_like(self.x_eq)
        print("Equilibrium found.")
        
        return np.concatenate((self.x_eq.flatten(), self.v_eq.flatten()))

    def dxv_dt(self, t, xv_flat):
        """
        Translates dxv_dt ODE formulation.
        Solves for x = kBT * ln(rho) + Vext, rather than rho directly.
        """
        xv = xv_flat.reshape(2 * self.N, self.nSpecies)
        x = xv[:self.N, :]
        v = xv[self.N:, :]
        
        rho = np.exp((x - self.Vext[:, None]) / self.kBT)
        mu_s = self.get_chemical_potential(x, rho, t)
        
        # h_s = Dy * x - grad(Vext)
        h_s = self.Dy.dot(x) - self.Vext_grad[:, None]
        
        # Equation 1: Density evolution (via variable x)
        dxdt = -self.kBT * self.Dy.dot(v) - h_s * v

        # Viscous term: (eta / (m * rho)) * ∇²v
        visc_term = (self.eta / (self.m[None, :] * rho)) * self.D2y.dot(v)

        # Optional learned superadiabatic correction (base class: always 0)
        f_sup = self.get_sup_force(rho, v, t)

        # Equation 2: Velocity evolution (Navier-Stokes-like / Inertial)
        dvdt = (-self.gamma[None, :] * v - v * self.Dy.dot(v) - self.mInv[None, :] * self.Dy.dot(mu_s)
                + visc_term + self.mInv[None, :] * f_sup)
        
        # Ensure boundary conditions (Dirichlet for walls)
        #dvdt[0, :] = 0.0
        #dvdt[-1, :] = 0.0
        
        return np.concatenate((dxdt.flatten(), dvdt.flatten()))

    def solve_dynamics(self, t_span, t_eval, xv_ic=None):
        self.t_range = t_span[1]
        # If no initial condition is provided, compute it naturally
        if xv_ic is None:
            xv_ic = self.compute_equilibrium()
        
        # Calculate initial mass for safety check
        self.initial_mass = np.dot(self.line.Int, np.exp((xv_ic[:self.N] - self.Vext) / self.kBT))
        print(f"Initial Mass: {self.initial_mass:.4f} (Target: {self.nParticlesS[0]:.4f})")

        print("Computing dynamics (BDF stepper)...")
        # --- Progress Bar Setup ---
        pbar = tqdm(total=t_span[1], desc="BDF Integration", bar_format="{l_bar}{bar}| {n:.4f}/{total} [{elapsed}<{remaining}, {rate_fmt}]")
        last_t = [t_span[0]] # Using a list to hold state inside the inner function

        def dxv_dt_wrapped(t, xv_flat):
            # Only update the progress bar if we step forward past our maximum reached time
            if t > last_t[0]:
                pbar.update(t - last_t[0])
                last_t[0] = t
            return self.dxv_dt(t, xv_flat)

        # --- Safety Event ---
        def blow_up_event(t, xv_flat):
            """Triggers if max velocity exceeds 100 or density exceeds 1000."""
            v = xv_flat[self.N:]
            x = xv_flat[:self.N]
            rho = np.exp((x - self.Vext) / self.kBT)
            max_v = np.max(np.abs(v))
            max_rho = np.max(rho)
            return 100.0 - max(max_v, max_rho / 10.0)
        
        blow_up_event.terminal = True 
        blow_up_event.direction = -1  

        sol = solve_ivp(
            dxv_dt_wrapped, t_span, xv_ic, method='BDF', t_eval=t_eval, 
            rtol=1e-5, atol=1e-5, events=blow_up_event
        )
        pbar.close() # Clean up the progress bar

        if sol.status == 1:
            print(f"INTEGRATION ABORTED AT t={sol.t[-1]:.3f}: Simulation blew up!")
        elif sol.status == 0:
            print("Integration completed successfully.")
        else:
            print(f"Integration stopped: {sol.message}")

        return sol


class DDFT_Inertia_1D_Neural(DDFT_Inertia_1D):
    def __init__(self, optsPhys, optsNum, nn_model, uniform_r, sup_model=None):
        super().__init__(optsPhys, optsNum)
        self.nn_model = nn_model
        self.uniform_r = uniform_r
        self.dz = uniform_r[1] - uniform_r[0]
        # Optional learned superadiabatic correction network (hybrid mode) -
        # mirrors neural-ddft-pcm's force_factory.py:get_hybrid_force, see
        # predict_f_sup/get_sup_force below. None (default) = pure
        # equilibrium dynamics, identical to this class's original behavior.
        self.sup_model = sup_model
        # Simple same-object cache for rho_prev/v_prev (on the UNIFORM grid,
        # matching sup_model's native convention) - this is a stateless ODE
        # RHS called repeatedly by solve_ivp with arbitrary (possibly
        # out-of-order, due to adaptive-step function evaluations) t, so
        # "previous step" has no precise meaning here; same None->self
        # fallback convention as force_factory.py's own rho_prev_grid=None
        # case - only used if sup_model.use_rho_prev/use_v_prev/use_accel.
        self._prev_rho_uniform = None
        self._prev_v_uniform = None

    def get_alpha(self, t, t_range):
        """
        Smoothly decays the NN multiplier from a starting boost to a final reduction.
        """
        alpha_start = 1.3  # 130% strength at t=0
        alpha_end = 0.6    # 60% strength at the end
        alpha_end = 0.8    # 80% strength at the end
        
        # tau controls how fast it decays (t_range/4 means it decays mostly in the first half)
        tau = t_range / 4.0 
        
        return 1
        #return alpha_end + (alpha_start - alpha_end) * np.exp(-t / tau) # exponential decay
        #return alpha_start + (alpha_end - alpha_start)*(t/t_range) + np.sin(t/t_range *2*np.pi)*(1- t/t_range) # sin with linear decaying amplitude (1 to 0) + linear interpolation from alpha_start to alpha_end
        #return alpha_start + (alpha_end - alpha_start)*(t/t_range) + np.sin(t/t_range *5/2*np.pi)*(1- t/t_range) # sin with linear decaying amplitude (1 to 0) + linear interpolation from alpha_start to alpha_end
        
    def predict_dFdrho(self, rho_cheb):
        """
        Takes density on the Chebyshev/Fourier grid, interpolates to uniform, 
        runs the NN, and returns both uniform and Chebyshev/Fourier dFdrho.
        """
        # 1. Interpolate to Uniform (Clamp at boundaries, no cubic extrapolation!
        #interp_func_to_uniform = interp1d(self.y, rho_cheb, kind='linear', fill_value=(rho_cheb[0], rho_cheb[-1]))
        #rho_uniform = interp_func_to_uniform(self.uniform_r)

        # 1. Fourier -> Uniform (Padding the Fourier grid)
        # Append the left boundary value to the right side to close the periodic loop
        rho_fourier = rho_cheb  # better naming
        y_padded = np.append(self.y, self.line.yMax)
        rho_padded = np.append(rho_fourier, rho_fourier[0])
        #y_padded = np.concatenate((self.y[-3:] - self.line.L, self.y, self.y[:3] + self.line.L))
        #rho_padded = np.concatenate((rho_fourier[-3:], rho_fourier, rho_fourier[:3]))
        
        # Now we can safely use cubic interpolation over the completely closed domain
        #interp_to_uniform = interp1d(
        #    y_padded, rho_padded, kind='cubic', 
        #    bounds_error=False, fill_value="extrapolate"
        #)
        #rho_uniform = interp_to_uniform(self.uniform_r)
        spline_to_uniform = CubicSpline(y_padded, rho_padded, bc_type='periodic')
        rho_uniform = spline_to_uniform(self.uniform_r)
        rho_uniform = np.maximum(rho_uniform, 1e-6)
        
        # 2. Evaluate PyTorch Neural Network
        rho_tensor = torch.tensor(rho_uniform).view(1, 1, -1).float().requires_grad_(True)
        
        torch.set_grad_enabled(True)
        F_pred = self.nn_model(rho_tensor)
        dFdrho_tensor = torch.autograd.grad(F_pred, rho_tensor)[0] / self.dz
        torch.set_grad_enabled(False)
        
        dFdrho_uniform = dFdrho_tensor.squeeze().detach().numpy()
        
        # 3. Interpolate dFdrho back to the Chebyshev grid for the PDE solver
        #interp_func_to_cheb = interp1d(self.uniform_r, dFdrho_uniform, kind='cubic', fill_value="extrapolate")
        #dFdrho_cheb = interp_func_to_cheb(self.y)
        
        #return dFdrho_uniform, dFdrho_cheb

        # 3. Uniform -> Fourier (Tripple Padding the Uniform PyTorch grid)
        # ---------------------------------------------------------
        # MD uniform grids (cell centers) also leave gaps at the edges. 
        # We wrap the last point to the left, and the first point to the right!
        r_padded = np.concatenate(([self.uniform_r[-1] - self.line.L], self.uniform_r, [self.uniform_r[0] + self.line.L]))
        dFdrho_padded = np.concatenate(([dFdrho_uniform[-1]], dFdrho_uniform, [dFdrho_uniform[0]]))
        #r_padded = np.concatenate((self.uniform_r[-3:] - self.line.L, self.uniform_r, self.uniform_r[:3] + self.line.L))
        #dFdrho_padded = np.concatenate((dFdrho_uniform[-3:], dFdrho_uniform, dFdrho_uniform[:3]))

        # Fix duplication errors
        r_unique, unique_idx = np.unique(r_padded, return_index=True)
        dFdrho_unique = dFdrho_padded[unique_idx]
        
        #interp_to_fourier = interp1d(
        #    r_unique, dFdrho_unique, kind='cubic', 
        #    bounds_error=False, fill_value="extrapolate"
        #)
        #dFdrho_fourier = interp_to_fourier(self.y)
        spline_to_fourier = CubicSpline(r_unique, dFdrho_unique)
        dFdrho_fourier = spline_to_fourier(self.y)

        return dFdrho_uniform, dFdrho_fourier

    def _cheb_to_uniform(self, field_cheb):
        """Periodic-spline interpolation from this solver's own Fourier grid
        to the uniform MD grid - same convention as predict_dFdrho's step 1."""
        y_padded = np.append(self.y, self.line.yMax)
        field_padded = np.append(field_cheb, field_cheb[0])
        spline = CubicSpline(y_padded, field_padded, bc_type='periodic')
        return spline(self.uniform_r)

    def predict_f_sup(self, rho_cheb, v_cheb, t=0.0):
        """
        Mirrors predict_dFdrho's interpolation pattern, but evaluates the
        learned superadiabatic correction network (sup_model) instead of
        the equilibrium free-energy model, directly in its default
        (non-potential-output) force-field mode - same channel-building
        convention as neural-ddft-pcm's force_factory.py:get_hybrid_force.
        Returns f_sup interpolated back onto this solver's own Fourier grid.
        """
        rho_uniform = np.maximum(self._cheb_to_uniform(rho_cheb), 1e-6)
        v_uniform = self._cheb_to_uniform(v_cheb)

        rho_t = torch.tensor(rho_uniform).view(1, -1).float()
        v_t = torch.tensor(v_uniform).view(1, -1).float()
        grad_rho_t = (torch.roll(rho_t, -1, dims=-1) - torch.roll(rho_t, 1, dims=-1)) / (2 * self.dz)
        grad_v_t = (torch.roll(v_t, -1, dims=-1) - torch.roll(v_t, 1, dims=-1)) / (2 * self.dz)

        extra_kwargs = {}
        if getattr(self.sup_model, "use_vext", False):
            vext_uniform = self._cheb_to_uniform(self.Vext)  # self.Vext is 1D (N,), unlike rho/v's (N, nSpecies)
            extra_kwargs["v_ext"] = torch.tensor(vext_uniform).view(1, -1).float()
        needs_rho_prev = getattr(self.sup_model, "use_rho_prev", False)
        needs_v_prev = getattr(self.sup_model, "use_v_prev", False)
        needs_accel = getattr(self.sup_model, "use_accel", False)
        if needs_rho_prev or needs_accel:
            prev = self._prev_rho_uniform if self._prev_rho_uniform is not None else rho_uniform
            rho_prev_t = torch.tensor(prev).view(1, -1).float()
            if needs_rho_prev:
                extra_kwargs["rho_prev"] = rho_prev_t
        if needs_v_prev or needs_accel:
            prev = self._prev_v_uniform if self._prev_v_uniform is not None else v_uniform
            v_prev_t = torch.tensor(prev).view(1, -1).float()
            if needs_v_prev:
                extra_kwargs["v_prev"] = v_prev_t
        if needs_accel:
            # No real dt available in this stateless RHS callback (solve_ivp
            # calls it at arbitrary adaptive substeps) - fall back to the
            # outer integration's nominal dt, matching force_factory.py's
            # own "accel requires dt" requirement as closely as this
            # solver's calling convention allows.
            dt_for_accel = getattr(self, "_accel_dt_hint", 1e-4)
            extra_kwargs["accel"] = (v_t - v_prev_t) / dt_for_accel
        if getattr(self.sup_model, "use_gamma", False):
            extra_kwargs["gamma"] = torch.full_like(rho_t, float(self.gamma[0]))
        if getattr(self.sup_model, "use_eta", False):
            extra_kwargs["eta"] = torch.full_like(rho_t, float(self.eta))

        with torch.no_grad():
            f_sup_uniform = self.sup_model(rho_t, v_t, grad_rho_t, grad_v_t, **extra_kwargs).squeeze(0).numpy()

        self._prev_rho_uniform, self._prev_v_uniform = rho_uniform, v_uniform

        r_padded = np.concatenate(([self.uniform_r[-1] - self.line.L], self.uniform_r, [self.uniform_r[0] + self.line.L]))
        f_sup_padded = np.concatenate(([f_sup_uniform[-1]], f_sup_uniform, [f_sup_uniform[0]]))
        r_unique, unique_idx = np.unique(r_padded, return_index=True)
        spline_to_fourier = CubicSpline(r_unique, f_sup_padded[unique_idx])
        return spline_to_fourier(self.y)

    def get_sup_force(self, rho, v, t=0.0):
        if self.sup_model is None:
            return 0.0
        f_sup_fourier = self.predict_f_sup(rho[:, 0], v[:, 0], t)
        return f_sup_fourier[:, None]

    def get_chemical_potential(self, x, rho, t=0.0):
        # Call our new helper method!
        _, dFdrho_cheb = self.predict_dFdrho(rho[:, 0])
        
        # Note: If you have an alpha scheduling function, apply it here:
        alpha = self.get_alpha(t, self.t_range)
        dFdrho_cheb *= alpha
        
        # mu_total = F_exc_deriv + x
        mu_total = dFdrho_cheb[:, None] + x
        return mu_total