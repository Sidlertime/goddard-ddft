import numpy as np
from spectral_utils import cheb_grid_diff, clencurt_weights
from spectral_utils import fourier_grid_diff

class PeriodicLine:
    def __init__(self, yMin, yMax, N):
        self.yMin = yMin
        self.yMax = yMax
        self.N = N
        self.L = yMax - yMin
        
        # Base Fourier grid [0, 2pi)
        self.x_base, self.Dx_base = fourier_grid_diff(N)
        self.w_base = np.ones(N) * 2 * np.pi / N
        
        # Linear map to physical space [yMin, yMax)
        scale = self.L / (2 * np.pi)
        
        self.y = scale * self.x_base + yMin
        self.Dy = self.Dx_base / scale       # Scaled differentiation matrix
        self.Int = self.w_base * scale       # Scaled integration weights

class SpectralLine:
    def __init__(self, yMin, yMax, N):
        self.yMin = yMin
        self.yMax = yMax
        self.N = N
        
        # Base Chebyshev grid and differentiation
        self.x_base, self.Dx_base = cheb_grid_diff(N)
        self.w_base = clencurt_weights(N)
        
        # Linear map to physical space [yMin, yMax]
        scale = 0.5 * (yMax - yMin)
        shift = 0.5 * (yMax + yMin)
        
        self.y = scale * self.x_base + shift
        self.Dy = self.Dx_base / scale       # Scaled differentiation matrix
        self.Int = self.w_base * scale       # Scaled integration weights