import numpy as np

def cheb_grid_diff(N):
    """
    Translates barychebdiff.m
    Generates N Chebyshev points in [-1, 1] and the differentiation matrix D.
    """
    if N <= 1:
        return np.array([0.0]), np.array([[0.0]])
    
    # Chebyshev points
    theta = np.pi * np.arange(N) / (N - 1)
    x = np.cos(theta)
    x = x[::-1] # Flip to strictly increasing order, matching Goddard's MATLAB logic
    
    # Barycentric weights
    w = np.ones(N) * (-1)**np.arange(N)
    w[0] *= 0.5
    w[-1] *= 0.5
    
    # Construct differentiation matrix D
    Dw = w[None, :] / w[:, None] - np.eye(N)
    Dx = x[:, None] - x[None, :] + np.eye(N)
    D = Dw / Dx
    
    # Fix the diagonal elements to be the negative sum of the rows
    np.fill_diagonal(D, 0)
    d_diag = -np.sum(D, axis=1)
    np.fill_diagonal(D, d_diag)
    
    return x, D

def fourier_grid_diff(N):
    """
    Generates N equispaced points in [0, 2*pi) and the Fourier differentiation matrix D.
    Note: N must be an even number.
    """
    if N % 2 != 0:
        raise ValueError("N must be even for this Fourier matrix formulation.")
    
    x = 2 * np.pi * np.arange(N) / N
    
    # Construct the Fourier differentiation matrix D
    column = np.zeros(N)
    column[1:] = 0.5 * (-1.0)**np.arange(1, N) / np.tan(np.arange(1, N) * np.pi / N)
    
    # D is a Toeplitz matrix where D_ij = column[i-j]
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    D = column[(i - j) % N]
    
    return x, D

def clencurt_weights(N):
    """Computes the Clenshaw-Curtis integration weights for domain [-1, 1]."""
    if N == 1: return np.array([2.0])
    theta = np.pi * np.arange(N) / (N - 1)
    w = np.zeros(N)
    ii = np.arange(1, N - 1)
    v = np.ones(N - 2)
    
    if (N - 1) % 2 == 0:
        w[0] = 1.0 / ((N - 1)**2 - 1)
        w[-1] = w[0]
        for k in range(1, (N - 1) // 2):
            v -= 2.0 * np.cos(2.0 * k * theta[ii]) / (4 * k**2 - 1)
        v -= np.cos((N - 1) * theta[ii]) / ((N - 1)**2 - 1)
    else:
        w[0] = 1.0 / (N - 1)**2
        w[-1] = w[0]
        for k in range(1, (N - 1) // 2 + 1):
            v -= 2.0 * np.cos(2.0 * k * theta[ii]) / (4 * k**2 - 1)
            
    w[ii] = 2.0 * v / (N - 1)
    return w[::-1] # Match flipped x grid