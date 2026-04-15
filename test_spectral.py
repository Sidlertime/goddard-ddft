import numpy as np
from spectral_utils import cheb_grid_diff, clencurt_weights
from spectral_line import SpectralLine

def test_differentiation_exactness():
    """Spectral methods should differentiate polynomials exactly."""
    N = 50
    line = SpectralLine(yMin=0.0, yMax=10.0, N=N)
    
    # Test function: f(y) = y^3 - 2y^2 + 5
    # Exact derivative: f'(y) = 3y^2 - 4y
    f_y = line.y**3 - 2 * line.y**2 + 5
    exact_dy = 3 * line.y**2 - 4 * line.y
    
    # Spectral derivative
    spectral_dy = line.Dy.dot(f_y)
    
    assert np.allclose(spectral_dy, exact_dy, atol=1e-10), "Differentiation matrix failed!"
    print("Differentiation exactness test passed.")

def test_integration_exactness():
    """Test Clenshaw-Curtis integration on a known function."""
    N = 50
    line = SpectralLine(yMin=0.0, yMax=np.pi, N=N)
    
    # Integral of sin(y) from 0 to pi is exactly 2.0
    f_y = np.sin(line.y)
    integral = np.dot(line.Int, f_y)
    
    assert np.isclose(integral, 2.0, atol=1e-10), "Integration weights failed!"
    print("Integration exactness test passed.")

if __name__ == "__main__":
    test_differentiation_exactness()
    test_integration_exactness()