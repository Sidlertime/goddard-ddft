# Notice and attribution

`goddard_psm` is a Python port of the MATLAB pseudo-spectral DDFT solver by
**Ben Goddard** (University of Edinburgh), from the 2DChebClass framework:

> A. Nold, B. D. Goddard, P. Yatsyshin, N. Savva, S. Kalliadasis,
> *Pseudospectral methods for density functional theory in bounded and unbounded domains*,
> J. Comput. Phys. 334, 639–664 (2017).

The original code has no public license, so all rights to the original algorithm and code
remain with its authors. The port and its extensions are by Jean-Pasqual Sindermann
(2026):
- periodic Fourier line;
- inertial (underdamped) DDFT with a neural PCM functional;
- optional learned superadiabatic correction from
  [neural-ddft-pcm](https://github.com/Sidlertime/neural-ddft-pcm).

This port is provided for academic reference only. No license is granted; please
contact the original authors before reuse.

Parts of the code and documentation were written with the help of AI assistants
(Anthropic Claude, Google Gemini).
