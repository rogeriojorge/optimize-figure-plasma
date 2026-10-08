# Optimize a picture through plasma dynamics

Find initial conditions whose **evolved** density or phase-space distribution draws an image. Six editable Python examples differentiate through their simulations with JAX.

| Model | Image coordinates | Snapshot objective | Shape-retention objective |
|---|---|---|---|
| Compressible isothermal Euler | Two spatial coordinates | `snapshot/euler.py` | `retention/euler.py` |
| SPECTRAX Vlasov | Position and velocity (1D1V) | `snapshot/vlasov.py` | `retention/vlasov.py` |
| JAX-in-Cell PIC | Position and velocity (1D1V) | `snapshot/pic.py` | `retention/pic.py` |

## Run

```sh
git clone https://github.com/rogeriojorge/optimize-figure-plasma.git
cd optimize-figure-plasma
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python snapshot/euler.py
python retention/euler.py
python snapshot/vlasov.py
python retention/vlasov.py
python snapshot/pic.py
python retention/pic.py
```

Use Python 3.11 or newer. Edit the input parameters near the top of a script, including `IMAGE` for your own JPG or PNG. There are no command-line arguments or main functions. Each script announces compilation, reports every optimizer iteration, and saves plots and arrays under `results/`. Install an appropriate JAX GPU wheel first if desired; the defaults also run on CPU. JAX, the optimizer, and upstream solver revisions are pinned in `requirements.txt`.

## What is optimized?

The snapshot examples minimize pixelwise mean squared error at a prescribed final time. The retention examples compare the initial image and many times along the trajectory, so an isolated good final snapshot is insufficient. The duration is fixed: increasing it makes retention harder, rather than allowing the optimizer to shorten the simulation. Only initial conditions are optimized; the equations continue to evolve them.

Images are converted to inverted grayscale (dark artwork becomes high density), resized while preserving aspect ratio, padded to the simulation grid, given a small positive floor, and normalized. This is a scalar-density demonstration: RGB color is not a physical variable. The supplied W7-X illustration is a target picture, not a simulated W7-X equilibrium.

Euler uses a conservative periodic finite-volume discretization with isothermal pressure, Rusanov fluxes, and SSP-RK2 time integration. The kinetic scripts call [SPECTRAX](https://github.com/uwplasma/SPECTRAX) and [JAX-in-Cell](https://github.com/uwplasma/JAX-in-Cell). Their two image axes represent one position and one velocity, as in a phase-space plot.

Each run saves `comparison.png`, `loss.png`, `trajectory.gif`, and `results.npz`. The archive contains the target, optimized initial image, all saved frames, actual simulation times, pixel errors, image mass, and the optimized initial state. `parameters_0` stores Euler’s `(rho, rho*u, rho*v)` array or SPECTRAX’s complex Hermite–Fourier coefficients. For PIC, `parameters_0` and `parameters_1` store full particle positions and velocities (electrons first, then ions).

`helpers.py` contains shared image preparation, optimization, plotting, the Euler kernel, and the two upstream solver adapters. Keep the example inputs in their scripts; increase resolution, integration steps, and optimization iterations together when refining a result.

## Image credit and license

`W7X-Spulen_Plasma_blau_gelb.jpg`: Max-Planck-Institut für Plasmaphysik, [Wikimedia Commons source](https://commons.wikimedia.org/wiki/File:W7X-Spulen_Plasma_blau_gelb.jpg), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The included image is the 330 × 205 version; simulations resize and convert it to grayscale. This image has its own license. The code is MIT licensed.
