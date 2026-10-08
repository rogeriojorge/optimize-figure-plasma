# Optimize an image through plasma dynamics

**Differentiate through a simulation, then change its initial condition to draw your image.** Six short, editable examples use JAX autodiff: compressible isothermal Euler in two spatial dimensions, SPECTRAX Vlasov in 1D1V, and JAX-in-Cell particle-in-cell phase space.

| Model | Match a final snapshot | Retain the shape over time |
|---|---|---|
| 2D isothermal Euler | [snapshot/euler.py](snapshot/euler.py) | [retention/euler.py](retention/euler.py) |
| 1D1V Vlasov–Maxwell | [snapshot/vlasov.py](snapshot/vlasov.py) | [retention/vlasov.py](retention/vlasov.py) |
| Electrostatic PIC | [snapshot/pic.py](snapshot/pic.py) | [retention/pic.py](retention/pic.py) |

## Watch the optimized dynamics

These GIF previews loop automatically in GitHub's README. The corresponding **1080p H.264 MP4s** are suitable for inserting into PowerPoint. Movies contain the evolving density/distribution and, for both plasma models, the electric field. Each movie uses actual saved solver states, a fixed color scale, and a simulation clock; the Euler snapshot displays density minus its unit background on a labeled symmetric-log scale; objective histories stay in separate figures.

### Euler · final snapshot

![Optimized Euler density evolving toward the target](media/snapshot_euler.gif)

![Euler initial and final density](media/snapshot_euler_initial_final.png)

[PowerPoint movie](media/snapshot_euler.mp4) · [Optimized initial velocity](media/snapshot_euler_initial_velocity.png) · [Target and baseline comparison](media/snapshot_euler_comparison.png) · [Loss history](media/snapshot_euler_loss.png)

### Euler · shape retention

![Euler density optimized to retain the image](media/retention_euler.gif)

![Euler retention initial and final density](media/retention_euler_initial_final.png)

[PowerPoint movie](media/retention_euler.mp4) · [Target and baseline comparison](media/retention_euler_comparison.png) · [Loss history](media/retention_euler_loss.png)

The longer plasma configurations are being refined; their phase-space/electric-field media will be added with the measured results.

## Run with your own image

Use Python 3.11 or newer:

```sh
git clone https://github.com/rogeriojorge/optimize-figure-plasma.git
cd optimize-figure-plasma
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python snapshot/euler.py
# Or any of the other five scripts in the table above.
```

Edit `IMAGE` and the input parameters at the top of the chosen script. JPG and PNG are supported, including PNG transparency. There are no parsed arguments or main functions: imports, inputs, simulation/optimization, and saved results are visible in order. Every optimizer iteration reports its objective and elapsed time. First-call JAX compilation takes longer than a warm iteration. A bundled FFmpeg binary creates the movies; no separate FFmpeg installation is required. CPU defaults are modest; install an appropriate JAX accelerator wheel to use a GPU.

Image preprocessing preserves aspect ratio, pads the canvas, inverts grayscale, and normalizes the mean. Dark artwork becomes high density. The Euler snapshot places an editable image contrast on a unit-density background; its initial density is held exactly uniform and only the initial velocity is optimized. Euler retention and PIC add a small density floor; Vlasov adds the image as a small perturbation of a positive Maxwellian background and displays `f − background` with an explicitly labeled scale. Colors are display choices rather than three independently simulated channels. The supplied W7-X picture is artwork to reproduce, rather than a W7-X equilibrium.

## Objective and autodiff

For a snapshot, minimize `mean((f(T) - target)**2)`. For retention, penalize disagreement at the initial time and multiple times over a fixed horizon. Retention cannot win by shortening the simulation. Only initial conditions change; all subsequent states follow the solver. Initial fidelity, negative spectral values, and particles leaving the velocity viewport receive additional penalties where appropriate.

Every simulation and objective gradient uses JAX. The shared optimizer defaults to Optax Adam; `method="lbfgs"` selects Optax L-BFGS, and `method="l-bfgs-b"` or `method="bfgs"` selects SciPy with a JIT-compiled **JAX value and gradient**. The SciPy alternatives still differentiate through the simulation, rather than estimate derivatives with finite differences. Choose an optimizer by changing one editable call in a script.

Euler advances conservative `(rho, rho*u, rho*v)` with periodic Rusanov fluxes and isothermal pressure. The snapshot uses a second-order MUSCL–Hancock predictor to create the image from uniform density; retention uses SSP-RK2 with first-order face states. Checkpointed JAX scans carry the gradient through time. Density is positive and initial velocities are bounded. SPECTRAX evolves electron and heavy-ion Hermite–Fourier coefficients with self-consistent fields; the rendered distribution is the transverse-velocity marginal. Its collisional Hermite relaxation damps unresolved high moments. Initially neutral electron/ion density profiles and a positive Maxwellian background permit longer runs with a small image perturbation; the simulation evolves the full distribution and electric field. PIC calls JAX-in-Cell's electrostatic field solver and relativistic Boris pusher with electrons and ions. Its image is a differentiable kernel deposition in `(x, vx)`; initial velocities use a smooth subluminal parameterization. The upstream PIC pusher has three velocity components, while the displayed and optimized image uses one position and one velocity.

A long self-consistent plasma trajectory can mix an arbitrary image beyond recognition. The initial/final figures and baseline comparisons show that tradeoff, including imperfect fits. Spectral reconstruction can leave small negative Gibbs residues; increasing resolution and checking the velocity tails is preferable to silently clipping the distribution. Fixed movie color scales and fixed PIC deposition normalization avoid concealing evolving amplitudes or particles leaving the viewport.

## Outputs and numerical checks

Each script writes `results/<mode>_<model>/`:

- `trajectory.mp4` and `trajectory.gif`: dynamics with electric fields for plasma cases.
- `initial_final.png`: the optimized initial and final states, including plasma electric fields.
- `comparison.png`: target, optimized initial, baseline final, and optimized final.
- `loss.png`: objective versus optimization iteration.
- `results.npz`: target, optimized initial, saved trajectory, baseline, times, objective history, per-frame errors, represented mass, fields, background, image contrast, normalized signal errors, and optimized initial-state arrays.

`parameters_0` is Euler's conservative state or SPECTRAX's complex Hermite–Fourier coefficients. PIC stores full initial particle positions and velocities as `parameters_0` and `parameters_1` (electrons first, then ions). `time_scale` converts saved physical times to the displayed normalized clock. Movies subsample long trajectories without interpolating artificial density states.

The Euler examples were run end to end on CPU, with directional autodiff checks and conservation/time-step checks. PIC snapshot optimization now spans 70.7 inverse plasma frequencies; the longer plasma retention configurations are still being refined. Upstream revisions and JAX dependencies are pinned in [requirements.txt](requirements.txt). Increase resolution, integration steps, and optimization iterations together when refining a result; the runtime target is a warm objective/gradient evaluation under ten seconds, rather than a hardware-independent guarantee.

## Credits and license

Solvers: [SPECTRAX](https://github.com/uwplasma/SPECTRAX) and [JAX-in-Cell](https://github.com/uwplasma/JAX-in-Cell), with examples written in the simple editable-input style of UWPlasma and [Simsopt](https://github.com/hiddenSymmetries/simsopt). Shared code is in [helpers.py](helpers.py); numerical Euler integration lives there too.

`W7X-Spulen_Plasma_blau_gelb.jpg`: Max-Planck-Institut für Plasmaphysik, [Wikimedia Commons source](https://commons.wikimedia.org/wiki/File:W7X-Spulen_Plasma_blau_gelb.jpg), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The included image is the 330 × 205 version; plots and movies are scalar-density and false-color transformations with this attribution. The code is GPL-3.0 licensed because the second-order Euler implementation is adapted from Philip Mocz’s [finitevolume-jax](https://github.com/pmocz/finitevolume-jax) (2024). The image retains its separate license.
