# Optimize an image through plasma dynamics

**Differentiate through a simulation, then change its initial condition to draw your image.** Six short, editable examples use JAX autodiff: compressible isothermal Euler in two spatial dimensions, SPECTRAX Vlasov in 1D1V, and JAX-in-Cell particle-in-cell phase space.

| Model | Match a final snapshot | Retain the shape over time |
|---|---|---|
| 2D isothermal Euler | [snapshot/euler.py](snapshot/euler.py) | [retention/euler.py](retention/euler.py) |
| 1D1V Vlasov–Maxwell | [snapshot/vlasov.py](snapshot/vlasov.py) | [retention/vlasov.py](retention/vlasov.py) |
| Electrostatic PIC | [snapshot/pic.py](snapshot/pic.py) | [retention/pic.py](retention/pic.py) |

## Watch the optimized dynamics

The Euler snapshot uses 128 × 128 cells; its retention preview remains the previous 96 × 96 validation. The Vlasov snapshot below uses 160 spatial points and 32 Hermite modes, reconstructed on 320 velocity samples. A scan of higher Hermite resolution and finer PIC sampling is in progress; the snapshot script now tests a 96-point, 64-Hermite candidate. Its completed movie will replace this checkpoint after long-horizon validation.

These six-second GIF previews loop automatically in GitHub's README. The corresponding **six-second, 1080p H.264 MP4s** are suitable for inserting into PowerPoint. Movies show only the evolving Euler density or plasma phase-space distribution. Electric fields are saved as separate figures. Each movie uses actual saved solver states, a fixed color scale, and a simulation clock; the Euler snapshot displays density minus its unit background on a labeled symmetric-log scale; objective histories stay in separate figures.

### Euler · final snapshot

![Optimized Euler density evolving toward the target](media/snapshot_euler.gif)

![Euler initial and final density](media/snapshot_euler_initial_final.png)

Same initial condition, evolved to 1.5T:

![Euler dynamics through 1.5T](media/snapshot_euler_extended.gif)

[Extended PowerPoint movie](media/snapshot_euler_extended.mp4) · [Initial and extended final state](media/snapshot_euler_extended_initial_final.png)

[PowerPoint movie](media/snapshot_euler.mp4) · [Optimized initial velocity](media/snapshot_euler_initial_velocity.png) · [Target and baseline comparison](media/snapshot_euler_comparison.png) · [Loss history](media/snapshot_euler_loss.png)

### Euler · shape retention

![Euler density optimized to retain the image](media/retention_euler.gif)

![Euler retention initial and final density](media/retention_euler_initial_final.png)

[PowerPoint movie](media/retention_euler.mp4) · [Target and baseline comparison](media/retention_euler_comparison.png) · [Loss history](media/retention_euler_loss.png)

### Vlasov · final snapshot

![Optimized Vlasov phase space](media/snapshot_vlasov.gif)

![Vlasov initial and final phase space](media/snapshot_vlasov_initial_final.png)

Same initial condition, evolved to 1.5T:

![Vlasov dynamics through 1.5T](media/snapshot_vlasov_extended.gif)

[Extended PowerPoint movie](media/snapshot_vlasov_extended.mp4) · [Initial and extended final state](media/snapshot_vlasov_extended_initial_final.png)

[PowerPoint movie](media/snapshot_vlasov.mp4) · [Electric field](media/snapshot_vlasov_electric_field.png) · [Density](media/snapshot_vlasov_density.png) · [Mean velocity](media/snapshot_vlasov_velocity.png) · [Target and baseline comparison](media/snapshot_vlasov_comparison.png) · [Loss history](media/snapshot_vlasov_loss.png)

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

Image preprocessing preserves aspect ratio, pads the canvas, inverts grayscale, and normalizes the mean. Dark artwork becomes high density. The Euler snapshot places an editable image contrast on a unit-density background; its initial density is held exactly uniform and only the initial velocity is optimized. Euler retention and PIC add a small density floor; Vlasov adds the image as a small perturbation of a positive Maxwellian background and displays `f − background` with an explicitly labeled scale. All density and phase-space images use jet with fixed color limits. Vlasov perturbations use a symmetric-log scale; PIC uses a linear scale. Plasma movies have no opacity cutoff. Euler retains its display-only cutoff near 5% of the target peak; it never alters dynamics, objectives, or saved physical arrays. Colors are display choices rather than three independently simulated channels. The supplied W7-X picture is artwork to reproduce, rather than a W7-X equilibrium.

## Resolution and duration

The numerical grids are independent of the input image's pixel dimensions. Change the input grid parameters to resample any JPG or PNG while preserving its proportions; increasing movie resolution alone does not increase solver accuracy.

| Model | Numerical resolution | Integration horizon |
|---|---|---|
| Euler snapshot | 128 × 128 cells, 352 final-horizon steps | 0.43 sound-crossing times of the unit box |
| Euler retention | 128 × 128 cells, 192 final-horizon steps | `t = 0.5`, with sound speed 0.45 |
| Vlasov snapshot / retention | 96 / 160 spatial Fourier points, 64 / 32 Hermite modes, 320 displayed velocity bins | `ωₚₑ t = 60`, domain length 250 |
| PIC | 64 field cells, 4,096 electrons and 4,096 ions, 256 × 256 phase-space bins | 2,000 snapshot / 2,400 retention steps |

Vlasov velocity bins sample a Hermite expansion: increasing `MODES` raises velocity-space solver resolution, while `VELOCITY_BINS` raises reconstruction resolution. PIC phase-space bins are separate from the field grid and particle count. Its Gaussian deposition spans three phase-space bins to reduce sampling noise; resolving smaller kinetic features requires more particles too. Euler's published grids contain about seven times as many cells as the initial 48 × 48 trials.

Each example optimizes short horizons first, then uses the optimized initial condition as the starting guess for a longer-horizon optimization. The simulation still starts at time zero at each stage. This continuation helps avoid tackling the longest, most sensitive trajectory immediately. Euler increases its integration step count with each horizon; Vlasov keeps a fixed time step; PIC increases the number of time steps. Loss plots separate the stages because each horizon defines a different objective.

## Objective and autodiff

For a snapshot, minimize `mean((f(T) - target)**2)`. For retention, penalize disagreement at the initial time and multiple times over a fixed horizon. Retention cannot win by shortening the simulation. Only initial conditions change; all subsequent states follow the solver. Initial fidelity, negative spectral values, and particles leaving the velocity viewport receive additional penalties where appropriate.

Every simulation and objective gradient uses JAX. The shared optimizer defaults to Optax Adam; `method="lbfgs"` selects Optax L-BFGS, and `method="l-bfgs-b"` or `method="bfgs"` selects SciPy with a JIT-compiled **JAX value and gradient**. The SciPy alternatives still differentiate through the simulation, rather than estimate derivatives with finite differences. Choose an optimizer by changing one editable call in a script.

Euler advances conservative `(rho, rho*u, rho*v)` with periodic Rusanov fluxes and isothermal pressure. The snapshot uses a second-order MUSCL–Hancock predictor to create the image from uniform density; retention uses SSP-RK2 with first-order face states. Checkpointed JAX scans carry the gradient through time. Density is positive and initial velocities are bounded. SPECTRAX evolves electron and heavy-ion Hermite–Fourier coefficients with self-consistent fields; the rendered distribution is the transverse-velocity marginal. Its collisional Hermite relaxation damps unresolved high moments. Initially neutral electron/ion density profiles and a positive Maxwellian background permit longer runs with a small image perturbation; the simulation evolves the full distribution and electric field. PIC calls JAX-in-Cell's electrostatic field solver and relativistic Boris pusher with electrons and ions. Its image is a differentiable kernel deposition in `(x, vx)`; initial velocities use a smooth subluminal parameterization. The upstream PIC pusher has three velocity components, while the displayed and optimized image uses one position and one velocity.

A long self-consistent plasma trajectory can mix an arbitrary image beyond recognition. The initial/final figures and baseline comparisons show that tradeoff, including imperfect fits. Spectral reconstruction can leave small negative Gibbs residues; increasing resolution and checking the velocity tails is preferable to silently clipping the distribution. Fixed movie color scales and fixed PIC deposition normalization avoid concealing evolving amplitudes or particles leaving the viewport.

## Outputs and numerical checks

Each script writes `results/<mode>_<model>/`:

- `trajectory.mp4` and `trajectory.gif`: density or phase-space dynamics through the optimization time T.
- `trajectory_extended.mp4` and `trajectory_extended.gif`: the same optimized initial condition evolved to 1.5T. Both versions last six seconds and share color limits; the extension receives no further optimization.
- `extended_initial_final.png`: initial state and final state at 1.5T.
- `initial_final.png`: the optimized initial and final density or phase-space states.
- `electric_field.png`: separate initial/final field profiles and a field history for plasma cases.
- `density.png` and `velocity.png`: separate velocity-window density and mean-velocity profiles for plasma cases.
- `comparison.png`: target, optimized initial, baseline final, and optimized final.
- `loss.png`: objective versus optimization iteration.
- `results.npz`: target, optimized initial, saved trajectory, baseline, times, objective history, per-frame errors, represented mass, fields, background, image contrast, normalized signal errors, horizon-stage histories, display settings, and optimized initial-state arrays.

`parameters_0` is Euler's conservative state or SPECTRAX's complex Hermite–Fourier coefficients. PIC stores full initial particle positions and velocities as `parameters_0` and `parameters_1` (electrons first, then ions). `time_scale` converts saved physical times to the displayed normalized clock. Movies subsample long trajectories without interpolating artificial density states.

Completed validation runs check autodiff against directional finite differences, Euler conservation, plasma electric fields, and time-step refinement. Finer retention and PIC fits and a higher-Hermite Vlasov candidate are still running; the gallery identifies the completed resolution. Upstream revisions and JAX dependencies are pinned in [requirements.txt](requirements.txt). Increase resolution, integration steps, and optimization iterations together when refining a result; the runtime target is a warm objective/gradient evaluation under ten seconds, rather than a hardware-independent guarantee.

## Credits and license

Solvers: [SPECTRAX](https://github.com/uwplasma/SPECTRAX) and [JAX-in-Cell](https://github.com/uwplasma/JAX-in-Cell), with examples written in the simple editable-input style of UWPlasma and [Simsopt](https://github.com/hiddenSymmetries/simsopt). Shared code is in [helpers.py](helpers.py); numerical Euler integration lives there too.

`W7X-Spulen_Plasma_blau_gelb.jpg`: Max-Planck-Institut für Plasmaphysik, [Wikimedia Commons source](https://commons.wikimedia.org/wiki/File:W7X-Spulen_Plasma_blau_gelb.jpg), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The included image is the 330 × 205 version; plots and movies are scalar-density and false-color transformations with this attribution. The code is GPL-3.0 licensed because the second-order Euler implementation is adapted from Philip Mocz’s [finitevolume-jax](https://github.com/pmocz/finitevolume-jax) (2024). The image retains its separate license.
