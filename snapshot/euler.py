"""Draw a target from a uniform fluid by optimizing its initial velocity."""

from pathlib import Path
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers import euler_evolve_muscl, load_target, optimize, save_results  # noqa: E402

IMAGE = ROOT / "W7X-Spulen_Plasma_blau_gelb.jpg"
SIZE, STEPS = 128, 352
HORIZONS = (0.10, 0.20, 0.30, 0.43)
ITERATIONS_PER_HORIZON = (30, 40, 60, 100)
SOUND_SPEED, VELOCITY_SCALE = 1.0, 0.80
IMAGE_CONTRAST = 0.10
LEARNING_RATE = 0.01
OUTPUT = ROOT / "results" / "snapshot_euler"

print("Loading target and preparing the periodic second-order isothermal Euler model...")
artwork = load_target(IMAGE, SIZE)
target = 1.0 + IMAGE_CONTRAST * (artwork - 1.0)
target = target / jnp.mean(target)


def conserved(parameters):
    """Keep density uniform; optimize only smoothly bounded velocity fields."""
    vx = VELOCITY_SCALE * jnp.tanh(parameters[0])
    vy = VELOCITY_SCALE * jnp.tanh(parameters[1])
    return jnp.stack((jnp.ones_like(vx), vx, vy))


initial = jnp.zeros((2, SIZE, SIZE), dtype=target.dtype)
parameters = initial
history_parts, horizon_losses = [], []
stage_min_density, stage_mass_drift = [], []
start = time.time()
print("Optimizing the image at progressively longer horizons from uniform density...")
for horizon, iterations in zip(HORIZONS, ITERATIONS_PER_HORIZON):
    stage_steps = max(32, round(STEPS * horizon / HORIZONS[-1]))
    dt = horizon / stage_steps

    def run(p):
        return euler_evolve_muscl(conserved(p), stage_steps, dt, 1.0 / SIZE, SOUND_SPEED)

    def loss(p):
        return jnp.mean(jnp.square(run(p)[-1] - target))

    print(f"Horizon T={horizon:g}: {iterations} L-BFGS-B iterations")
    parameters, stage_history = optimize(
        loss, parameters, iterations, LEARNING_RATE, method="l-bfgs-b"
    )
    history_parts.append(stage_history)
    horizon_losses.append(float(loss(parameters)))
    stage_frames = np.asarray(run(parameters))
    stage_min_density.append(float(stage_frames.min()))
    stage_mass_drift.append(float(np.max(np.abs(stage_frames.mean(axis=(-2, -1)) - stage_frames[0].mean()))))
    if stage_min_density[-1] <= 0:
        raise FloatingPointError(f"Negative density at snapshot horizon T={horizon:g}.")
    print(f"  minimum density={stage_min_density[-1]:.6g}, mass drift={stage_mass_drift[-1]:.3g}")

T_FINAL = HORIZONS[-1]
dt = T_FINAL / STEPS
all_frames = run(parameters)
baseline_parameters = jnp.zeros_like(parameters)
baseline_frames = run(baseline_parameters)
times = jnp.arange(STEPS + 1) * dt
history = np.concatenate(history_parts)
gradient = jax.value_and_grad(loss)(parameters)[1]
direction = jnp.sin(jnp.arange(parameters.size, dtype=parameters.dtype)).reshape(parameters.shape)
direction = direction / jnp.linalg.norm(direction)
ad_directional = float(jnp.vdot(gradient, direction))
epsilon = 1e-4
fd_directional = float((loss(parameters + epsilon * direction) - loss(parameters - epsilon * direction)) / (2 * epsilon))
gradient_check_error = abs(ad_directional - fd_directional) / max(abs(ad_directional), abs(fd_directional), 1e-12)
fine_frames = euler_evolve_muscl(conserved(parameters), 2 * STEPS, dt / 2, 1.0 / SIZE, SOUND_SPEED)
replay_difference = np.asarray(fine_frames[-1] - all_frames[-1])
fine_replay_rms = float(np.sqrt(np.mean(replay_difference**2)))
fine_replay_max = float(np.max(np.abs(replay_difference)))
fine_min_density = float(np.min(fine_frames))
fine_mass_drift = float(np.max(np.abs(np.asarray(fine_frames).mean(axis=(-2, -1)) - np.asarray(fine_frames[0]).mean())))
if fine_min_density <= 0:
    raise FloatingPointError("Negative density during the twice-fine Euler replay.")
print(
    f"Euler snapshot fit finished in {time.time() - start:.1f}s; "
    f"final density MSE={horizon_losses[-1]:.6g}; AD/FD={gradient_check_error:.3g}; "
    f"fine-step endpoint RMS={fine_replay_rms:.3g}"
)
save_results(
    target, all_frames[0], all_frames, history, OUTPUT,
    parameters=conserved(parameters), times=times, baseline=baseline_frames,
    extent=(0, 1, 0, 1), labels=("x", "y"),
    title="Isothermal Euler · velocity-designed snapshot",
    density_background=1.0, image_contrast=IMAGE_CONTRAST,
    stage_lengths=[len(part) for part in history_parts], stage_times=HORIZONS,
)
archive = np.load(OUTPUT / "results.npz")
data = {key: archive[key] for key in archive.files}
data.update(stage_min_density=np.asarray(stage_min_density), stage_mass_drift=np.asarray(stage_mass_drift),
            horizon_final_losses=np.asarray(horizon_losses),
            gradient_check_error=gradient_check_error, fine_replay_endpoint_rms=fine_replay_rms,
            fine_replay_endpoint_max=fine_replay_max, fine_replay_min_density=fine_min_density,
            fine_replay_mass_drift=fine_mass_drift)
np.savez_compressed(OUTPUT / "results.npz", **data)
