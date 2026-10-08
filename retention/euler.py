"""Fit a 2D isothermal Euler state that retains the target over its trajectory."""

from pathlib import Path
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers import euler_evolve, load_target, optimize, save_results  # noqa: E402

IMAGE = ROOT / "W7X-Spulen_Plasma_blau_gelb.jpg"
SIZE, STEPS = 128, 192
HORIZONS = (0.10, 0.20, 0.35, 0.50)
ITERATIONS_PER_HORIZON = (30, 40, 60, 100)
SOUND_SPEED, VELOCITY_SCALE = 0.45, 0.80
LEARNING_RATE = 0.015
OUTPUT = ROOT / "results" / "retention_euler"

print("Loading target and preparing the periodic 2D isothermal Euler model...")
target = load_target(IMAGE, SIZE)
initial = jnp.stack((jnp.log(target), jnp.zeros_like(target), jnp.zeros_like(target)))
parameters = initial
history_parts, horizon_losses = [], []
stage_min_density, stage_mass_drift = [], []
start = time.time()
print("Optimizing image retention at progressively longer horizons...")
for horizon, iterations in zip(HORIZONS, ITERATIONS_PER_HORIZON):
    stage_steps = max(32, round(STEPS * horizon / HORIZONS[-1]))
    dt = horizon / stage_steps

    def conserved(p):
        """Map log density and bounded velocities to conservative Euler variables."""
        density = jnp.exp(p[0])
        vx = VELOCITY_SCALE * jnp.tanh(p[1])
        vy = VELOCITY_SCALE * jnp.tanh(p[2])
        return jnp.stack((density, density * vx, density * vy))

    def run(p):
        return euler_evolve(conserved(p), stage_steps, dt, SOUND_SPEED)

    def loss(p):
        densities = run(p)
        trajectory_error = jnp.mean(jnp.square(densities - target[None, :, :]))
        initial_error = jnp.mean(jnp.square(densities[0] - target))
        return trajectory_error + 0.5 * initial_error

    print(f"Horizon T={horizon:g}: {iterations} Adam iterations")
    parameters, stage_history = optimize(loss, parameters, iterations, LEARNING_RATE)
    history_parts.append(stage_history)
    horizon_losses.append(float(loss(parameters)))
    stage_frames = np.asarray(run(parameters))
    stage_min_density.append(float(stage_frames.min()))
    stage_mass_drift.append(float(np.max(np.abs(stage_frames.mean(axis=(-2, -1)) - stage_frames[0].mean()))))
    if stage_min_density[-1] <= 0:
        raise FloatingPointError(f"Negative density at retention horizon T={horizon:g}.")
    print(f"  minimum density={stage_min_density[-1]:.6g}, mass drift={stage_mass_drift[-1]:.3g}")

T_FINAL = HORIZONS[-1]
dt = T_FINAL / STEPS
all_frames = run(parameters)
baseline_frames = run(initial)
times = jnp.arange(STEPS + 1) * dt
history = np.concatenate(history_parts)
gradient = jax.value_and_grad(loss)(parameters)[1]
direction = jnp.sin(jnp.arange(parameters.size, dtype=parameters.dtype)).reshape(parameters.shape)
direction = direction / jnp.linalg.norm(direction)
ad_directional = float(jnp.vdot(gradient, direction))
epsilon = 1e-4
fd_directional = float((loss(parameters + epsilon * direction) - loss(parameters - epsilon * direction)) / (2 * epsilon))
gradient_check_error = abs(ad_directional - fd_directional) / max(abs(ad_directional), abs(fd_directional), 1e-12)
fine_frames = euler_evolve(conserved(parameters), 2 * STEPS, dt / 2, SOUND_SPEED)
replay_difference = np.asarray(fine_frames[-1] - all_frames[-1])
fine_replay_rms = float(np.sqrt(np.mean(replay_difference**2)))
fine_replay_max = float(np.max(np.abs(replay_difference)))
fine_min_density = float(np.min(fine_frames))
fine_mass_drift = float(np.max(np.abs(np.asarray(fine_frames).mean(axis=(-2, -1)) - np.asarray(fine_frames[0]).mean())))
if fine_min_density <= 0:
    raise FloatingPointError("Negative density during the twice-fine Euler replay.")
print(
    f"Euler retention fit finished in {time.time() - start:.1f}s; "
    f"trajectory-average density MSE={horizon_losses[-1]:.6g}; AD/FD={gradient_check_error:.3g}; "
    f"fine-step endpoint RMS={fine_replay_rms:.3g}"
)
save_results(
    target, all_frames[0], all_frames, history, OUTPUT,
    parameters=conserved(parameters), times=times, baseline=baseline_frames,
    extent=(0, 2 * jnp.pi, 0, 2 * jnp.pi), labels=("x", "y"),
    title="Isothermal Euler · retention",
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
