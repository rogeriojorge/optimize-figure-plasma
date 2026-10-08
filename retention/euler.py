"""Fit a 2D isothermal Euler state that retains the target over its trajectory."""

from pathlib import Path
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers import euler_evolve, load_target, optimize, save_results  # noqa: E402

IMAGE = ROOT / "W7X-Spulen_Plasma_blau_gelb.jpg"
SIZE, STEPS = 48, 128
T_FINAL = 0.50
SOUND_SPEED, VELOCITY_SCALE = 0.45, 0.80
ITERATIONS, LEARNING_RATE = 200, 0.015
SAVED_FRAMES = STEPS + 1
OUTPUT = ROOT / "results" / "retention_euler"

print("Loading target and preparing the periodic 2D isothermal Euler model...")
target = load_target(IMAGE, SIZE)
dt = T_FINAL / STEPS


def conserved(parameters):
    """Map log density and bounded velocities to conservative Euler variables."""
    density = jnp.exp(parameters[0])
    vx = VELOCITY_SCALE * jnp.tanh(parameters[1])
    vy = VELOCITY_SCALE * jnp.tanh(parameters[2])
    return jnp.stack((density, density * vx, density * vy))


def run(parameters):
    return euler_evolve(conserved(parameters), STEPS, dt, SOUND_SPEED)


initial = jnp.stack((jnp.log(target), jnp.zeros_like(target), jnp.zeros_like(target)))


def loss(parameters):
    densities = run(parameters)
    trajectory_error = jnp.mean(jnp.square(densities - target[None, :, :]))
    initial_error = jnp.mean(jnp.square(densities[0] - target))
    return trajectory_error + 0.5 * initial_error


print("Optimizing initial density and velocity to retain the image across the trajectory...")
start = time.time()
parameters, history = optimize(loss, initial, ITERATIONS, LEARNING_RATE)
all_frames = run(parameters)
frame_indices = jnp.linspace(0, STEPS, SAVED_FRAMES).astype(jnp.int32)
baseline_frames = run(initial)[frame_indices]
frames = all_frames[frame_indices]
print(
    f"Euler retention fit finished in {time.time() - start:.1f}s; "
    f"trajectory-average density MSE={float(loss(parameters)):.6g}"
)
times = frame_indices * dt
save_results(target, all_frames[0], frames, history, OUTPUT,
             parameters=conserved(parameters), times=times,
             baseline=baseline_frames, extent=(0, 2*jnp.pi, 0, 2*jnp.pi),
             labels=('x', 'y'), title='Isothermal Euler · retention')
