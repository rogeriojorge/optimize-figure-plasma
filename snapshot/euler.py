"""Fit an initial isothermal Euler density and velocity to a final 2D image."""

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
SIZE, STEPS = 48, 64
T_FINAL = 0.10
SOUND_SPEED, VELOCITY_SCALE = 0.45, 0.22
ITERATIONS, LEARNING_RATE = 100, 0.015
SAVED_FRAMES = 9
OUTPUT = ROOT / "results" / "snapshot_euler"

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


initial = jnp.stack((jnp.log(0.2 + 0.8 * target), jnp.zeros_like(target), jnp.zeros_like(target)))


def loss(parameters):
    final_density = run(parameters)[-1]
    return jnp.mean(jnp.square(final_density - target))


print("Optimizing initial density and velocity against the final Euler snapshot...")
start = time.time()
parameters, history = optimize(loss, initial, ITERATIONS, LEARNING_RATE)
all_frames = run(parameters)
frame_indices = jnp.linspace(0, STEPS, SAVED_FRAMES).astype(jnp.int32)
frames = all_frames[frame_indices]
times = frame_indices * dt
print(
    f"Euler snapshot fit finished in {time.time() - start:.1f}s; "
    f"final density MSE={float(loss(parameters)):.6g}"
)
save_results(target, all_frames[0], frames, history, OUTPUT,
             parameters=conserved(parameters), times=times)
