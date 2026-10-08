"""Draw a target from a uniform fluid by optimizing its initial velocity."""

from pathlib import Path
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers import euler_evolve_muscl, load_target, optimize, save_results  # noqa: E402

IMAGE = ROOT / "W7X-Spulen_Plasma_blau_gelb.jpg"
SIZE, STEPS = 48, 256
T_FINAL = 0.43
SOUND_SPEED, VELOCITY_SCALE = 1.0, 0.80
IMAGE_CONTRAST = 0.10
ITERATIONS, LEARNING_RATE = 200, 0.01
SAVED_FRAMES = STEPS + 1
OUTPUT = ROOT / "results" / "snapshot_euler"

print("Loading target and preparing the periodic second-order isothermal Euler model...")
artwork = load_target(IMAGE, SIZE)
target = 1.0 + IMAGE_CONTRAST * (artwork - 1.0)
target = target / jnp.mean(target)
dt = T_FINAL / STEPS


def conserved(parameters):
    """Keep density uniform; optimize only smoothly bounded velocity fields."""
    vx = VELOCITY_SCALE * jnp.tanh(parameters[0])
    vy = VELOCITY_SCALE * jnp.tanh(parameters[1])
    return jnp.stack((jnp.ones_like(vx), vx, vy))


def run(parameters):
    return euler_evolve_muscl(conserved(parameters), STEPS, dt, 1.0/SIZE, SOUND_SPEED)


initial = jnp.zeros((2, SIZE, SIZE), dtype=target.dtype)


def loss(parameters):
    final_density = run(parameters)[-1]
    return jnp.mean(jnp.square(final_density - target))


print("Optimizing velocity alone from uniform density against the final target...")
start = time.time()
parameters, history = optimize(loss, initial, ITERATIONS, LEARNING_RATE, method="l-bfgs-b")
all_frames = run(parameters)
frame_indices = jnp.linspace(0, STEPS, SAVED_FRAMES).astype(jnp.int32)
baseline_frames = run(initial)[frame_indices]
frames = all_frames[frame_indices]
times = frame_indices * dt
print(
    f"Euler snapshot fit finished in {time.time() - start:.1f}s; "
    f"final density MSE={float(loss(parameters)):.6g}"
)
save_results(target, all_frames[0], frames, history, OUTPUT,
             parameters=conserved(parameters), times=times,
             baseline=baseline_frames, extent=(0, 1, 0, 1),
             labels=('x', 'y'), title='Isothermal Euler · velocity-designed snapshot', density_background=1.0, image_contrast=IMAGE_CONTRAST)
