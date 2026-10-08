"""Fit a final x-vx phase-space image with the self-consistent JAX-in-Cell PIC solver."""

from pathlib import Path
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers import load_target, make_pic, optimize, save_results  # noqa: E402

IMAGE = ROOT / "W7X-Spulen_Plasma_blau_gelb.jpg"
SIZE, PARTICLES, STEPS = 48, 2048, 24
LENGTH, VELOCITY_RANGE = 1.0, 0.20
KERNEL_WIDTH = 1.0
TIMESTEP_RATIO, DX_OVER_DEBYE_LENGTH = 0.65, 0.7
ITERATIONS, LEARNING_RATE = 60, 0.01
OUTPUT = ROOT / "results" / "snapshot_pic"

target = load_target(IMAGE, SIZE, floor=0.05)
initial_parameters, run_simulation, image_from_phase_space = make_pic(
    target, PARTICLES, STEPS, LENGTH, VELOCITY_RANGE, seed=71,
    kernel_width=KERNEL_WIDTH, timestep_ratio=TIMESTEP_RATIO,
    debye_ratio=DX_OVER_DEBYE_LENGTH,
)


def loss(parameters):
    output = run_simulation(parameters)
    final_image = image_from_phase_space(
        output["positions"][-1, :PARTICLES, 0],
        output["velocities"][-1, :PARTICLES, 0],
    )
    return jnp.mean((final_image - target) ** 2)


print(f"Optimizing PIC snapshot with {ITERATIONS} Adam steps...")
start = time.time()
optimized_parameters, history = optimize(
    loss, initial_parameters, ITERATIONS, LEARNING_RATE
)
optimized_output = run_simulation(optimized_parameters)
optimized_initial_image = image_from_phase_space(
    optimized_parameters[:, 0], optimized_parameters[:, 1] * 299792458.0
)
trajectory_images = jnp.stack([
    image_from_phase_space(
        optimized_output["positions"][i, :PARTICLES, 0],
        optimized_output["velocities"][i, :PARTICLES, 0],
    )
    for i in range(STEPS)
])
frames = jnp.concatenate((optimized_initial_image[None, :, :], trajectory_images), axis=0)
times = jnp.concatenate((jnp.zeros((1,)), optimized_output["time_array"]))
print(
    f"PIC snapshot completed in {time.time()-start:.1f}s; "
    f"loss {float(loss(initial_parameters)):.6g} -> {float(loss(optimized_parameters)):.6g}."
)
field_rms = jnp.sqrt(jnp.mean(optimized_output["electric_field"] ** 2))
velocity_change = jnp.max(jnp.abs(
    optimized_output["velocities"][-1, :PARTICLES, 0]
    - optimized_output["velocities"][0, :PARTICLES, 0]
))
vx_history = optimized_output["velocities"][:, :PARTICLES, 0] / 299792458.0
window_occupancy = jnp.mean(jnp.abs(vx_history) <= VELOCITY_RANGE)
represented_mass = jnp.mean(jnp.mean(frames, axis=(1, 2)))
print(
    f"Physics check: {PARTICLES} electrons and {PARTICLES} ions, "
    f"field RMS={float(field_rms):.6g}, max electron vx change={float(velocity_change):.6g} m/s, "
    f"vx window occupancy={float(window_occupancy):.4f}, represented image mass={float(represented_mass):.4f}."
)
save_results(
    target, optimized_initial_image, frames, history, OUTPUT,
    parameters={
        "initial_positions": optimized_output["initial_positions"],
        "initial_velocities": optimized_output["initial_velocities"],
    },
    times=times,
)
