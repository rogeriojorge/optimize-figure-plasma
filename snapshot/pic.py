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
SIZE, PARTICLES = 32, 512
STEPS = 1000
HORIZONS = (100, 300, 600, STEPS)
STAGE_ITERATIONS = (50, 50, 50, 100)
STAGE_RATES = (0.001, 0.0005, 0.0002, 0.0001)
LENGTH, VELOCITY_RANGE = 1.0, 0.20
MAX_SPEED = 0.90  # smooth latent parameterization keeps the relativistic pusher subluminal
KERNEL_WIDTH = 1.0
TIMESTEP_RATIO, DX_OVER_DEBYE_LENGTH, THERMAL_SPEED = 1.0, 1.0, 0.10
OUTPUT = ROOT / "results" / "snapshot_pic"

target = load_target(IMAGE, SIZE, floor=0.05)
start = time.time()
optimized_parameters = None
for horizon, iterations, learning_rate in zip(HORIZONS, STAGE_ITERATIONS, STAGE_RATES):
    stage_initial, run_simulation, image_from_phase_space, parameter_image, initial_electric_field = make_pic(
        target, PARTICLES, horizon, LENGTH, VELOCITY_RANGE, seed=71,
        kernel_width=KERNEL_WIDTH, timestep_ratio=TIMESTEP_RATIO,
        debye_ratio=DX_OVER_DEBYE_LENGTH, thermal_speed=THERMAL_SPEED,
        max_speed=MAX_SPEED,
    )
    if optimized_parameters is None:
        initial_parameters = stage_initial

    @jax.checkpoint
    def stage_loss(parameters):
        output = run_simulation(parameters)
        final_image = image_from_phase_space(
            output["positions"][-1, :PARTICLES, 0],
            output["velocities"][-1, :PARTICLES, 0],
        )
        return jnp.mean((final_image - target) ** 2)

    start_loss = float(stage_loss(stage_initial if optimized_parameters is None else optimized_parameters))
    print(
        f"Optimizing PIC snapshot through {horizon} steps: "
        f"{iterations} Adam updates at lr={learning_rate:g}; start loss={start_loss:.6g}.",
        flush=True,
    )
    optimized_parameters, history = optimize(
        stage_loss, stage_initial if optimized_parameters is None else optimized_parameters,
        iterations, learning_rate,
    )
    print(f"  horizon {horizon} best loss={float(stage_loss(optimized_parameters)):.6g}.", flush=True)
loss = stage_loss
direction = jax.random.normal(jax.random.PRNGKey(701), optimized_parameters.shape)
direction /= jnp.linalg.norm(direction)
gradient = jax.grad(loss)(optimized_parameters)
autodiff_directional = jnp.sum(gradient * direction)
print(f"Long-horizon directional derivative (AD): {float(autodiff_directional):.6g}.", flush=True)
for epsilon in (1e-3, 3e-4, 1e-4):
    finite_difference = (
        loss(optimized_parameters + epsilon * direction)
        - loss(optimized_parameters - epsilon * direction)
    ) / (2 * epsilon)
    relative_error = jnp.abs(finite_difference - autodiff_directional) / jnp.maximum(
        jnp.abs(autodiff_directional), 1e-8
    )
    print(
        f"  central FD ε={epsilon:g}: {float(finite_difference):.6g}; "
        f"relative error={float(relative_error):.3g}.",
        flush=True,
    )
baseline_output = run_simulation(initial_parameters)
optimized_output = run_simulation(optimized_parameters)
baseline_initial_image = parameter_image(initial_parameters)
optimized_initial_image = parameter_image(optimized_parameters)
baseline_electric_fields = jnp.concatenate((
    initial_electric_field(initial_parameters)[None, :],
    baseline_output["electric_field"][:, :, 0],
), axis=0)
electric_fields = jnp.concatenate((
    initial_electric_field(optimized_parameters)[None, :],
    optimized_output["electric_field"][:, :, 0],
), axis=0)
trajectory_images = jax.lax.map(
    lambda state: image_from_phase_space(state[0], state[1]),
    (optimized_output["positions"][:, :PARTICLES, 0],
     optimized_output["velocities"][:, :PARTICLES, 0]),
    batch_size=32,
)
baseline_trajectory_images = jax.lax.map(
    lambda state: image_from_phase_space(state[0], state[1]),
    (baseline_output["positions"][:, :PARTICLES, 0],
     baseline_output["velocities"][:, :PARTICLES, 0]),
    batch_size=32,
)
baseline_frames = jnp.concatenate((baseline_initial_image[None, :, :], baseline_trajectory_images), axis=0)
frames = jnp.concatenate((optimized_initial_image[None, :, :], trajectory_images), axis=0)
times = jnp.concatenate((jnp.zeros((1,)), optimized_output["time_array"]))
print(
    f"PIC snapshot completed in {time.time()-start:.1f}s; "
    f"loss {float(loss(initial_parameters)):.6g} -> {float(loss(optimized_parameters)):.6g}; "
    f"duration={float(times[-1] * optimized_output['plasma_frequency']):.2f} ω_pe⁻¹."
)
field_rms = jnp.sqrt(jnp.mean(optimized_output["electric_field"] ** 2))
velocity_change = jnp.max(jnp.abs(
    optimized_output["velocities"][-1, :PARTICLES, 0]
    - optimized_output["velocities"][0, :PARTICLES, 0]
))
vx_history = optimized_output["velocities"][:, :PARTICLES, 0] / 299792458.0
window_occupancy = jnp.mean(jnp.abs(vx_history) <= VELOCITY_RANGE)
max_speed_ratio = jnp.max(jnp.linalg.norm(
    optimized_output["velocities"][:, :PARTICLES, :], axis=-1
)) / 299792458.0
represented_mass = jnp.mean(jnp.mean(frames, axis=(1, 2)))
print(
    f"Physics check: {PARTICLES} electrons and {PARTICLES} ions, "
    f"field RMS={float(field_rms):.6g}, max electron vx change={float(velocity_change):.6g} m/s, "
    f"vx window occupancy={float(window_occupancy):.4f}, max |v|/c={float(max_speed_ratio):.4f}, "
    f"represented image mass={float(represented_mass):.4f}."
)
save_results(
    target, optimized_initial_image, frames, history, OUTPUT,
    parameters={
        "initial_positions": optimized_output["initial_positions"],
        "initial_velocities": optimized_output["initial_velocities"],
    },
    times=times,
    baseline=baseline_frames,
    time_scale=optimized_output["plasma_frequency"],
    extent=(-LENGTH / 2, LENGTH / 2, -VELOCITY_RANGE, VELOCITY_RANGE),
    labels=("x [m]", "vx / c"),
    title="Particle-in-cell · snapshot · tω_pe",
    electric_fields=electric_fields,
    baseline_electric_fields=baseline_electric_fields,
    time_label="ωₚₑ t",
    field_label="Ex [V/m]",
)
