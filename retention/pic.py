"""Fit and retain an x-vx phase-space image over a self-consistent PIC trajectory."""

from pathlib import Path
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers import load_target, make_pic, optimize, save_results  # noqa: E402

IMAGE = ROOT / "W7X-Spulen_Plasma_blau_gelb.jpg"
SIZE, PARTICLES = 256, 2048
FIELD_GRID = 64
STEPS = 2400
HORIZONS = (240, 800, 1600, STEPS)
STAGE_ITERATIONS = (50, 50, 50, 100)
STAGE_RATES = (0.001, 0.0005, 0.0002, 0.0001)
LENGTH, VELOCITY_RANGE = 1.0, 0.20
MAX_SPEED = 0.90  # smooth latent parameterization keeps the relativistic pusher subluminal
KERNEL_WIDTH = 3.0
TIMESTEP_RATIO, DX_OVER_DEBYE_LENGTH, THERMAL_SPEED = 0.5, 1.0, 0.10
SAMPLED_STEPS = 16
OUTPUT = ROOT / "results" / "retention_pic"
RESUME = False  # Set True to continue from a saved result when every model setting matches.

target = load_target(IMAGE, SIZE, floor=0.05)
warm_start = None
saved_results = OUTPUT / "results.npz"
if RESUME and saved_results.exists():
    with np.load(saved_results) as previous:
        compatible = (
            all(f"parameters_{i}" in previous.files for i in range(2, 11))
            and previous["target"].shape == target.shape
            and np.allclose(previous["target"], np.asarray(target), rtol=0, atol=1e-12)
            and previous["parameters_0"].shape == (2 * PARTICLES, 3)
            and previous["parameters_1"].shape == (2 * PARTICLES, 3)
            and float(previous["parameters_2"]) == MAX_SPEED
            and float(previous["parameters_3"]) == FIELD_GRID
            and float(previous["parameters_4"]) == KERNEL_WIDTH
            and float(previous["parameters_5"]) == TIMESTEP_RATIO
            and float(previous["parameters_6"]) == DX_OVER_DEBYE_LENGTH
            and float(previous["parameters_7"]) == THERMAL_SPEED
            and float(previous["parameters_8"]) == LENGTH
            and float(previous["parameters_9"]) == VELOCITY_RANGE
            and float(previous["parameters_10"]) == SAMPLED_STEPS
            and previous["frames"].shape[0] == STEPS + 1
        )
        if not compatible:
            raise ValueError("Cannot resume: saved result does not match the target and PIC settings.")
        saved_positions = previous["parameters_0"][:PARTICLES, 0]
        saved_vx = previous["parameters_1"][:PARTICLES, 0] / 299792458.0
        warm_start = jnp.asarray(np.stack((
            saved_positions,
            np.arctanh(np.clip(saved_vx / MAX_SPEED, -0.999999, 0.999999)),
        ), axis=1), dtype=jnp.float64)
if warm_start is not None:
    HORIZONS, STAGE_ITERATIONS, STAGE_RATES = (STEPS,), (100,), (1e-5,)
    print("Resuming from a matching saved PIC result.", flush=True)
elif saved_results.exists():
    print("Starting from seeded particles; set RESUME=True to continue a matching saved fit.", flush=True)
start = time.time()
optimized_parameters = None
stage_histories = []
for horizon, iterations, learning_rate in zip(HORIZONS, STAGE_ITERATIONS, STAGE_RATES):
    stage_initial, run_simulation, image_from_phase_space, parameter_image, initial_electric_field = make_pic(
        target, PARTICLES, horizon, LENGTH, VELOCITY_RANGE, seed=113,
        kernel_width=KERNEL_WIDTH, timestep_ratio=TIMESTEP_RATIO,
        debye_ratio=DX_OVER_DEBYE_LENGTH, thermal_speed=THERMAL_SPEED,
        max_speed=MAX_SPEED, field_grid_points=FIELD_GRID,
    )
    if optimized_parameters is None:
        initial_parameters = stage_initial
        if warm_start is not None:
            optimized_parameters = warm_start
    retention_indices = np.linspace(
        0, horizon - 1, min(SAMPLED_STEPS, horizon)
    ).astype(int)

    @jax.checkpoint
    def stage_loss(parameters):
        output = run_simulation(parameters)
        initial_image = parameter_image(parameters)
        positions = output["positions"][:, :PARTICLES, 0]
        velocities = output["velocities"][:, :PARTICLES, 0]
        sampled_images = jax.lax.map(
            lambda state: image_from_phase_space(state[0], state[1]),
            (positions[retention_indices], velocities[retention_indices]),
        )
        initial_error = jnp.mean((initial_image - target) ** 2)
        trajectory_error = jnp.mean((sampled_images - target[None, :, :]) ** 2)
        vx_history = output["velocities"][:, :PARTICLES, 0] / 299792458.0
        initial_vx = MAX_SPEED * jnp.tanh(parameters[:, 1])
        vx_history = jnp.concatenate((initial_vx[None, :], vx_history), axis=0)
        outside_view = jnp.maximum(jnp.abs(vx_history) - VELOCITY_RANGE, 0.0) / VELOCITY_RANGE
        viewport_penalty = jnp.mean(outside_view**2)
        return 0.5 * initial_error + 0.5 * trajectory_error + viewport_penalty

    starting_point = stage_initial if optimized_parameters is None else optimized_parameters
    start_loss = float(stage_loss(starting_point))
    print(
        f"Optimizing PIC retention through {horizon} steps with "
        f"{len(retention_indices)} sampled frames: {iterations} Adam updates at "
        f"lr={learning_rate:g}; start loss={start_loss:.6g}.",
        flush=True,
    )
    optimized_parameters, history = optimize(stage_loss, starting_point, iterations, learning_rate)
    stage_histories.append(np.asarray(history))
    print(f"  horizon {horizon} best loss={float(stage_loss(optimized_parameters)):.6g}.", flush=True)
loss = stage_loss
direction = jax.random.normal(jax.random.PRNGKey(702), optimized_parameters.shape)
direction /= jnp.linalg.norm(direction)
gradient = jax.grad(loss)(optimized_parameters)
autodiff_directional = jnp.sum(gradient * direction)
print(f"Long-horizon directional derivative (AD): {float(autodiff_directional):.6g}.", flush=True)
for epsilon in (1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11):
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
    f"PIC retention completed in {time.time()-start:.1f}s; "
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
    target, optimized_initial_image, frames, np.concatenate(stage_histories), OUTPUT,
    parameters=(
        optimized_output["initial_positions"],
        optimized_output["initial_velocities"],
        jnp.asarray(MAX_SPEED),
        jnp.asarray(FIELD_GRID),
        jnp.asarray(KERNEL_WIDTH),
        jnp.asarray(TIMESTEP_RATIO),
        jnp.asarray(DX_OVER_DEBYE_LENGTH),
        jnp.asarray(THERMAL_SPEED),
        jnp.asarray(LENGTH),
        jnp.asarray(VELOCITY_RANGE),
        jnp.asarray(SAMPLED_STEPS),
    ),
    times=times,
    baseline=baseline_frames,
    time_scale=optimized_output["plasma_frequency"],
    extent=(-LENGTH / 2, LENGTH / 2, -VELOCITY_RANGE, VELOCITY_RANGE),
    labels=("x [m]", "vx / c"),
    title="Particle-in-cell · retention",
    electric_fields=electric_fields,
    baseline_electric_fields=baseline_electric_fields,
    time_label="ωₚₑ t",
    field_label="Ex [V/m]",
    stage_lengths=STAGE_ITERATIONS,
    stage_times=tuple(float(times[-1] * optimized_output["plasma_frequency"]) * h / STEPS for h in HORIZONS),
)
