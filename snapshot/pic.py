"""Fit a final x-vx phase-space image with the self-consistent JAX-in-Cell PIC solver."""

from pathlib import Path
import sys
import time
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers import load_target, make_pic, optimize, save_results  # noqa: E402

IMAGE = ROOT / "W7X-Spulen_Plasma_blau_gelb.jpg"
SIZE, PARTICLES = 256, 4096
FIELD_GRID = 64
STEPS = 2000
HORIZONS = (200, 600, 1200, STEPS)
STAGE_ITERATIONS = (50, 50, 50, 100)
STAGE_RATES = (0.001, 0.0005, 0.0002, 0.0001)
LENGTH, VELOCITY_RANGE = 1.0, 0.20
MAX_SPEED = 0.90  # smooth latent parameterization keeps the relativistic pusher subluminal
KERNEL_WIDTH = 3.0
TIMESTEP_RATIO, DX_OVER_DEBYE_LENGTH, THERMAL_SPEED = 0.5, 1.0, 0.10
INITIALIZATION = "image"
TWO_STREAM_DRIFT, TWO_STREAM_THERMAL, TWO_STREAM_PERTURBATION = 0.20, 0.05, 5e-5
SAVE_STRIDE = 10
OUTPUT = ROOT / "results" / "snapshot_pic"
RESUME = os.environ.get("PIC_RESUME", "0") == "1"

def checkpoint_signature():
    return repr((SIZE, PARTICLES, FIELD_GRID, STEPS, HORIZONS, STAGE_ITERATIONS, STAGE_RATES,
                 LENGTH, VELOCITY_RANGE, MAX_SPEED, KERNEL_WIDTH, TIMESTEP_RATIO,
                 DX_OVER_DEBYE_LENGTH, THERMAL_SPEED, INITIALIZATION, SAVE_STRIDE))

def save_stage_checkpoint(stage, parameters, history):
    """Atomically retain the latest completed optimization stage."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT / "stage_checkpoint.tmp.npz"
    np.savez_compressed(temporary, stage=np.asarray(stage), parameters=np.asarray(parameters),
                        history=np.asarray(history), target=np.asarray(target),
                        signature=np.asarray(checkpoint_signature()))
    os.replace(temporary, OUTPUT / "stage_checkpoint.npz")

target = load_target(IMAGE, SIZE, floor=0.05)
warm_start = None
saved_results = OUTPUT / "results.npz"
if RESUME and saved_results.exists():
    with np.load(saved_results) as previous:
        compatible = (
            all(f"parameters_{i}" in previous.files for i in range(2, 10))
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
            and "simulation_steps" in previous.files
            and int(previous["simulation_steps"]) == STEPS
            and "save_stride" in previous.files
            and int(previous["save_stride"]) == SAVE_STRIDE
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
stage_resume = None
checkpoint_file = OUTPUT / "stage_checkpoint.npz"
if RESUME and warm_start is None and checkpoint_file.exists():
    with np.load(checkpoint_file) as checkpoint:
        signature_matches = (
            ("signature" in checkpoint.files and str(checkpoint["signature"]) == checkpoint_signature())
            or ("signature" not in checkpoint.files
                and checkpoint["parameters"].shape == (PARTICLES, 2)
                and str(checkpoint["initialization"]) == INITIALIZATION)
        )
        if (signature_matches
                and np.allclose(checkpoint["target"], np.asarray(target), rtol=0, atol=1e-12)):
            completed_stage = HORIZONS.index(int(checkpoint["stage"]))
            expected_history = sum(count + 1 for count in STAGE_ITERATIONS[:completed_stage+1])
            if checkpoint["history"].shape != (expected_history,):
                raise ValueError("Cannot resume: stage checkpoint history does not match completed stages.")
            stage_resume = (completed_stage, np.asarray(checkpoint["parameters"]),
                            np.asarray(checkpoint["history"]))
            print(f"Resuming from atomic snapshot checkpoint after {HORIZONS[completed_stage]} steps.", flush=True)
start = time.time()
optimized_parameters = None if stage_resume is None else jnp.asarray(stage_resume[1])
stage_histories = []
if stage_resume is not None:
    offset = 0
    for count in STAGE_ITERATIONS[:stage_resume[0]+1]:
        stage_histories.append(stage_resume[2][offset:offset+count+1])
        offset += count + 1
initial_parameters = None
for stage_index, (horizon, iterations, learning_rate) in enumerate(zip(HORIZONS, STAGE_ITERATIONS, STAGE_RATES)):
    stage_initial, run_simulation, image_from_phase_space, parameter_image, initial_electric_field = make_pic(
        target, PARTICLES, horizon, LENGTH, VELOCITY_RANGE, seed=71,
        kernel_width=KERNEL_WIDTH, timestep_ratio=TIMESTEP_RATIO,
        debye_ratio=DX_OVER_DEBYE_LENGTH, thermal_speed=THERMAL_SPEED,
        max_speed=MAX_SPEED, field_grid_points=FIELD_GRID,
        initialization=INITIALIZATION, two_stream_drift=TWO_STREAM_DRIFT,
        two_stream_thermal=TWO_STREAM_THERMAL, two_stream_perturbation=TWO_STREAM_PERTURBATION,
    )
    if stage_index == 0:
        initial_parameters = stage_initial
    if stage_resume is not None and stage_index <= stage_resume[0]:
        continue
    if optimized_parameters is None and warm_start is not None:
        optimized_parameters = warm_start

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
    stage_histories.append(np.asarray(history))
    print(f"  horizon {horizon} best loss={float(stage_loss(optimized_parameters)):.6g}.", flush=True)
    save_stage_checkpoint(horizon, optimized_parameters, np.concatenate(stage_histories))
loss = stage_loss
direction = jax.random.normal(jax.random.PRNGKey(701), optimized_parameters.shape)
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
extended_steps = int(round(1.5 * STEPS))
_, run_extended, _, _, _ = make_pic(
    target, PARTICLES, extended_steps, LENGTH, VELOCITY_RANGE, seed=71,
    kernel_width=KERNEL_WIDTH, timestep_ratio=TIMESTEP_RATIO,
    debye_ratio=DX_OVER_DEBYE_LENGTH, thermal_speed=THERMAL_SPEED,
    max_speed=MAX_SPEED, field_grid_points=FIELD_GRID,
    initialization=INITIALIZATION, two_stream_drift=TWO_STREAM_DRIFT,
    two_stream_thermal=TWO_STREAM_THERMAL,
    two_stream_perturbation=TWO_STREAM_PERTURBATION,
)
extended_output = run_extended(optimized_parameters)
baseline_initial_image = parameter_image(initial_parameters)
optimized_initial_image = parameter_image(optimized_parameters)
def sampled_trajectory(output, initial_image, initial_field, steps):
    full_indices = np.unique(np.r_[np.arange(0, steps + 1, SAVE_STRIDE), steps])
    output_indices = full_indices[1:] - 1  # solver output begins at step 1; frame 0 is the IC.
    sampled_images = jax.lax.map(
        lambda state: image_from_phase_space(state[0], state[1]),
        (output["positions"][output_indices, :PARTICLES, 0],
         output["velocities"][output_indices, :PARTICLES, 0]),
        batch_size=32,
    )
    frames = jnp.concatenate((initial_image[None, :, :], sampled_images), axis=0)
    times = jnp.concatenate((jnp.zeros((1,)), output["time_array"][output_indices]))
    fields = jnp.concatenate((initial_field[None, :], output["electric_field"][output_indices, :, 0]), axis=0)
    return frames, times, fields, full_indices

baseline_frames, baseline_times, baseline_electric_fields, save_indices = sampled_trajectory(
    baseline_output, baseline_initial_image, initial_electric_field(initial_parameters), STEPS)
frames, times, electric_fields, _ = sampled_trajectory(
    optimized_output, optimized_initial_image, initial_electric_field(optimized_parameters), STEPS)
extended_frames, extended_times, extended_electric_fields, extended_save_indices = sampled_trajectory(
    extended_output, optimized_initial_image, initial_electric_field(optimized_parameters), extended_steps)
prefix_position_error = float(jnp.max(jnp.abs(
    extended_output["positions"][:STEPS] - optimized_output["positions"]
)))
prefix_velocity_error = float(jnp.max(jnp.abs(
    extended_output["velocities"][:STEPS] - optimized_output["velocities"]
)))
prefix_frame_error = float(jnp.max(jnp.abs(extended_frames[:len(frames)] - frames)))
prefix_field_error = float(jnp.max(jnp.abs(extended_electric_fields[:len(electric_fields)] - electric_fields)))
print(
    f"Paired same-IC prefix check: max position error={prefix_position_error:.3g} m, "
    f"velocity error={prefix_velocity_error:.3g} m/s, image error={prefix_frame_error:.3g}, "
    f"field error={prefix_field_error:.3g} V/m; "
    f"extension={float(extended_times[-1]/times[-1]):.6g}x.",
    flush=True,
)
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
    ),
    times=times,
    baseline=baseline_frames,
    time_scale=optimized_output["plasma_frequency"],
    extent=(-LENGTH / 2, LENGTH / 2, -VELOCITY_RANGE, VELOCITY_RANGE),
    labels=("x [m]", "vx / c"),
    title="Particle-in-cell · snapshot",
    electric_fields=electric_fields,
    baseline_electric_fields=baseline_electric_fields,
    extended_frames=extended_frames,
    extended_times=extended_times,
    extended_electric_fields=extended_electric_fields,
    metadata={
        "paired_same_optimized_initial_condition": np.asarray(True),
        "paired_prefix_position_error_m": np.asarray(prefix_position_error),
        "paired_prefix_velocity_error_m_s": np.asarray(prefix_velocity_error),
        "paired_prefix_image_error": np.asarray(prefix_frame_error),
        "paired_prefix_field_error_v_m": np.asarray(prefix_field_error),
        "simulation_steps": np.asarray(STEPS),
        "save_stride": np.asarray(SAVE_STRIDE),
        "paired_standard_steps": np.asarray(STEPS),
        "paired_extended_steps": np.asarray(extended_steps),
    },
    time_label="ωₚₑ t",
    field_label="Ex [V/m]",
    stage_lengths=tuple(count + 1 for count in STAGE_ITERATIONS),
    stage_times=tuple(float(times[-1] * optimized_output["plasma_frequency"]) * h / STEPS for h in HORIZONS),
)
