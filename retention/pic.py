"""Optimize an initial PIC state to match and retain a phase-space image."""

from pathlib import Path
import sys

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers import (StageCheckpoint, load_target, make_pic, optimize,
                     save_pic_artifacts)  # noqa: E402

# Editable model, optimizer, objective, and artifact inputs.
IMAGE = ROOT / 'W7X-Spulen_Plasma_blau_gelb.jpg'
TARGET_SIZE, PARTICLES, FIELD_GRID = 256, 4096, 64
STEPS = 2400
HORIZONS = (240, 800, 1600, STEPS)
STAGE_ITERATIONS, STAGE_RATES = (50, 50, 50, 100), (1e-3, 5e-4, 2e-4, 1e-4)
OPTIMIZER, SEED = 'adam', 113
LENGTH, VELOCITY_RANGE, MAX_SPEED = 1.0, 0.20, 0.90
KERNEL_WIDTH, TIMESTEP_RATIO, DX_OVER_DEBYE, THERMAL_SPEED = 3.0, 0.5, 1.0, 0.10
INITIALIZATION = 'image'
TWO_STREAM_DRIFT, TWO_STREAM_THERMAL, TWO_STREAM_PERTURBATION = 0.20, 0.05, 5e-5
SAMPLED_STEPS, SAVE_STRIDE, MOVIE_SECONDS, EXTENSION_FACTOR = 16, 10, 6.0, 1.5
INITIAL_IMAGE_WEIGHT, TRAJECTORY_WEIGHT, VIEWPORT_WEIGHT = 0.5, 0.5, 1.0
FD_EPSILONS = (1e-8, 1e-9, 1e-10)
OUTPUT = ROOT / 'results' / 'retention_pic'
RESUME = False
TARGET_FLOOR = 0.05
TARGET = load_target(IMAGE, TARGET_SIZE, floor=TARGET_FLOOR)
SETTINGS = dict(
    size=TARGET_SIZE, particles=PARTICLES, field_grid=FIELD_GRID, steps=STEPS,
    horizons=HORIZONS, iterations=STAGE_ITERATIONS, rates=STAGE_RATES, optimizer=OPTIMIZER,
    seed=SEED, length=LENGTH, velocity_range=VELOCITY_RANGE, max_speed=MAX_SPEED,
    kernel_width=KERNEL_WIDTH, timestep_ratio=TIMESTEP_RATIO, debye_ratio=DX_OVER_DEBYE,
    thermal_speed=THERMAL_SPEED, initialization=INITIALIZATION,
    two_stream_drift=TWO_STREAM_DRIFT, two_stream_thermal=TWO_STREAM_THERMAL,
    two_stream_perturbation=TWO_STREAM_PERTURBATION, sampled_steps=SAMPLED_STEPS,
    initial_image_weight=INITIAL_IMAGE_WEIGHT, trajectory_weight=TRAJECTORY_WEIGHT,
    viewport_weight=VIEWPORT_WEIGHT, save_stride=SAVE_STRIDE, movie_seconds=MOVIE_SECONDS,
    extension_factor=EXTENSION_FACTOR, fd_epsilons=FD_EPSILONS, title='Particle-in-cell · retention',
)

def pic_problem(horizon):
    return make_pic(TARGET, PARTICLES, horizon, LENGTH, VELOCITY_RANGE, seed=SEED,
        kernel_width=KERNEL_WIDTH, timestep_ratio=TIMESTEP_RATIO,
        debye_ratio=DX_OVER_DEBYE, thermal_speed=THERMAL_SPEED,
        max_speed=MAX_SPEED, field_grid_points=FIELD_GRID, initialization=INITIALIZATION,
        two_stream_drift=TWO_STREAM_DRIFT, two_stream_thermal=TWO_STREAM_THERMAL,
        two_stream_perturbation=TWO_STREAM_PERTURBATION)

first_problem = pic_problem(HORIZONS[0])
initial_parameters = first_problem[0]
checkpoint = StageCheckpoint(OUTPUT, initial_parameters, TARGET, SETTINGS, resume=RESUME)
stage_loss = None
for stage_index, (horizon, iterations, rate) in enumerate(zip(HORIZONS, STAGE_ITERATIONS, STAGE_RATES)):
    problem = first_problem if stage_index == 0 else pic_problem(horizon)
    _, run_simulation, image_from_phase_space, parameter_image, initial_electric_field = problem
    sample_indices = np.linspace(0, horizon - 1, min(SAMPLED_STEPS, horizon)).astype(int)

    @jax.checkpoint
    def stage_loss(parameters):
        trajectory = run_simulation(parameters, sample_indices=sample_indices)
        initial_image = parameter_image(parameters)
        sampled_images = jax.lax.map(
            lambda state: image_from_phase_space(state[0], state[1]),
            (trajectory['positions'][:, :, 0], trajectory['velocities'][:, :, 0]))
        initial_error = jnp.mean((initial_image - TARGET)**2)
        trajectory_error = jnp.mean((sampled_images - TARGET[None])**2)
        initial_vx = MAX_SPEED * jnp.tanh(parameters[:, 1])
        initial_outside = jnp.maximum(jnp.abs(initial_vx) - VELOCITY_RANGE, 0.0) / VELOCITY_RANGE
        viewport_error = (jnp.sum(initial_outside**2) + trajectory['viewport_penalty_sum'])
        viewport_error /= (horizon + 1) * PARTICLES
        return (INITIAL_IMAGE_WEIGHT*initial_error + TRAJECTORY_WEIGHT*trajectory_error
                + VIEWPORT_WEIGHT*viewport_error)

    if stage_index < checkpoint.next_stage:
        continue
    start_loss = float(stage_loss(checkpoint.parameters))
    print(f'Retention objective through {horizon} steps with {len(sample_indices)} samples; '
          f'{iterations} {OPTIMIZER} updates at {rate:g}; start loss={start_loss:.6g}.', flush=True)
    parameters, history = optimize(stage_loss, checkpoint.parameters, iterations, rate, method=OPTIMIZER)
    print(f'Stage objective={float(stage_loss(parameters)):.6g}.', flush=True)
    checkpoint.save(parameters, history)

extended_problem = pic_problem(round(STEPS * EXTENSION_FACTOR))
save_pic_artifacts(
    TARGET, initial_parameters, checkpoint.parameters, checkpoint.history, OUTPUT,
    run_simulation, extended_problem[1], image_from_phase_space, parameter_image, initial_electric_field,
    SETTINGS, checkpoint.lengths, stage_loss,
)
