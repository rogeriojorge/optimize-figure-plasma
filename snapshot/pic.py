"""Optimize an initial PIC particle state for a final phase-space image."""

from pathlib import Path
import sys

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers import (StageCheckpoint, load_target, make_pic, optimize,
                     save_pic_artifacts)  # noqa: E402

# Editable model, optimizer, objective, and artifact inputs.
IMAGE = ROOT / 'W7X-Spulen_Plasma_blau_gelb.jpg'
TARGET_SIZE, PARTICLES, FIELD_GRID = 256, 4096, 64
STEPS = 2000
HORIZONS = (200, 600, 1200, STEPS)
STAGE_ITERATIONS, STAGE_RATES = (50, 50, 50, 100), (1e-3, 5e-4, 2e-4, 1e-4)
OPTIMIZER, SEED = 'adam', 71
LENGTH, VELOCITY_RANGE, MAX_SPEED = 1.0, 0.20, 0.90
KERNEL_WIDTH, TIMESTEP_RATIO, DX_OVER_DEBYE, THERMAL_SPEED = 3.0, 0.5, 1.0, 0.10
INITIALIZATION = 'image'
TWO_STREAM_DRIFT, TWO_STREAM_THERMAL, TWO_STREAM_PERTURBATION = 0.20, 0.05, 5e-5
FINAL_IMAGE_WEIGHT = 1.0
SAVE_STRIDE, MOVIE_SECONDS, EXTENSION_FACTOR = 10, 6.0, 1.5
FD_EPSILONS = (1e-8, 1e-9, 1e-10)
OUTPUT = ROOT / 'results' / 'snapshot_pic'
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
    two_stream_perturbation=TWO_STREAM_PERTURBATION, final_image_weight=FINAL_IMAGE_WEIGHT,
    save_stride=SAVE_STRIDE, movie_seconds=MOVIE_SECONDS,
    extension_factor=EXTENSION_FACTOR, fd_epsilons=FD_EPSILONS, title='Particle-in-cell · snapshot',
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

    @jax.checkpoint
    def stage_loss(parameters):
        trajectory = run_simulation(parameters, endpoint_only=True)
        final_image = image_from_phase_space(
            trajectory['positions'][-1, :, 0], trajectory['velocities'][-1, :, 0])
        return FINAL_IMAGE_WEIGHT * jnp.mean((final_image - TARGET)**2)

    if stage_index < checkpoint.next_stage:
        continue
    start_loss = float(stage_loss(checkpoint.parameters))
    print(f'Snapshot objective through {horizon} steps; {iterations} {OPTIMIZER} updates '
          f'at {rate:g}; start loss={start_loss:.6g}.', flush=True)
    parameters, history = optimize(stage_loss, checkpoint.parameters, iterations, rate, method=OPTIMIZER)
    print(f'Stage objective={float(stage_loss(parameters)):.6g}.', flush=True)
    checkpoint.save(parameters, history)

extended_problem = pic_problem(round(STEPS * EXTENSION_FACTOR))
save_pic_artifacts(
    TARGET, initial_parameters, checkpoint.parameters, checkpoint.history, OUTPUT,
    run_simulation, extended_problem[1], image_from_phase_space, parameter_image,
    initial_electric_field, SETTINGS, checkpoint.lengths, stage_loss,
)
