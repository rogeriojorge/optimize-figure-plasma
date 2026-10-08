"""Fit a 1D1V SPECTRAX initial distribution to the target at final time."""

from pathlib import Path
import sys
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import load_target, make_vlasov, optimize, save_results

IMAGE = Path(__file__).resolve().parents[1] / 'W7X-Spulen_Plasma_blau_gelb.jpg'
VELOCITY_BINS, SPATIAL_BINS = 256, 128
MODES, VELOCITY_RANGE, DOMAIN_LENGTH = 32, 4.0, 200.0
SNAPSHOTS, T_FINAL, DT = 9, 60.0, 0.04
HORIZONS = (10.0, 20.0, 40.0, T_FINAL)
MOVIE_FRAMES = 301
ITERATIONS, LEARNING_RATE, POSITIVITY_WEIGHT = 120, 0.005, 1000.0
COLLISION_RATE, ALPHA_X, IMAGE_CONTRAST = 0.1, 2.0, 0.005
OUTPUT = Path(__file__).resolve().parents[1] / 'results' / 'snapshot_vlasov'

print('Loading target with zero image tails and setting 1D1V input values...', flush=True)
target = load_target(IMAGE, (VELOCITY_BINS, SPATIAL_BINS), floor=0.0)

def build_model(horizon):
    return make_vlasov(
        target, SNAPSHOTS, horizon, DT, VELOCITY_RANGE, MODES, alpha_x=ALPHA_X,
        domain_length=DOMAIN_LENGTH, collision_rate=COLLISION_RATE, neutral_initial=True,
        contrast=IMAGE_CONTRAST,
    )

initial, _, initial_fourier, velocities, image_from_parameters, image_at_velocity, background = build_model(T_FINAL)
print(f'Optimizing with horizon continuation at normalized plasma times {HORIZONS}...', flush=True)
start = time.time()


def objective_for(run, image_from_parameters, image_at_velocity, background, horizon):
    def loss(parameters):
        _, frames, _ = run(parameters)
        signal = (frames - background(velocities)[None, :, None])/IMAGE_CONTRAST
        fit_error = jnp.mean(jnp.square(signal[-1] - target))
        negative_error = jnp.mean(jnp.square(jnp.minimum(frames, 0.0)))
        tail_v = jnp.linspace(-8*VELOCITY_RANGE, 8*VELOCITY_RANGE, 129)
        tail = background(tail_v)[:, None] + IMAGE_CONTRAST*image_at_velocity(parameters, tail_v)
        tail_error = jnp.mean(jnp.square(jnp.minimum(tail, 0.0)))
        return fit_error + POSITIVITY_WEIGHT * (negative_error + tail_error)
    return loss


stage_horizons = HORIZONS
stage_iterations = ITERATIONS // len(stage_horizons)
parameters = initial
histories, stage_lengths = [], []
for horizon in stage_horizons:
    model = build_model(horizon)
    stage_initial, run, stage_initial_fourier, stage_velocities, stage_image, stage_at_velocity, stage_background = model
    # All stages share the same projection and scales, so the previous optimum is a valid warm start.
    loss = objective_for(run, stage_image, stage_at_velocity, stage_background, horizon)
    print(f'Optimizing through t={horizon:g} with {stage_iterations} L-BFGS-B iterations...', flush=True)
    parameters, history = optimize(loss, parameters, stage_iterations, LEARNING_RATE, method='l-bfgs-b')
    histories.extend(history.tolist())
    stage_lengths.append(len(history))
    print(f't={horizon:g}: objective={float(loss(parameters)):.6g}', flush=True)

_, run, initial_fourier, velocities, image_from_parameters, image_at_velocity, background = build_model(T_FINAL)
loss = objective_for(run, image_from_parameters, image_at_velocity, background, T_FINAL)
times, frames, electric_fields = run(parameters, save_count=MOVIE_FRAMES)
baseline_times, baseline, baseline_electric_fields = run(initial, save_count=MOVIE_FRAMES)
print(f'SPECTRAX snapshot fit finished in {time.time()-start:.1f}s; objective={float(loss(parameters)):.6g}')
physical_target = background(velocities)[:, None] + IMAGE_CONTRAST*target
save_results(physical_target, frames[0], frames, histories, OUTPUT,
             parameters=initial_fourier(parameters), times=times,
             baseline=baseline, extent=(0, DOMAIN_LENGTH, -VELOCITY_RANGE, VELOCITY_RANGE),
             labels=('x', 'vx'), title='Vlasov image perturbation · snapshot',
             electric_fields=electric_fields, baseline_electric_fields=baseline_electric_fields,
             time_label='ωₚₑ t', time_scale=1.0, field_label='Ex',
             density_background=background(velocities)[:, None], image_contrast=IMAGE_CONTRAST,
             stage_lengths=stage_lengths, stage_times=stage_horizons)
