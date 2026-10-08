"""Fit a 1D1V SPECTRAX distribution that keeps the target over time."""

from pathlib import Path
import sys
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import load_target, make_vlasov, optimize, save_results

IMAGE = Path(__file__).resolve().parents[1] / 'W7X-Spulen_Plasma_blau_gelb.jpg'
SIZE, MODES, VELOCITY_RANGE, DOMAIN_LENGTH = 32, 32, 4.0, 100.0
SNAPSHOTS, T_FINAL, DT = 9, 60.0, 0.01
MOVIE_FRAMES = 65
ITERATIONS, LEARNING_RATE, POSITIVITY_WEIGHT = 8, 0.005, 100000.0
COLLISION_RATE, ALPHA_X, IMAGE_CONTRAST = 0.1, 2.0, 0.005
INITIAL_WEIGHT = 0.5
OUTPUT = Path(__file__).resolve().parents[1] / 'results' / 'retention_vlasov'

print('Loading target with zero kinetic tails and setting 1D1V input values...', flush=True)
target = load_target(IMAGE, SIZE, floor=0.0)
initial, run, initial_fourier, velocities, image_from_parameters, image_at_velocity, background = make_vlasov(
    target, SNAPSHOTS, T_FINAL, DT, VELOCITY_RANGE, MODES, alpha_x=ALPHA_X,
    domain_length=DOMAIN_LENGTH, collision_rate=COLLISION_RATE, neutral_initial=True,
    contrast=IMAGE_CONTRAST,
)
print('Optimizing SPECTRAX initial moments across all saved snapshots...', flush=True)
start = time.time()


def loss(parameters):
    _, frames, _ = run(parameters)
    signal = (frames - background(velocities)[None, :, None])/IMAGE_CONTRAST
    temporal_error = jnp.mean(jnp.square(signal - target[None, :, :]))
    positive_error = jnp.mean(jnp.square(jnp.minimum(frames, 0.0)))
    initial_error = jnp.mean(jnp.square(image_from_parameters(parameters) - target))
    tail_v = jnp.linspace(-8*VELOCITY_RANGE, 8*VELOCITY_RANGE, 129)
    tail = background(tail_v)[:, None] + IMAGE_CONTRAST*image_at_velocity(parameters, tail_v)
    tail_error = jnp.mean(jnp.square(jnp.minimum(tail, 0.0)))
    return (1-INITIAL_WEIGHT)*temporal_error + INITIAL_WEIGHT*initial_error + POSITIVITY_WEIGHT*(positive_error+tail_error)


parameters, history = optimize(loss, initial, ITERATIONS, LEARNING_RATE, method='l-bfgs-b')
times, frames, electric_fields = run(parameters, save_count=MOVIE_FRAMES)
baseline_times, baseline, baseline_electric_fields = run(initial, save_count=MOVIE_FRAMES)
print(f'SPECTRAX retention fit finished in {time.time()-start:.1f}s; objective={float(loss(parameters)):.6g}')
physical_target = background(velocities)[:, None] + IMAGE_CONTRAST*target
save_results(physical_target, frames[0], frames, history, OUTPUT,
             parameters=initial_fourier(parameters), times=times,
             baseline=baseline, extent=(0, DOMAIN_LENGTH, -VELOCITY_RANGE, VELOCITY_RANGE),
             labels=('x', 'vx'), title='Self-consistent Vlasov · retention',
             electric_fields=electric_fields, baseline_electric_fields=baseline_electric_fields,
             time_label='ωₚₑ t', time_scale=1.0, field_label='Ex',
             density_background=background(velocities)[:, None], image_contrast=IMAGE_CONTRAST)
