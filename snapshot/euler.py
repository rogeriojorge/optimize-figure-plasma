"""Draw an image from uniform fluid by optimizing its initial velocity."""
from pathlib import Path
from functools import partial
import sys
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers import (StageCheckpoint, density_checks, gradient_check, load_target,
                     optimize, save_euler_fit, euler_evolve_muscl)

IMAGE, OUTPUT = ROOT / 'W7X-Spulen_Plasma_blau_gelb.jpg', ROOT / 'results' / 'snapshot_euler'
SIZE, STEPS = 128, 352
HORIZONS = (0.10, 0.20, 0.30, 0.43)
ITERATIONS = (30, 40, 60, 100)
SOUND_SPEED, VELOCITY_SCALE, TARGET_FLOOR = 1.0, 0.80, 0.05
OPTIMIZER, LEARNING_RATE = 'l-bfgs-b', 0.01
IMAGE_CONTRAST = 0.10
EXTENSION_FACTOR, MOVIE_SECONDS, CHECKS, RESUME = 1.5, 6.0, True, False

target = load_target(IMAGE, SIZE, floor=TARGET_FLOOR)
target = 1.0 + IMAGE_CONTRAST*(target-1.0)
initial = jnp.zeros((2, SIZE, SIZE))
evolve = partial(euler_evolve_muscl, dx=1.0/SIZE, sound_speed=SOUND_SPEED)

def conserved(parameters):
    vx, vy = VELOCITY_SCALE*jnp.tanh(parameters)
    return jnp.stack((jnp.ones_like(vx), vx, vy))

settings = dict(size=SIZE, steps=STEPS, horizons=HORIZONS, iterations=ITERATIONS,
                sound_speed=SOUND_SPEED, velocity_scale=VELOCITY_SCALE,
                optimizer=OPTIMIZER, learning_rate=LEARNING_RATE, contrast=IMAGE_CONTRAST)
checkpoint = StageCheckpoint(OUTPUT, initial, target, settings, resume=RESUME)
parameters = checkpoint.parameters
for stage, (horizon, iterations) in enumerate(zip(HORIZONS, ITERATIONS)):
    stage_steps = max(min(32, STEPS), round(STEPS*horizon/HORIZONS[-1]))
    stage_dt = horizon/stage_steps

    def loss(parameters):
        frames = evolve(conserved(parameters), stage_steps, stage_dt)
        return jnp.mean((frames[-1]-target)**2)

    if stage < checkpoint.next_stage:
        continue
    print(f'Horizon T={horizon:g}: {iterations} {OPTIMIZER} iterations', flush=True)
    parameters, history = optimize(loss, parameters, iterations, LEARNING_RATE, method=OPTIMIZER)
    density_checks(evolve(conserved(parameters), stage_steps, stage_dt), f'T={horizon:g}')
    checkpoint.save(parameters, history)

metadata = {'gradient_check_error': gradient_check(loss, parameters, epsilon=1e-4)} if CHECKS else {}
save_euler_fit(target, conserved(parameters), conserved(initial), evolve, checkpoint.history, OUTPUT,
    steps=STEPS, dt=HORIZONS[-1]/STEPS, horizons=HORIZONS, stage_lengths=checkpoint.lengths,
    title='Isothermal Euler · snapshot', extent=(0, 1, 0, 1),
    extension_factor=EXTENSION_FACTOR, movie_seconds=MOVIE_SECONDS, checks=CHECKS, metadata=metadata,
    density_background=1.0, image_contrast=IMAGE_CONTRAST)
