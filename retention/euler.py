"""Optimize an initial fluid state to retain an image over time."""
from pathlib import Path
from functools import partial
import sys
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers import (StageCheckpoint, density_checks, gradient_check, load_target,
                     optimize, save_euler_fit, euler_evolve)

IMAGE, OUTPUT = ROOT / 'W7X-Spulen_Plasma_blau_gelb.jpg', ROOT / 'results' / 'retention_euler'
SIZE, STEPS = 128, 192
HORIZONS = (0.10, 0.20, 0.35, 0.50)
ITERATIONS = (30, 40, 60, 100)
SOUND_SPEED, VELOCITY_SCALE, TARGET_FLOOR = 0.45, 0.80, 0.05
OPTIMIZER, LEARNING_RATE = 'adam', 0.015
INITIAL_WEIGHT, POLISH_ITERATIONS, POLISH_OPTIMIZER = 0.5, 60, 'l-bfgs-b'
EXTENSION_FACTOR, MOVIE_SECONDS, CHECKS, RESUME = 1.5, 6.0, True, False

target = load_target(IMAGE, SIZE, floor=TARGET_FLOOR)
initial = jnp.stack((jnp.log(target), jnp.zeros_like(target), jnp.zeros_like(target)))
evolve = partial(euler_evolve, sound_speed=SOUND_SPEED)

def conserved(parameters):
    rho = jnp.exp(parameters[0])
    vx, vy = VELOCITY_SCALE*jnp.tanh(parameters[1:])
    return jnp.stack((rho, rho*vx, rho*vy))

settings = dict(size=SIZE, steps=STEPS, horizons=HORIZONS, iterations=ITERATIONS,
                sound_speed=SOUND_SPEED, velocity_scale=VELOCITY_SCALE,
                optimizer=OPTIMIZER, learning_rate=LEARNING_RATE, initial_weight=INITIAL_WEIGHT,
                polish_iterations=POLISH_ITERATIONS, polish_optimizer=POLISH_OPTIMIZER)
checkpoint = StageCheckpoint(OUTPUT, initial, target, settings, resume=RESUME)
parameters = checkpoint.parameters
for stage, (horizon, iterations) in enumerate(zip(HORIZONS, ITERATIONS)):
    stage_steps = max(min(32, STEPS), round(STEPS*horizon/HORIZONS[-1]))
    stage_dt = horizon/stage_steps

    def loss(parameters):
        frames = evolve(conserved(parameters), stage_steps, stage_dt)
        return jnp.mean((frames-target)**2) + INITIAL_WEIGHT*jnp.mean((frames[0]-target)**2)

    if stage < checkpoint.next_stage:
        continue
    print(f'Horizon T={horizon:g}: {iterations} {OPTIMIZER} iterations', flush=True)
    parameters, history = optimize(loss, parameters, iterations, LEARNING_RATE, method=OPTIMIZER)
    if stage == len(HORIZONS)-1 and POLISH_ITERATIONS:
        parameters, polish = optimize(loss, parameters, POLISH_ITERATIONS, LEARNING_RATE, method=POLISH_OPTIMIZER)
        history = np.r_[history, polish[1:]]
    density_checks(evolve(conserved(parameters), stage_steps, stage_dt), f'T={horizon:g}')
    checkpoint.save(parameters, history)

metadata = {'gradient_check_error': gradient_check(loss, parameters, epsilon=1e-4)} if CHECKS else {}
save_euler_fit(target, conserved(parameters), conserved(initial), evolve, checkpoint.history, OUTPUT,
    steps=STEPS, dt=HORIZONS[-1]/STEPS, horizons=HORIZONS, stage_lengths=checkpoint.lengths,
    title='Isothermal Euler · retention', extent=(0, 2*np.pi, 0, 2*np.pi),
    extension_factor=EXTENSION_FACTOR, movie_seconds=MOVIE_SECONDS, checks=CHECKS, metadata=metadata)
