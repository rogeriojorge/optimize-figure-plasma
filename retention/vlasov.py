"""Fit a 1D1V SPECTRAX distribution that keeps the target over time."""
from pathlib import Path
import sys
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import (StageCheckpoint, load_target, make_vlasov, optimize_vlasov_stages,
                     save_vlasov_fit)

# Editable physical, optimization, and output inputs.
IMAGE = Path(__file__).resolve().parents[1] / 'W7X-Spulen_Plasma_blau_gelb.jpg'
VELOCITY_BINS, SPATIAL_BINS, MODES = 320, 160, 32
VELOCITY_RANGE, WIDE_VELOCITY_RANGE, DOMAIN_LENGTH = 4.0, 12.0, 250.0
SNAPSHOTS, T_FINAL, DT = 9, 60.0, 0.04
HORIZONS = (10.0, 20.0, 40.0, T_FINAL)
STAGE_ITERATIONS, OPTIMIZER, LEARNING_RATE = (20, 30, 40, 80), 'l-bfgs-b', 0.005
INITIAL_WEIGHT, POSITIVITY_WEIGHT = 0.5, 1000.0
COLLISION_RATE, ALPHA_X, IMAGE_CONTRAST = 0.1, 2.0, 0.005
ION_MASS, PROJECTION, NEUTRAL_INITIAL = 1836.0, 'moments', True
MOVIE_FRAMES, EXTENSION_FACTOR, MOVIE_SECONDS = 301, 1.5, 6.0
EXTENDED_FINAL_TIME = EXTENSION_FACTOR*T_FINAL
EXTENDED_MOVIE_FRAMES = round((MOVIE_FRAMES-1)*EXTENSION_FACTOR)+1
CHECKS, RESUME = True, False
OUTPUT = Path(__file__).resolve().parents[1] / 'results' / 'retention_vlasov'

target = load_target(IMAGE, (VELOCITY_BINS, SPATIAL_BINS), floor=0.0)
def build_model(horizon):
    return make_vlasov(target, SNAPSHOTS, horizon, DT, VELOCITY_RANGE, MODES,
        alpha_x=ALPHA_X, domain_length=DOMAIN_LENGTH, collision_rate=COLLISION_RATE,
        ion_mass=ION_MASS, projection=PROJECTION, neutral_initial=NEUTRAL_INITIAL, contrast=IMAGE_CONTRAST)

def objective(model, horizon):
    _, run, _, velocities, image, image_at_velocity, background = model
    def loss(parameters):
        _, frames, _ = run(parameters)
        signal = (frames-background(velocities)[None, :, None])/IMAGE_CONTRAST
        temporal = jnp.mean(jnp.square(signal-target[None, :, :]))
        initial = jnp.mean(jnp.square(image(parameters)-target))
        negative = jnp.mean(jnp.square(jnp.minimum(frames, 0.0)))
        tail_v = jnp.linspace(-WIDE_VELOCITY_RANGE, WIDE_VELOCITY_RANGE, 129)
        tail = background(tail_v)[:, None]+IMAGE_CONTRAST*image_at_velocity(parameters, tail_v)
        tail_negative = jnp.mean(jnp.square(jnp.minimum(tail, 0.0)))
        return (1-INITIAL_WEIGHT)*temporal+INITIAL_WEIGHT*initial+POSITIVITY_WEIGHT*(negative+tail_negative)
    return loss

settings = dict(Nv=VELOCITY_BINS, Nx=SPATIAL_BINS, modes=MODES, velocity_range=VELOCITY_RANGE,
    domain_length=DOMAIN_LENGTH, ion_mass=ION_MASS, projection=PROJECTION,
    neutral_initial=NEUTRAL_INITIAL, snapshots=SNAPSHOTS, final_time=T_FINAL, dt=DT, horizons=HORIZONS,
    iterations=STAGE_ITERATIONS, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
    initial_weight=INITIAL_WEIGHT, positivity_weight=POSITIVITY_WEIGHT, collision_rate=COLLISION_RATE,
    alpha_x=ALPHA_X, contrast=IMAGE_CONTRAST, wide_velocity_range=WIDE_VELOCITY_RANGE,
    extended_final_time=EXTENDED_FINAL_TIME, extended_frames=EXTENDED_MOVIE_FRAMES)
initial_model = build_model(T_FINAL)
checkpoint = StageCheckpoint(OUTPUT, initial_model[0], target, settings, resume=RESUME)
parameters, history, stage_lengths = optimize_vlasov_stages(
    build_model, objective, checkpoint, HORIZONS, STAGE_ITERATIONS, LEARNING_RATE, OPTIMIZER)
model = build_model(T_FINAL)
save_vlasov_fit(target, parameters, initial_model[0], model, build_model,
    objective(model, T_FINAL), OUTPUT, domain_length=DOMAIN_LENGTH, velocity_range=VELOCITY_RANGE,
    wide_velocity_range=WIDE_VELOCITY_RANGE, contrast=IMAGE_CONTRAST, dt=DT,
    frames_count=MOVIE_FRAMES, extended_final_time=EXTENDED_FINAL_TIME,
    extended_frames_count=EXTENDED_MOVIE_FRAMES, horizons=HORIZONS, histories=history,
    stage_lengths=stage_lengths, title='Vlasov image perturbation · retention', movie_seconds=MOVIE_SECONDS, checks=CHECKS)
