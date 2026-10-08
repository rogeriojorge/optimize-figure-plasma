"""Fit a 1D1V SPECTRAX initial distribution to the target at final time."""

from pathlib import Path
import sys
import time
import os
import hashlib
import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import load_target, make_vlasov, optimize, save_results

IMAGE = Path(__file__).resolve().parents[1] / 'W7X-Spulen_Plasma_blau_gelb.jpg'
VELOCITY_BINS, SPATIAL_BINS = 320, 96
MODES, VELOCITY_RANGE, DOMAIN_LENGTH = 64, 4.0, 250.0
WIDE_VELOCITY_RANGE = 12.0
SNAPSHOTS, T_FINAL, DT = 9, 60.0, 0.04
HORIZONS = (10.0, 20.0, 40.0, T_FINAL)
MOVIE_FRAMES = 301
STAGE_ITERATIONS, LEARNING_RATE, POSITIVITY_WEIGHT = (20, 30, 40, 80), 0.005, 1000.0
COLLISION_RATE, ALPHA_X, IMAGE_CONTRAST = 0.1, 2.0, 0.005
OUTPUT = Path(__file__).resolve().parents[1] / 'results' / 'snapshot_vlasov'
EXTENDED_FINAL_TIME = 90.0
EXTENDED_MOVIE_FRAMES = 451
RESUME = False

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
        tail_v = jnp.linspace(-WIDE_VELOCITY_RANGE, WIDE_VELOCITY_RANGE, 129)
        tail = background(tail_v)[:, None] + IMAGE_CONTRAST*image_at_velocity(parameters, tail_v)
        tail_error = jnp.mean(jnp.square(jnp.minimum(tail, 0.0)))
        return fit_error + POSITIVITY_WEIGHT * (negative_error + tail_error)
    return loss


stage_horizons = HORIZONS
assert len(STAGE_ITERATIONS) == len(HORIZONS)
assert sum(STAGE_ITERATIONS) >= 1
run_signature = np.asarray([VELOCITY_BINS, SPATIAL_BINS, MODES, VELOCITY_RANGE, DOMAIN_LENGTH,
                            SNAPSHOTS, T_FINAL, DT, COLLISION_RATE, ALPHA_X, IMAGE_CONTRAST,
                            LEARNING_RATE, POSITIVITY_WEIGHT, *HORIZONS, *STAGE_ITERATIONS,
                            EXTENDED_FINAL_TIME, EXTENDED_MOVIE_FRAMES,
                            WIDE_VELOCITY_RANGE], dtype=float)
target_signature = hashlib.sha256(np.asarray(target).tobytes()).hexdigest()
stage_checkpoint = OUTPUT / 'optimization_checkpoint.npz'
OUTPUT.mkdir(parents=True, exist_ok=True)

# Results archives are ignored unless the user explicitly resumes a matching stage checkpoint.
parameters = initial
histories, stage_lengths, first_stage = [], [], 0
if RESUME and stage_checkpoint.exists():
    with np.load(stage_checkpoint) as saved:
        if (np.array_equal(saved['run_signature'], run_signature)
                and str(saved['target_signature']) == target_signature):
            parameters = jnp.asarray(saved['parameters'])
            histories = list(saved['histories'].tolist())
            stage_lengths = list(saved['stage_lengths'].astype(int))
            first_stage = int(saved['next_stage'])
            print(f'Resuming after optimization stage {first_stage}.', flush=True)
for stage_index, (horizon, stage_iterations) in enumerate(zip(stage_horizons, STAGE_ITERATIONS)):
    if stage_index < first_stage:
        continue
    model = build_model(horizon)
    stage_initial, run, stage_initial_fourier, stage_velocities, stage_image, stage_at_velocity, stage_background = model
    # All stages share the same projection and scales, so the previous optimum is a valid warm start.
    loss = objective_for(run, stage_image, stage_at_velocity, stage_background, horizon)
    print(f'Optimizing through t={horizon:g} with {stage_iterations} L-BFGS-B iterations...', flush=True)
    parameters, history = optimize(loss, parameters, stage_iterations, LEARNING_RATE, method='l-bfgs-b')
    histories.extend(history.tolist())
    stage_lengths.append(len(history))
    print(f't={horizon:g}: objective={float(loss(parameters)):.6g}', flush=True)
    temp_checkpoint = stage_checkpoint.with_suffix('.npz.tmp')
    with temp_checkpoint.open('wb') as stream:
        np.savez_compressed(stream, run_signature=run_signature, parameters=np.asarray(parameters),
                            histories=np.asarray(histories), stage_lengths=np.asarray(stage_lengths),
                            next_stage=stage_index+1, target_signature=target_signature)
    os.replace(temp_checkpoint, stage_checkpoint)

_, run, initial_fourier, velocities, image_from_parameters, image_at_velocity, background = build_model(T_FINAL)
loss = objective_for(run, image_from_parameters, image_at_velocity, background, T_FINAL)
times, frames, electric_fields = run(parameters, save_count=MOVIE_FRAMES)
baseline_times, baseline, baseline_electric_fields = run(initial, save_count=MOVIE_FRAMES)
print(f'Running the same optimized initial condition through t={EXTENDED_FINAL_TIME:g}...', flush=True)
_, extended_run, _, _, _, _, _ = build_model(EXTENDED_FINAL_TIME)
extended_times, extended_frames, extended_electric_fields = extended_run(
    parameters, save_count=EXTENDED_MOVIE_FRAMES)
if not np.allclose(np.asarray(extended_times[:len(times)]), np.asarray(times), rtol=0, atol=1e-10):
    raise RuntimeError('The standard and extended Vlasov trajectories do not share their saved time grid.')
extended_prefix_error = float(jnp.max(jnp.abs(extended_frames[:len(frames)]-frames)))
if extended_prefix_error > 1e-6:
    raise RuntimeError(f'The extended trajectory differs before t={T_FINAL:g} (max error {extended_prefix_error:.3g}).')
print(f'Extended-run prefix matches the standard trajectory through t={T_FINAL:g}; '
      f'maximum phase-space difference={extended_prefix_error:.3g}.', flush=True)
print(f'SPECTRAX snapshot fit finished in {time.time()-start:.1f}s; objective={float(loss(parameters)):.6g}')
print('Checking broad-velocity positivity, time-step refinement, and the AD gradient...', flush=True)
wide_velocities = jnp.linspace(-WIDE_VELOCITY_RANGE, WIDE_VELOCITY_RANGE, 257)
_, _, _, wide_frames = run(parameters, save_count=65, velocity_samples=wide_velocities)
wide_minimum = float(jnp.min(wide_frames))
wide_negative_fraction = float(jnp.mean(wide_frames < 0.0))
viewport_minimum = float(jnp.min(frames))
viewport_negative_fraction = float(jnp.mean(frames < 0.0))
wide_negative_mass_fraction = float(jnp.sum(jnp.maximum(-wide_frames, 0.0))/
                                     jnp.maximum(jnp.sum(jnp.maximum(wide_frames, 0.0)), 1e-30))
_, half_step_frames, _ = run(parameters, save_count=2, time_step=DT/2)
signal_scale = jnp.sqrt(jnp.mean(jnp.square((frames[-1]-background(velocities)[:, None])/IMAGE_CONTRAST)))
step_error = jnp.sqrt(jnp.mean(jnp.square((half_step_frames[-1]-frames[-1])/IMAGE_CONTRAST))) / jnp.maximum(signal_scale, 1e-12)
value, gradient = jax.value_and_grad(loss)(parameters)
direction = jnp.sin(jnp.arange(parameters.size, dtype=parameters.dtype).reshape(parameters.shape)*0.37 + 0.2)
direction = direction / jnp.linalg.norm(direction)
epsilon = 1e-5
finite_difference = (loss(parameters+epsilon*direction)-loss(parameters-epsilon*direction))/(2*epsilon)
ad_direction = jnp.vdot(gradient, direction).real
fd_relative_error = jnp.abs(ad_direction-finite_difference)/jnp.maximum(jnp.abs(finite_difference), 1e-12)
print(f'wide-v positivity: min={wide_minimum:.6g}, negative fraction={wide_negative_fraction:.3%}; '
      f'viewport min={viewport_minimum:.6g}, negative fraction={viewport_negative_fraction:.3%}, '
      f'wide integrated negative fraction={wide_negative_mass_fraction:.3%}; '
      f'dt/2 endpoint relative signal RMS={float(step_error):.3%}; '
      f'AD directional derivative={float(ad_direction):.6g}, FD={float(finite_difference):.6g}, '
      f'relative error={float(fd_relative_error):.3%}', flush=True)
physical_target = background(velocities)[:, None] + IMAGE_CONTRAST*target
initial_signal_mse = jnp.mean(jnp.square(image_from_parameters(parameters)-target))
save_results(physical_target, frames[0], frames, histories, OUTPUT,
             parameters=initial_fourier(parameters), times=times,
             baseline=baseline, extent=(0, DOMAIN_LENGTH, -VELOCITY_RANGE, VELOCITY_RANGE),
             labels=('x', 'vx'), title='Vlasov image perturbation · snapshot',
             electric_fields=electric_fields, baseline_electric_fields=baseline_electric_fields,
             time_label='ωₚₑ t', time_scale=1.0, field_label='Ex',
             density_background=background(velocities)[:, None], image_contrast=IMAGE_CONTRAST,
             stage_lengths=stage_lengths, stage_times=stage_horizons,
             extended_frames=extended_frames, extended_times=extended_times,
             extended_electric_fields=extended_electric_fields,
             metadata={
                 'coefficient_density_scale': float(initial_fourier.density_scale),
                 'diagnostic_initial_signal_mse': float(initial_signal_mse),
                 'diagnostic_extended_prefix_max_error': extended_prefix_error,
                 'diagnostic_viewport_minimum': viewport_minimum,
                 'diagnostic_viewport_negative_fraction': viewport_negative_fraction,
                 'diagnostic_wide_velocity_minimum': wide_minimum,
                 'diagnostic_wide_velocity_negative_fraction': wide_negative_fraction,
                 'diagnostic_wide_integrated_negative_fraction': wide_negative_mass_fraction,
                 'diagnostic_dt_half_relative_signal_rms': float(step_error),
                 'diagnostic_ad_fd_relative_error': float(fd_relative_error),
                 'diagnostic_electric_field_rms': float(jnp.sqrt(jnp.mean(electric_fields**2))),
                 'diagnostic_mass_relative_range': float(
                     (frames.mean(axis=(-2, -1)).max()-frames.mean(axis=(-2, -1)).min()) /
                     jnp.maximum(jnp.abs(frames[0].mean()), 1e-30)),
             })
stage_checkpoint.unlink(missing_ok=True)
