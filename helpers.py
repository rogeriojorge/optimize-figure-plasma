"""Image preparation, verbose optimization, and shared plotting."""
from pathlib import Path
from time import perf_counter
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_target(image, size, floor=0.05):
    """Letterbox inverted grayscale pixels, add a floor, normalize mean density."""
    image = Image.open(image).convert('RGBA')
    image = Image.alpha_composite(Image.new('RGBA', image.size, 'white'), image).convert('L')
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (size, size), color=255)
    canvas.paste(image, ((size-image.width)//2, (size-image.height)//2))
    density = floor + (255-np.asarray(canvas, dtype=float))/255
    if density.mean() <= 0:
        raise ValueError("The target has no nonwhite pixels; use a nonblank image or positive floor.")
    return jnp.asarray(density[::-1].copy()/density.mean())


def optimize(loss, initial, iterations, learning_rate, method="adam"):
    """Optimize with JAX autodiff and report progress, including compilation time."""
    if method in ('bfgs', 'l-bfgs-b'):
        from scipy.optimize import minimize
        from jax.flatten_util import ravel_pytree
        flat, unravel = ravel_pytree(initial)
        evaluate = jax.jit(jax.value_and_grad(lambda p: loss(unravel(p))))
        history, durations = [], []
        print(f'Compiling JAX objective/gradient for SciPy {method} ...', flush=True)
        def objective(p):
            start = perf_counter()
            value, gradient = evaluate(jnp.asarray(p))
            value, gradient = float(value), np.asarray(gradient)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise FloatingPointError('Nonfinite objective or gradient: reduce time step or parameter step.')
            durations.append(perf_counter()-start)
            return value, gradient
        def report(p):
            value, _ = objective(p)
            history.append(value)
            print(f'{len(history):4d}/{iterations} loss={value:.6g} evaluation={durations[-1]:.2f}s', flush=True)
        report(np.asarray(flat))
        result = minimize(objective, np.asarray(flat), jac=True,
                          method='BFGS' if method == 'bfgs' else 'L-BFGS-B', callback=report,
                          options={'maxiter': iterations, 'gtol': 1e-8})
        report(result.x)
        print(result.message, flush=True)
        return unravel(jnp.asarray(result.x)), np.asarray(history)
    optimizer = (optax.lbfgs(linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=5))
                 if method == "lbfgs" else optax.adam(learning_rate))
    evaluate = optax.value_and_grad_from_state(loss) if method == "lbfgs" else jax.value_and_grad(loss)
    state = optimizer.init(initial)
    @jax.jit
    def step(parameters, state):
        value, gradient = (evaluate(parameters, state=state) if method == "lbfgs"
                           else evaluate(parameters))
        updates, state = optimizer.update(gradient, state, parameters,
                                         value=value, grad=gradient, value_fn=loss)
        return optax.apply_updates(parameters, updates), state, value
    print(f'Compiling objective and gradient on {jax.devices()} ...', flush=True)
    history, best, best_loss = [], initial, float('inf')
    for iteration in range(iterations):
        start = perf_counter()
        updated, state, value = step(initial, state)
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite objective: reduce time step or learning rate.')
        if value < best_loss:
            best, best_loss = initial, value
        history.append(value)
        print(f'{iteration+1:4d}/{iterations} loss={value:.6g} elapsed={perf_counter()-start:.2f}s', flush=True)
        initial = updated
    value = float(loss(initial))
    if value < best_loss:
        best = initial
    history.append(value)
    return best, np.asarray(history)


def save_results(target, initial, frames, history, output, parameters=None, times=None,
                 baseline=None, extent=None, labels=('x', 'y'), title='Image optimization',
                 electric_fields=None, baseline_electric_fields=None, time_scale=1.0,
                 time_label='t', field_label='Ex', density_background=None, image_contrast=1.0):
    """Save arrays, endpoint/loss figures, and dynamics-only presentation movies."""
    from matplotlib.animation import FFMpegWriter
    import imageio_ffmpeg
    import subprocess
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target, initial, frames = map(np.asarray, (target, initial, frames))
    baseline = frames if baseline is None else np.asarray(baseline)
    times = np.arange(len(frames)) if times is None else np.asarray(times)
    fields = None if electric_fields is None else np.asarray(electric_fields)
    if not all(np.isfinite(a).all() for a in (frames, baseline)):
        raise FloatingPointError('Nonfinite simulation output.')
    if fields is not None and (len(fields) != len(frames) or not np.isfinite(fields).all()):
        raise ValueError('Electric fields must be finite and aligned with the density frames.')
    clock = times * float(time_scale)
    losses = np.mean((frames-target)**2, axis=(-2, -1))
    data = dict(target=target, initial=initial, frames=frames, baseline=baseline,
                loss=history, frame_mse=losses,
                baseline_mse=np.mean((baseline-target)**2, axis=(-2, -1)),
                frame_mass=frames.mean(axis=(-2, -1)), times=times,
                time_scale=float(time_scale), image_contrast=float(image_contrast),
                signal_mse=losses/float(image_contrast)**2,
                baseline_signal_mse=np.mean((baseline-target)**2, axis=(-2, -1))/float(image_contrast)**2)
    if fields is not None:
        data['electric_fields'] = fields
        if baseline_electric_fields is not None:
            data['baseline_electric_fields'] = np.asarray(baseline_electric_fields)
    if density_background is not None:
        data['density_background'] = np.asarray(density_background)
    if parameters is not None:
        for i, leaf in enumerate(jax.tree_util.tree_leaves(parameters)):
            data[f'parameters_{i}'] = np.asarray(leaf)
    np.savez_compressed(output/'results.npz', **data)
    style = {'font.family': 'DejaVu Sans', 'font.size': 12,
             'axes.spines.top': False, 'axes.spines.right': False}
    vmin = min(target.min(), np.quantile(frames, .001))
    vmax = max(target.max(), np.quantile(frames, .999))
    x = np.linspace(extent[0], extent[1], frames.shape[-1], endpoint=False) if extent else np.arange(frames.shape[-1])
    field_limit = max(np.max(np.abs(fields)), 1e-12) * 1.08 if fields is not None else None
    density_label = 'Density' if fields is None else 'f(x, vx)'
    color_options = dict(cmap='inferno', vmin=vmin, vmax=vmax)
    if density_background is not None:
        from matplotlib.colors import SymLogNorm
        density_background = np.asarray(density_background)
        limit = max(np.max(np.abs(frames-density_background)), np.max(np.abs(target-density_background)), 1e-12)
        color_options = dict(cmap='RdBu_r', norm=SymLogNorm(linthresh=.03*limit, linscale=.4, vmin=-limit, vmax=limit))
        density_label = (f'Density − {float(density_background):g}' if density_background.ndim == 0
                         else 'f − background') + ' (symlog scale)'
    with plt.rc_context(style):
        def phase(ax, image, caption):
            artist = ax.imshow(image if density_background is None else image-density_background,
                               origin='lower', extent=extent, aspect='auto',
                               interpolation='nearest', **color_options)
            ax.set(title=caption, xlabel=labels[0], ylabel=labels[1])
            return artist
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), layout='constrained')
        for ax, image, caption in zip(axes, [target, initial, baseline[-1], frames[-1]],
                ['Target', 'Optimized initial', 'Baseline final', 'Optimized final']):
            phase(ax, image, caption)
        fig.suptitle(title, fontweight='bold')
        fig.savefig(output/'comparison.png', dpi=200)
        plt.close(fig)
        fig, axes = plt.subplots(2 if fields is not None else 1, 2, figsize=(10, 6 if fields is not None else 4),
                                 layout='constrained', squeeze=False,
                                 gridspec_kw={'height_ratios': [3, 1]} if fields is not None else {})
        for column, i in enumerate([0, len(frames)-1]):
            artist = phase(axes[0, column], frames[i], f'{"Initial" if i == 0 else "Final"} · {time_label} = {clock[i]:.3g}')
            if fields is not None:
                axes[1, column].plot(x, fields[i], color='#2563eb', linewidth=2)
                axes[1, column].set(xlabel=labels[0], ylabel=field_label, ylim=(-field_limit, field_limit))
                axes[1, column].grid(alpha=.15)
        bar = fig.colorbar(artist, ax=axes[0, :], shrink=.8, label=density_label)
        if density_background is not None:
            ticks = np.array([-1, -.1, 0, .1, 1])*limit
            bar.set_ticks(ticks, labels=[f'{t:.2g}' for t in ticks])
        fig.suptitle(title, fontweight='bold')
        fig.savefig(output/'initial_final.png', dpi=200)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3.5), layout='constrained')
        ax.semilogy(np.maximum(history, 1e-16), color='#2563eb', linewidth=2)
        ax.set(xlabel='Optimization iteration', ylabel='Objective', title=title)
        ax.grid(alpha=.15)
        fig.savefig(output/'loss.png', dpi=200)
        plt.close(fig)
        if fields is None and parameters is not None and np.shape(parameters) == (3, *target.shape):
            rho, mx, my = np.asarray(parameters)
            u, v = mx/rho, my/rho
            fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
            speed = ax.imshow(np.hypot(u, v), origin='lower', extent=extent, cmap='viridis', aspect='auto')
            y = np.linspace(extent[2], extent[3], target.shape[0], endpoint=False)
            stride = max(1, target.shape[0]//16)
            ax.quiver(x[::stride], y[::stride], u[::stride, ::stride], v[::stride, ::stride], color='white', alpha=.7)
            ax.set(title='Optimized initial velocity', xlabel=labels[0], ylabel=labels[1])
            fig.colorbar(speed, ax=ax, label='Speed')
            fig.savefig(output/'initial_velocity.png', dpi=200)
            plt.close(fig)
        fig, axes = plt.subplots(2 if fields is not None else 1, 1, figsize=(12.8, 7.2),
                                 layout='constrained', squeeze=False,
                                 gridspec_kw={'height_ratios': [3, 1]} if fields is not None else {})
        image = phase(axes[0, 0], frames[0], '')
        bar = fig.colorbar(image, ax=axes[0, 0], shrink=.9, label=density_label)
        if density_background is not None:
            ticks = np.array([-1, -.1, 0, .1, 1])*limit
            bar.set_ticks(ticks, labels=[f'{t:.2g}' for t in ticks])
        if fields is not None:
            line, = axes[1, 0].plot(x, fields[0], color='#2563eb', linewidth=2)
            axes[1, 0].set(xlabel=labels[0], ylabel=field_label, ylim=(-field_limit, field_limit),
                           xlim=(x[0], x[-1]))
            axes[1, 0].grid(alpha=.15)
        heading = fig.suptitle(title, fontweight='bold', fontsize=20)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=20, codec='libx264', bitrate=-1,
                             extra_args=['-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
        indices = np.unique(np.linspace(0, len(frames)-1, min(241, len(frames))).astype(int))
        sequence = np.r_[np.zeros(10, dtype=int), indices, np.full(20, len(frames)-1, dtype=int)]
        print(f'Rendering {len(sequence)} dynamics-only frames ...', flush=True)
        with plt.rc_context({'animation.ffmpeg_path': ffmpeg}):
            with writer.saving(fig, str(output/'trajectory.mp4'), dpi=150):
                for count, i in enumerate(sequence):
                    image.set_data(frames[i] if density_background is None else frames[i]-density_background)
                    if fields is not None:
                        line.set_ydata(fields[i])
                    heading.set_text(f'{title}   |   {time_label} = {clock[i]:.3g}')
                    writer.grab_frame()
                    if count % 40 == 0:
                        print(f'  video {count+1}/{len(sequence)}', flush=True)
        plt.close(fig)
        subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', str(output/'trajectory.mp4'),
                        '-filter_complex', 'fps=10,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4',
                        '-loop', '0', str(output/'trajectory.gif')], check=True)
    print(f'Saved {output}: final MSE={losses[-1]:.6g}, mean MSE={losses.mean():.6g}', flush=True)


def euler_evolve(initial_state, steps, dt, sound_speed=0.7):
    """Conservative periodic Euler on a square 2π domain, Rusanov flux/SSP-RK2."""
    def flux(left, right, direction):
        def physical(q):
            rho, mx, my = q
            momentum = q[direction]
            pressure = sound_speed**2*rho
            return jnp.stack((momentum, mx*momentum/rho + (direction == 1)*pressure,
                              my*momentum/rho + (direction == 2)*pressure))
        speed = jnp.maximum(jnp.abs(left[direction]/left[0]),
                            jnp.abs(right[direction]/right[0])) + sound_speed
        return (physical(left)+physical(right))/2 - speed[None]*(right-left)/2
    def rhs(q):
        fx = flux(q, jnp.roll(q, -1, axis=2), 1)
        fy = flux(q, jnp.roll(q, -1, axis=1), 2)
        return -((fx-jnp.roll(fx, 1, axis=2)) + (fy-jnp.roll(fy, 1, axis=1)))/(2*jnp.pi/q.shape[2])
    def step(q, _):
        q1 = q + dt*rhs(q)
        q2 = (q+q1+dt*rhs(q1))/2
        return q2, q2[0]
    _, densities = jax.lax.scan(jax.checkpoint(step), initial_state, xs=None, length=steps)
    return jnp.concatenate((initial_state[None, 0], densities), axis=0)


def euler_evolve_muscl(initial_state, steps, dt, dx, sound_speed=1.0):
    """Periodic second-order MUSCL-Hancock isothermal Euler, unit-safe via dx/dt.

    Copyright (c) 2024 Philip Mocz.
    Adapted from Philip Mocz's finite-volume Euler example (2024),
    https://github.com/pmocz/finitevolume-jax, distributed under GPL-3.0.
    """
    def primitive(q):
        rho = q[0]
        return jnp.stack((rho, q[1]/rho, q[2]/rho))

    def conservative(w):
        rho, u, v = w
        return jnp.stack((rho, rho*u, rho*v))

    def slope(w, axis):
        return (jnp.roll(w, -1, axis=axis)-jnp.roll(w, 1, axis=axis))/(2*dx)

    def face_flux(left, right, direction):
        rho_l, u_l, v_l = left
        rho_r, u_r, v_r = right
        un_l, un_r = (u_l, u_r) if direction == 2 else (v_l, v_r)
        mom_l, mom_r = (u_l, u_r) if direction == 2 else (v_l, v_r)
        mass_flux = (rho_l*un_l + rho_r*un_r)/2
        momx_flux = (rho_l*u_l*un_l + rho_r*u_r*un_r)/2
        momy_flux = (rho_l*v_l*un_l + rho_r*v_r*un_r)/2
        if direction == 2:
            momx_flux = momx_flux + sound_speed**2*(rho_l+rho_r)/2
        else:
            momy_flux = momy_flux + sound_speed**2*(rho_l+rho_r)/2
        speed = jnp.maximum(jnp.abs(un_l), jnp.abs(un_r)) + sound_speed
        return jnp.stack((mass_flux, momx_flux, momy_flux)) - \
            speed[None]*(conservative(right)-conservative(left))/2

    def step(q, _):
        w = primitive(q)
        wx, wy = slope(w, 2), slope(w, 1)
        rho, u, v = w
        rho_x, rho_y = wx[0], wy[0]
        u_x, u_y = wx[1], wy[1]
        v_x, v_y = wx[2], wy[2]
        # Multidimensional half-step predictor in primitive variables.
        half = w.at[0].add(-0.5*dt*(u*rho_x + rho*u_x + v*rho_y + rho*v_y))
        half = half.at[1].add(-0.5*dt*(u*u_x + v*u_y + sound_speed**2*rho_x/rho))
        half = half.at[2].add(-0.5*dt*(u*v_x + v*v_y + sound_speed**2*rho_y/rho))
        # Face states at i+1/2; rolls align cell i+1 as the right state.
        left_x = half + 0.5*dx*wx
        right_x = jnp.roll(half - 0.5*dx*wx, -1, axis=2)
        left_y = half + 0.5*dx*wy
        right_y = jnp.roll(half - 0.5*dx*wy, -1, axis=1)
        fx = face_flux(left_x, right_x, 2)
        fy = face_flux(left_y, right_y, 1)
        updated = q - dt/dx*((fx-jnp.roll(fx, 1, axis=2)) +
                             (fy-jnp.roll(fy, 1, axis=1)))
        return updated, updated[0]

    _, densities = jax.lax.scan(jax.checkpoint(step), initial_state, xs=None, length=steps)
    return jnp.concatenate((initial_state[None, 0], densities), axis=0)


def make_pic(target, particles, steps, length, velocity_range, seed, kernel_width=1.0,
             timestep_ratio=0.65, debye_ratio=0.7, thermal_speed=0.035,
             max_speed=0.9):
    """Build a differentiable 1D PIC x-vx image-fit problem using JAX-in-Cell.

    The upstream solver evolves one spatial coordinate and three velocity
    components. The fitted image is the 1D1V marginal, with x in columns and
    v_x in rows; ions provide the neutralizing, self-consistent background.
    """
    from jaxincell import Simulation

    target = jnp.asarray(target)
    size = target.shape[0]
    if target.ndim != 2 or target.shape[1] != size:
        raise ValueError('PIC phase-space fitting requires a square image target.')
    light_speed = 299792458.0
    dx, dv = length / size, 2 * velocity_range / (size - 1)

    def image_from_phase_space(positions, velocities):
        x_centers = jnp.linspace(-length / 2, length / 2, size, endpoint=False)
        vx_centers = jnp.linspace(-velocity_range, velocity_range, size)
        x_delta = jnp.mod(positions[:, None] - x_centers[None, :] + length / 2, length) - length / 2
        vx_delta = velocities[:, None] / light_speed - vx_centers[None, :]
        x_kernel = jnp.exp(-0.5 * (x_delta / (kernel_width * dx)) ** 2)
        vx_kernel = jnp.exp(-0.5 * (vx_delta / (kernel_width * dv)) ** 2)
        density = jnp.einsum('ni,nj->ji', x_kernel, vx_kernel)
        return density * (size * size) / (particles * 2 * jnp.pi * kernel_width**2)

    weights = np.maximum(np.asarray(target, dtype=np.float64).ravel(), 0)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    rows, columns = np.divmod(rng.choice(weights.size, size=particles, p=weights), size)
    jitter = rng.uniform(-0.35, 0.35, size=(particles, 2))
    vx = ((rows + jitter[:, 0]) / (size - 1) * 2 - 1) * velocity_range
    x = (columns + jitter[:, 1]) / size * length - length / 2
    # Optimize unconstrained latent velocities, but map them smoothly to a
    # subluminal physical speed for the relativistic Boris pusher.
    vx_latent = np.arctanh(np.clip(vx / max_speed, -0.999, 0.999))
    initial_parameters = jnp.asarray(np.stack((x, vx_latent), axis=1), dtype=jnp.float64)

    def parameter_image(parameters):
        return image_from_phase_space(parameters[:, 0], max_speed * jnp.tanh(parameters[:, 1]) * light_speed)

    zero_state = jnp.zeros((particles, 3), dtype=jnp.float64)
    ion_positions = zero_state.at[:, 0].set(
        jnp.linspace(-length / 2, length / 2, particles, endpoint=False)
    )
    print(
        f'Preparing JAX-in-Cell PIC: {particles} electrons + {particles} ions, '
        f'{size} grid cells, {steps} steps; image axes are x (horizontal) and vx (vertical).',
        flush=True,
    )
    sim = Simulation({
        'domain_parameters': {
            'length': length,
            'total_steps': steps,
            'number_grid_points': size,
            'timestep_over_spatialstep_times_c': timestep_ratio,
        },
        'solver_parameters': {
            'field_solver': 0,
            'time_evolution_algorithm': 0,
            'relativistic': True,
            'filter_passes': 1,
            'print_info': False,
            'seed': seed,
        },
        'species_parameters': {
            'electrons': {'electrons0': {
                'number_pseudoparticles': particles,
                'initial_positions': zero_state,
                'initial_velocities': zero_state,
                'dx_over_Debye_length': debye_ratio,
                'vth_over_c_x': thermal_speed,
                'vth_over_c_y': thermal_speed,
                'vth_over_c_z': thermal_speed,
                'charge_over_elementary_charge': -1.0,
            }},
            'ions': {'ions0': {
                'number_pseudoparticles': particles,
                'initial_positions': ion_positions,
                'initial_velocities': zero_state,
                'dx_over_Debye_length': debye_ratio,
                'charge_over_elementary_charge': 1.0,
                'mass_over_proton_mass': 1.0,
                'vth_over_c_x': '_electrons0',
                'vth_over_c_y': '_electrons0',
                'vth_over_c_z': '_electrons0',
            }},
        },
    })

    def run_simulation(parameters):
        electrons = jnp.zeros((particles, 3), dtype=jnp.float64)
        electrons = electrons.at[:, 0].set(parameters[:, 0])
        electron_velocities = jnp.zeros_like(electrons).at[:, 0].set(
            max_speed * jnp.tanh(parameters[:, 1]) * light_speed
        )
        return sim.run({'electrons': {'electrons0': {
            'initial_positions': electrons,
            'initial_velocities': electron_velocities,
        }}})

    def initial_electric_field(parameters):
        # Use the same upstream initialization path as Simulation.run so t=0
        # electric fields correspond to the optimized particle positions.
        from jaxincell._routing import build_runtime_parameter_sections
        from jaxincell._parameters._species_parameters import resolve_species_references
        from jaxincell._state_initialization import (
            build_domain_state, initialize_field_state, initialize_particle_state,
        )
        electrons = jnp.zeros((particles, 3), dtype=jnp.float64)
        electrons = electrons.at[:, 0].set(parameters[:, 0])
        electron_velocities = jnp.zeros_like(electrons).at[:, 0].set(
            max_speed * jnp.tanh(parameters[:, 1]) * light_speed
        )
        runtime = build_runtime_parameter_sections(
            {
                'domain_parameters': sim._domain_parameters,
                'species_parameters': sim._species_parameters,
                'solver_parameters': sim._solver_parameters,
                'external_field_parameters': sim._external_field_parameters,
                'source_parameters': sim._source_parameters,
            },
            {'species_parameters': {'electrons': {'_electrons0': {
                'initial_positions': electrons,
                'initial_velocities': electron_velocities,
            }}}},
        )
        domain_parameters = runtime['domain_parameters']
        species_parameters = runtime['species_parameters']
        resolve_species_references(species_parameters)
        solver_parameters = runtime['solver_parameters']
        external_parameters = runtime['external_field_parameters']
        domain_state = build_domain_state(domain_parameters)
        particle_state = initialize_particle_state(
            species_parameters, domain_parameters, solver_parameters, domain_state
        )
        field_state = initialize_field_state(
            domain_parameters, solver_parameters, external_parameters,
            domain_state, particle_state,
        )
        return field_state['fields'][0][:, 0]

    return initial_parameters, run_simulation, image_from_phase_space, parameter_image, initial_electric_field


def make_vlasov(target, snapshots, final_time, dt, velocity_range=4.0,
                hermite_modes=12, alpha_x=1.0, domain_length=20.0,
                ion_mass=1836.0, collision_rate=0.005, neutral_initial=False,
                projection='moments', contrast=0.0):
    """Build a self-consistent electron-ion 1D1V SPECTRAX fit."""
    from diffrax import Dopri5
    from spectrax import simulation, inverse_HF_transform
    from orthax.hermite import hermval

    target = jnp.asarray(target)
    size = target.shape[0]
    if target.ndim != 2 or target.shape[1] != size:
        raise ValueError('Vlasov phase-space fitting requires a square image target.')
    species, modes = 2, hermite_modes
    alpha_e = jnp.array([alpha_x, 1.0, 1.0])
    alpha_i = alpha_e / jnp.sqrt(ion_mass)
    alpha = jnp.concatenate((alpha_e, alpha_i))
    alpha_product = jnp.prod(alpha_e)
    ion_product = jnp.prod(alpha_i)
    charges = jnp.array([-1.0, 1.0])
    cyclotron = jnp.array([1.0, 1.0/ion_mass])
    velocities = jnp.linspace(-velocity_range, velocity_range, size)
    xi = velocities / alpha_x
    def background_at_velocity(sample_velocity):
        sample_xi = jnp.asarray(sample_velocity)/alpha_x
        return jnp.exp(-sample_xi**2)/(jnp.sqrt(jnp.pi)*alpha_x)
    encoded_target = (background_at_velocity(velocities)[:, None] + contrast*target
                      if contrast else target)
    zero_velocity = jnp.zeros((size, 1, 1))
    basis = jnp.stack([
        hermval(xi, jnp.eye(modes)[n]) * jnp.exp(-xi**2) * (jnp.pi * alpha[1] * alpha[2])
        / jnp.sqrt(jnp.pi**3 * 2.0**n * jax.scipy.special.factorial(n))
        for n in range(modes)
    ], axis=1)
    if projection == 'svd':
        coefficients = jnp.linalg.pinv(basis, rtol=1e-4) @ encoded_target
    elif projection == 'moments':
        weights = jnp.ones(size).at[0].set(0.5).at[-1].set(0.5)
        dual = jnp.stack([
            hermval(xi, jnp.eye(modes)[n]) / jnp.sqrt(2.0**n * jax.scipy.special.factorial(n))
            for n in range(modes)
        ], axis=1) * (2*velocity_range/(size-1)/alpha_x) * weights[:, None]
        coefficients = dual.T @ encoded_target
    else:
        raise ValueError("projection must be 'moments' or 'svd'")
    density_scale = 1.0 / (alpha_product*jnp.mean(coefficients[0]))
    coefficients = coefficients * density_scale
    rms = jnp.sqrt(jnp.mean(coefficients**2, axis=1))
    scales = jnp.maximum(rms, 1e-3*jnp.max(rms))
    initial = coefficients / scales[:, None]
    def decode(image, sample_velocity=velocities):
        if contrast:
            return (image-background_at_velocity(sample_velocity)[:, None])/contrast
        return image
    print(f'Projecting the nonnegative pixel target with {projection} Hermite projection; '
          'the differentiable loss penalizes negative spectral values.', flush=True)
    def initial_fourier(parameters):
        hermite_coefficients = parameters * scales[:, None]
        spatial_modes = jnp.fft.rfft(hermite_coefficients, axis=-1, norm='forward')
        ck = jnp.zeros((species*modes, 1, size//2+1, 1), dtype=spatial_modes.dtype)
        ck = ck.at[:modes, 0, :, 0].set(spatial_modes)
        ion_modes = jnp.zeros_like(spatial_modes[0])
        if neutral_initial:
            ion_modes = spatial_modes[0] * alpha_product/ion_product
        else:
            ion_modes = ion_modes.at[0].set(
                spatial_modes[0, 0] * alpha_product/ion_product
            )
        ck = ck.at[modes, 0, :, 0].set(ion_modes)
        return ck

    def image_from_parameters(parameters):
        return decode(basis @ (parameters * scales[:, None]) / density_scale)

    def image_at_velocity(parameters, sample_velocity):
        sample_xi = jnp.asarray(sample_velocity) / alpha_x
        sample_basis = jnp.stack([
            hermval(sample_xi, jnp.eye(modes)[n]) * jnp.exp(-sample_xi**2)
            * (jnp.pi*alpha[1]*alpha[2])
            / jnp.sqrt(jnp.pi**3*2.0**n*jax.scipy.special.factorial(n))
            for n in range(modes)
        ], axis=1)
        return decode(sample_basis @ (parameters * scales[:, None]) / density_scale, sample_velocity)

    def run_simulation(parameters, save_count=snapshots):
        electron_coefficients = parameters * scales[:, None]
        electron_modes = jnp.fft.rfft(electron_coefficients, axis=-1, norm='forward')
        wave_numbers = jnp.arange(size//2+1)
        safe_wave_numbers = jnp.maximum(wave_numbers, 1)
        electric = (1j*alpha_product*electron_modes[0]*domain_length
                    /(2*jnp.pi*safe_wave_numbers*cyclotron[0]))
        electric = electric.at[0].set(0.0)
        if neutral_initial:
            electric = jnp.zeros_like(electric)
        initial_fields = jnp.zeros((6, 1, size//2+1, 1), dtype=jnp.complex128)
        initial_fields = initial_fields.at[0, 0, :, 0].set(electric)
        result = simulation(
            {'Ck_0': initial_fourier(parameters), 'Fk_0': initial_fields,
             'Lx': domain_length, 'Ly': 1.0, 'Lz': 1.0, 'qs': charges,
             'alpha_s': alpha, 'u_s': jnp.zeros(3*species),
             'Omega_cs': cyclotron, 'nu': collision_rate, 'D': 0.0,
             'mi_me': ion_mass, 'Ti_Te': 0.0,
             't_max': final_time, 'ode_tolerance': 1e-5},
            Nx=size, Ny=1, Nz=1, Nn=modes, Nm=1, Np=1, Ns=species,
            timesteps=save_count, dt=dt, solver=Dopri5(), adaptive_time_step=False,
        )
        ck = result['Ck']
        coefficients = ck[:, :modes]
        distribution = inverse_HF_transform(
            coefficients, modes, 1, 1, size, 1, 1,
            velocities[:, None, None]/alpha_x, zero_velocity, zero_velocity,
        )[:, 0, :, 0, :, 0, 0].transpose(0, 2, 1)
        distribution = distribution * (jnp.pi*alpha_e[1]*alpha_e[2]) / density_scale
        electric_field = jnp.fft.irfft(result['Fk'][:, 0, 0, :, 0],
                                       n=size, axis=-1, norm='forward')
        return result['time'], distribution, electric_field

    print(
        f'Preparing self-consistent SPECTRAX 1D1V: {size}×{size}, {modes} Hermite modes, '
        f'electrons + {ion_mass:g}×-mass ions, Lx={domain_length:g}, t={final_time:g}, '
        f'ν={collision_rate:g}; image axes x,vx.', flush=True,
    )
    return (initial, run_simulation, initial_fourier, velocities, image_from_parameters,
            image_at_velocity, background_at_velocity)
