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
    if isinstance(size, (tuple, list)):
        size_v, size_x = map(int, size)
        canvas_size = max(size_v, size_x)
    else:
        size_v = size_x = int(size)
        canvas_size = size_x
    image = Image.open(image).convert('RGBA')
    image = Image.alpha_composite(Image.new('RGBA', image.size, 'white'), image).convert('L')
    image.thumbnail((canvas_size, canvas_size), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (canvas_size, canvas_size), color=255)
    canvas.paste(image, ((canvas_size-image.width)//2, (canvas_size-image.height)//2))
    if (size_v, size_x) != (canvas_size, canvas_size):
        canvas = canvas.resize((size_x, size_v), Image.Resampling.LANCZOS)
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
        history, durations, cache = [], [], []
        print(f'Compiling JAX objective/gradient for SciPy {method} ...', flush=True)
        def objective(p):
            if cache and np.array_equal(p, cache[0]):
                return cache[1], cache[2]
            start = perf_counter()
            value, gradient = evaluate(jnp.asarray(p))
            value, gradient = float(value), np.asarray(gradient)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise FloatingPointError('Nonfinite objective or gradient: reduce time step or parameter step.')
            durations.append(perf_counter()-start)
            cache[:] = [np.array(p, copy=True), value, gradient]
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
                 time_label='t', field_label='Ex', density_background=None, image_contrast=1.0,
                 stage_lengths=None, stage_times=None, plot_threshold=None,
                 extended_frames=None, extended_times=None, extended_electric_fields=None,
                 extended_baseline=None, metadata=None):
    """Save full physical arrays, separate diagnostics, and paired six-second movies."""
    from matplotlib.animation import FFMpegWriter
    from matplotlib.colors import SymLogNorm
    import imageio_ffmpeg
    import subprocess
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target, initial, frames = map(np.asarray, (target, initial, frames))
    baseline = frames if baseline is None else np.asarray(baseline)
    times = np.arange(len(frames)) if times is None else np.asarray(times)
    fields = None if electric_fields is None else np.asarray(electric_fields)
    extended = None if extended_frames is None else np.asarray(extended_frames)
    if not all(np.isfinite(a).all() for a in (frames, baseline)):
        raise FloatingPointError('Nonfinite simulation output.')
    if fields is not None and (len(fields) != len(frames) or not np.isfinite(fields).all()):
        raise ValueError('Electric fields must be finite and aligned with density frames.')
    if extended is not None:
        extended_times = np.asarray(extended_times)
        if extended.shape[1:] != frames.shape[1:] or len(extended_times) != len(extended):
            raise ValueError('Extended times and states must match the standard trajectory grid.')
        if not np.isfinite(extended).all() or not np.allclose(extended[0], frames[0], rtol=1e-9, atol=1e-11):
            raise ValueError('Extended trajectory must be finite and start from the same optimized state.')
        if extended_times[-1] <= times[-1]:
            raise ValueError('Extended trajectory must continue beyond the optimization horizon.')
    plot_threshold = (.05 if fields is None else 0.0) if plot_threshold is None else float(plot_threshold)
    extent = tuple(extent) if extent is not None else (0, target.shape[-1], 0, target.shape[-2])
    clock = times * float(time_scale)
    losses = np.mean((frames-target)**2, axis=(-2, -1))
    data = dict(target=target, initial=initial, frames=frames, baseline=baseline,
                loss=history, frame_mse=losses,
                baseline_mse=np.mean((baseline-target)**2, axis=(-2, -1)),
                frame_mass=frames.mean(axis=(-2, -1)), times=times,
                time_scale=float(time_scale), image_contrast=float(image_contrast),
                plot_threshold=plot_threshold, extent=np.asarray(extent),
                labels=np.asarray(labels), title=np.asarray(title),
                time_label=np.asarray(time_label), field_label=np.asarray(field_label),
                signal_mse=losses/float(image_contrast)**2,
                baseline_signal_mse=np.mean((baseline-target)**2, axis=(-2, -1))/float(image_contrast)**2)
    if stage_lengths is not None:
        if sum(stage_lengths) != len(history) or len(stage_lengths) != len(stage_times):
            raise ValueError('Stage histories must align with horizon times and objective history.')
        data.update(stage_lengths=np.asarray(stage_lengths), stage_times=np.asarray(stage_times))
    if fields is not None:
        data['electric_fields'] = fields
        if baseline_electric_fields is not None:
            data['baseline_electric_fields'] = np.asarray(baseline_electric_fields)
    if density_background is not None:
        density_background = np.asarray(density_background)
        data['density_background'] = density_background
    if extended is not None:
        data.update(extended_frames=extended, extended_times=extended_times,
                    extended_frame_mse=np.mean((extended-target)**2, axis=(-2, -1)),
                    extended_frame_mass=extended.mean(axis=(-2, -1)),
                    extension_factor=float(extended_times[-1]/times[-1]))
        if extended_electric_fields is not None:
            extended_electric_fields = np.asarray(extended_electric_fields)
            if len(extended_electric_fields) != len(extended) or not np.isfinite(extended_electric_fields).all():
                raise ValueError('Extended electric fields must be finite and aligned with states.')
            data['extended_electric_fields'] = extended_electric_fields
        if extended_baseline is not None:
            data['extended_baseline'] = np.asarray(extended_baseline)
    if parameters is not None:
        for i, leaf in enumerate(jax.tree_util.tree_leaves(parameters)):
            data[f'parameters_{i}'] = np.asarray(leaf)
    if metadata:
        if set(metadata) & set(data):
            raise ValueError('Metadata cannot override saved physical arrays or display settings.')
        data.update(metadata)
    np.savez_compressed(output/'results.npz', **data)
    style = {'font.family': 'DejaVu Sans', 'font.size': 12,
             'axes.spines.top': False, 'axes.spines.right': False}
    all_runs = [frames] if extended is None else [frames, extended]
    x = np.linspace(extent[0], extent[1], frames.shape[-1], endpoint=False)
    density_label = 'Density' if fields is None else 'f(x, vx)'
    def display_values(image):
        return image if density_background is None else image-density_background
    reference = display_values(target)
    if fields is not None:
        # Keep the vivid jet palette with fixed limits and continuous colors.
        # No moving opacity threshold is applied to plasma phase space.
        cmap = plt.get_cmap('jet').copy()
        limit = max(np.max(np.abs(reference)),
                    *(np.quantile(np.abs(display_values(run)), .995) for run in all_runs), 1e-12)
        color_options = dict(cmap=cmap, vmin=0, vmax=limit)
        if density_background is not None:
            color_options = dict(cmap=cmap, norm=SymLogNorm(
                linthresh=.03*limit, linscale=.4, vmin=-limit, vmax=limit))
            density_label = 'f − background · fixed symlog scale'
        else:
            density_label += ' · fixed linear scale'
    else:
        cmap = plt.get_cmap('jet').copy()
        cmap.set_bad('white')
        color_options = dict(cmap=cmap, vmin=min(target.min(), *(np.quantile(run, .001) for run in all_runs)),
                             vmax=max(target.max(), *(np.quantile(run, .999) for run in all_runs)))
        if density_background is not None:
            limit = max(np.max(np.abs(reference)), *(np.max(np.abs(display_values(run))) for run in all_runs), 1e-12)
            color_options = dict(cmap=cmap, norm=SymLogNorm(linthresh=.03*limit, linscale=.4, vmin=-limit, vmax=limit))
            density_label = f'Density − {float(density_background):g} (symlog scale)'
    cutoff = max(0.0, plot_threshold)*np.max(np.abs(reference))
    if cutoff:
        density_label += f' · display cutoff {cutoff:.2g}'
    def display_alpha(image):
        values = display_values(image)
        return 1.0 if cutoff == 0 else np.clip((np.abs(values)-cutoff)/cutoff, 0, 1)
    with plt.rc_context(style):
        def phase(ax, image, caption):
            artist = ax.imshow(display_values(image), alpha=display_alpha(image), origin='lower',
                               extent=extent, aspect='auto', interpolation='bilinear',
                               interpolation_stage='data', **color_options)
            ax.set(title=caption, xlabel=labels[0], ylabel=labels[1])
            return artist
        def colorbar(fig, artist, axes):
            bar = fig.colorbar(artist, ax=axes, shrink=.85, label=density_label)
            if fields is None and density_background is not None:
                ticks = np.array([-1, -.1, 0, .1, 1])*limit
                bar.set_ticks(ticks, labels=[f'{t:.2g}' for t in ticks])
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), layout='constrained')
        for ax, image, caption in zip(axes, [target, initial, baseline[-1], frames[-1]],
                ['Target', 'Optimized initial', 'Baseline final', 'Optimized final']):
            phase(ax, image, caption)
        fig.suptitle(title, fontweight='bold')
        fig.savefig(output/'comparison.png', dpi=200)
        plt.close(fig)
        for run, run_times, name in [(frames, clock, 'initial_final'),
                                     *(([(extended, extended_times*float(time_scale), 'extended_initial_final')]) if extended is not None else [])]:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
            for ax, i in zip(axes, [0, -1]):
                artist = phase(ax, run[i], f'{"Initial" if i == 0 else "Final"} · {time_label} = {run_times[i]:.3g}')
            colorbar(fig, artist, axes)
            fig.suptitle(title, fontweight='bold')
            fig.savefig(output/f'{name}.png', dpi=200)
            plt.close(fig)
        if fields is not None:
            plot_fields = fields if extended_electric_fields is None else extended_electric_fields
            plot_clock = clock if extended_electric_fields is None else extended_times*float(time_scale)
            field_x = np.linspace(extent[0], extent[1], plot_fields.shape[-1], endpoint=False)
            field_limit = max(np.quantile(np.abs(plot_fields), .995), 1e-12)*1.08
            fig, axes = plt.subplots(2, 1, figsize=(9, 6), layout='constrained')
            for i, caption, color in [(0, 'Initial', '#64748b'), (-1, 'At optimization time', '#2563eb')]:
                axes[0].plot(field_x, fields[i], label=caption, color=color, linewidth=2)
            if extended_electric_fields is not None:
                axes[0].plot(field_x, plot_fields[-1], label='At 1.5T', color='#dc2626', linewidth=1.5)
            axes[0].set(xlabel=labels[0], ylabel=field_label)
            axes[0].legend(frameon=False)
            axes[0].grid(alpha=.15)
            field_image = axes[1].imshow(plot_fields, origin='lower', aspect='auto', cmap='RdBu_r',
                extent=(extent[0], extent[1], plot_clock[0], plot_clock[-1]),
                vmin=-field_limit, vmax=field_limit, interpolation='bilinear')
            if extended is not None:
                axes[1].axhline(clock[-1], color='black', linestyle='--', linewidth=1)
            axes[1].set(xlabel=labels[0], ylabel=time_label, title='Electric-field evolution')
            fig.colorbar(field_image, ax=axes[1], label=field_label)
            fig.suptitle(title, fontweight='bold')
            fig.savefig(output/'electric_field.png', dpi=200)
            plt.close(fig)
            sample_velocity = np.linspace(extent[2], extent[3], frames.shape[-2])
            weights = np.ones(len(sample_velocity))
            weights[[0, -1]] = .5
            weights *= sample_velocity[1]-sample_velocity[0]
            density = np.einsum('tvx,v->tx', frames, weights)
            mean_velocity = np.einsum('tvx,v->tx', frames, weights*sample_velocity)/np.maximum(density, 1e-30)
            for name, values, ylabel in [('density', density, 'Velocity-window integral of f'),
                                        ('velocity', mean_velocity, f'Mean {labels[1]} in velocity window')]:
                fig, ax = plt.subplots(figsize=(8, 3.5), layout='constrained')
                for i, caption, color in [(0, 'Initial', '#64748b'), (-1, 'At optimization time', '#2563eb')]:
                    ax.plot(x, values[i], label=caption, color=color, linewidth=2)
                ax.set(xlabel=labels[0], ylabel=ylabel, title=title)
                ax.legend(frameon=False)
                ax.grid(alpha=.15)
                fig.savefig(output/f'{name}.png', dpi=200)
                plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3.5), layout='constrained')
        if stage_lengths is None:
            ax.semilogy(np.maximum(history, 1e-16), color='#2563eb', linewidth=2)
        else:
            offset = 0
            for count, horizon in zip(stage_lengths, stage_times):
                ax.semilogy(np.arange(offset, offset+count), np.maximum(history[offset:offset+count], 1e-16),
                            linewidth=2, label=f'{time_label} max = {horizon:.3g}')
                offset += count
            ax.legend(frameon=False, fontsize=8)
        ax.set(xlabel='Optimization iteration', ylabel='Objective', title=title)
        ax.grid(alpha=.15)
        fig.savefig(output/'loss.png', dpi=200)
        plt.close(fig)
        if fields is None and parameters is not None and np.shape(parameters) == (3, *target.shape):
            rho, mx, my = np.asarray(parameters)
            u, v = mx/rho, my/rho
            fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
            speed_values = np.hypot(u, v)
            speed = ax.imshow(np.ma.masked_less(speed_values, plot_threshold*speed_values.max()),
                              origin='lower', extent=extent, cmap=cmap, aspect='auto')
            y = np.linspace(extent[2], extent[3], target.shape[0], endpoint=False)
            stride = max(1, target.shape[0]//16)
            ax.quiver(x[::stride], y[::stride], u[::stride, ::stride], v[::stride, ::stride], color='white', alpha=.7)
            ax.set(title='Optimized initial velocity', xlabel=labels[0], ylabel=labels[1])
            fig.colorbar(speed, ax=ax, label='Speed')
            fig.savefig(output/'initial_velocity.png', dpi=200)
            plt.close(fig)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        movie_runs = [(frames, clock, 'trajectory')]
        if extended is not None:
            movie_runs.append((extended, extended_times*float(time_scale), 'trajectory_extended'))
        for run, run_clock, name in movie_runs:
            fig, ax = plt.subplots(figsize=(12.8, 7.2), layout='constrained')
            image = phase(ax, run[0], '')
            colorbar(fig, image, ax)
            heading = fig.suptitle(title, fontweight='bold', fontsize=20)
            indices = np.unique(np.linspace(0, len(run)-1, min(241, len(run))).astype(int))
            sequence = np.r_[np.zeros(10, dtype=int), indices, np.full(20, len(run)-1, dtype=int)]
            writer = FFMpegWriter(fps=len(sequence)/6.0, codec='libx264', bitrate=-1,
                extra_args=['-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', '2'])
            print(f'Rendering {name}: {len(sequence)} dynamics-only frames, six seconds ...', flush=True)
            fig.set_dpi(150)
            image.set_animated(True)
            heading.set_animated(True)
            with plt.rc_context({'animation.ffmpeg_path': ffmpeg}):
                with writer.saving(fig, str(output/f'{name}.mp4'), dpi=150):
                    fig.canvas.draw()
                    background = fig.canvas.copy_from_bbox(fig.bbox)
                    for count, i in enumerate(sequence):
                        image.set_data(display_values(run[i]))
                        image.set_alpha(display_alpha(run[i]))
                        suffix = ' · after optimization time' if run_clock[i] > clock[-1]+1e-10 else ''
                        heading.set_text(f'{title}   |   {time_label} = {run_clock[i]:.3g}{suffix}')
                        fig.canvas.restore_region(background)
                        ax.draw_artist(image)
                        fig.draw_artist(heading)
                        writer._proc.stdin.write(fig.canvas.buffer_rgba())
                        if count % 40 == 0:
                            print(f'  video {count+1}/{len(sequence)}', flush=True)
            plt.close(fig)
            subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', str(output/f'{name}.mp4'),
                '-filter_complex_threads', '2', '-filter_complex',
                'fps=20,scale=600:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4',
                '-threads', '2', '-loop', '0', str(output/f'{name}.gif')], check=True)
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
             max_speed=0.9, field_grid_points=None, initialization='image',
             two_stream_drift=0.20, two_stream_thermal=0.05,
             two_stream_perturbation=5e-5):
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
    field_grid_points = size if field_grid_points is None else int(field_grid_points)
    if field_grid_points < 4:
        raise ValueError('PIC field grid requires at least four spatial cells.')
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

    rng = np.random.default_rng(seed)
    if initialization == 'image':
        weights = np.maximum(np.asarray(target, dtype=np.float64).ravel(), 0)
        weights /= weights.sum()
        rows, columns = np.divmod(rng.choice(weights.size, size=particles, p=weights), size)
        jitter = rng.uniform(-0.35, 0.35, size=(particles, 2))
        vx = ((rows + jitter[:, 0]) / (size - 1) * 2 - 1) * velocity_range
        x = (columns + jitter[:, 1]) / size * length - length / 2
    elif initialization == 'two_stream':
        if particles % 2:
            raise ValueError('Two-stream initialization requires an even particle count.')
        if not 0 < two_stream_drift < max_speed or two_stream_thermal < 0 or two_stream_perturbation < 0:
            raise ValueError('Two-stream drift must be subluminal; thermal spread and perturbation must be nonnegative.')
        if particles % 4:
            raise ValueError('Quiet two-stream initialization requires a particle count divisible by four.')
        # Quiet-start particles suppress shot noise while the small seeded mode
        # grows. Alternate equal counter-streams and paired Gaussian quantiles.
        x_grid = (np.arange(particles) + 0.5) / particles * length - length / 2
        x = x_grid + (two_stream_perturbation * length
                      * np.sin(2 * np.pi * x_grid / length))
        from scipy.special import ndtri
        quantiles = ndtri((np.arange(particles // 2) + 0.5) / (particles // 2))
        thermal = np.repeat(quantiles, 2) * (two_stream_thermal / np.sqrt(2))
        signs = np.where(np.arange(particles) % 2, -1.0, 1.0)
        vx = np.clip((two_stream_drift + thermal) * signs, -0.98 * max_speed, 0.98 * max_speed)
    else:
        raise ValueError("initialization must be 'image' or 'two_stream'.")
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
        f'{field_grid_points} field cells, {size}x{size} phase-space bins, {steps} steps; '
        'image axes are x (horizontal) and vx (vertical).',
        flush=True,
    )
    sim = Simulation({
        'domain_parameters': {
            'length': length,
            'total_steps': steps,
            'number_grid_points': field_grid_points,
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

    def initialize_runtime(electrons, electron_velocities):
        from jaxincell._parameters._species_parameters import resolve_species_references
        from jaxincell._routing import build_runtime_parameter_sections
        from jaxincell._state_initialization import (
            build_domain_state, initialize_field_state, initialize_particle_state,
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
        external_parameters = {
            **external_parameters,
            'external_electric_field': field_state['external_electric_field'],
            'external_magnetic_field': field_state['external_magnetic_field'],
        }
        return (domain_parameters, solver_parameters, external_parameters,
                domain_state, particle_state, field_state)

    def run_simulation(parameters, endpoint_only=False, sample_indices=None):
        electrons = jnp.zeros((particles, 3), dtype=jnp.float64)
        electrons = electrons.at[:, 0].set(parameters[:, 0])
        electron_velocities = jnp.zeros_like(electrons).at[:, 0].set(
            max_speed * jnp.tanh(parameters[:, 1]) * light_speed
        )
        if endpoint_only or sample_indices is not None:
            # The upstream public run method stores every solver state. For
            # image objectives, use its same initialization and Boris step in
            # a rematerialized scan that retains only requested states.
            from jax import lax
            from jaxincell._algorithms import Boris_step
            from jaxincell._boundary_conditions import set_BC_particles, set_BC_positions
            (domain_parameters, solver_parameters, external_parameters,
             domain_state, particle_state, field_state) = initialize_runtime(electrons, electron_velocities)
            dx, dt = domain_state['dx'], domain_state['dt']
            grid, box_size = domain_state['grid'], domain_state['box_size']
            positions, velocities = particle_state['positions'], particle_state['velocities']
            charges, masses = particle_state['charges'], particle_state['masses']
            charge_to_mass = particle_state['charge_to_mass_ratios']
            pbc_left, pbc_right = domain_parameters['particle_BC_left'], domain_parameters['particle_BC_right']
            fbc_left, fbc_right = domain_parameters['field_BC_left'], domain_parameters['field_BC_right']
            positions_plus, velocities, charges, masses, charge_to_mass = set_BC_particles(
                positions + (dt / 2) * velocities, velocities, charges, masses, charge_to_mass,
                dx, grid, *box_size, pbc_left, pbc_right,
            )
            positions_minus = set_BC_positions(
                positions - (dt / 2) * velocities, charges, dx, grid, *box_size,
                pbc_left, pbc_right,
            )
            carry = (
                field_state['fields'][0], field_state['fields'][1], positions_minus,
                positions, positions_plus, velocities, charges, masses, charge_to_mass,
            )

            requested = np.asarray([steps - 1] if endpoint_only else sample_indices, dtype=int).ravel()
            if requested.size == 0 or np.any(requested < 0) or np.any(requested >= steps):
                raise ValueError('Sample indices must select at least one solver step.')
            selected = jnp.zeros((len(requested), 2, particles, 3), dtype=jnp.float64)

            def sampled_step(state_samples_and_penalty, index):
                state, samples, penalty_sum = state_samples_and_penalty
                next_state, _ = Boris_step(
                    state, index, solver_parameters, external_parameters, dx, dt, grid, box_size,
                    pbc_left, pbc_right, fbc_left, fbc_right, solver_parameters['field_solver'],
                )
                match = requested == index
                sample_slot = jnp.argmax(match)
                samples = lax.cond(
                    jnp.any(match),
                    lambda values: values.at[sample_slot].set(
                        jnp.stack((next_state[3][:particles], next_state[5][:particles]))
                    ),
                    lambda values: values,
                    samples,
                )
                vx_ratio = next_state[5][:particles, 0] / light_speed
                outside = jnp.maximum(jnp.abs(vx_ratio) - velocity_range, 0.0) / velocity_range
                penalty_sum = penalty_sum + jnp.sum(outside**2)
                return (next_state, samples, penalty_sum), None

            (_, selected, viewport_penalty_sum), _ = lax.scan(
                jax.checkpoint(sampled_step), (carry, selected, jnp.asarray(0.0)), jnp.arange(steps)
            )
            return {
                'positions': selected[:, 0],
                'velocities': selected[:, 1],
                'viewport_penalty_sum': viewport_penalty_sum,
            }
        return sim.run({'electrons': {'electrons0': {
            'initial_positions': electrons,
            'initial_velocities': electron_velocities,
        }}})

    def initial_electric_field(parameters):
        # Use the same upstream initialization path as Simulation.run so t=0
        # electric fields correspond to the optimized particle positions.
        electrons = jnp.zeros((particles, 3), dtype=jnp.float64)
        electrons = electrons.at[:, 0].set(parameters[:, 0])
        electron_velocities = jnp.zeros_like(electrons).at[:, 0].set(
            max_speed * jnp.tanh(parameters[:, 1]) * light_speed
        )
        _, _, _, _, _, field_state = initialize_runtime(electrons, electron_velocities)
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
    if target.ndim != 2:
        raise ValueError('Vlasov phase-space fitting requires a 2D image target.')
    size_v, size_x = target.shape
    species, modes = 2, hermite_modes
    alpha_e = jnp.array([alpha_x, 1.0, 1.0])
    alpha_i = alpha_e / jnp.sqrt(ion_mass)
    alpha = jnp.concatenate((alpha_e, alpha_i))
    alpha_product = jnp.prod(alpha_e)
    ion_product = jnp.prod(alpha_i)
    charges = jnp.array([-1.0, 1.0])
    cyclotron = jnp.array([1.0, 1.0/ion_mass])
    velocities = jnp.linspace(-velocity_range, velocity_range, size_v)
    xi = velocities / alpha_x
    def background_at_velocity(sample_velocity):
        sample_xi = jnp.asarray(sample_velocity)/alpha_x
        return jnp.exp(-sample_xi**2)/(jnp.sqrt(jnp.pi)*alpha_x)
    # Keep the Maxwellian analytically in C0. Projecting its cropped velocity
    # tail into every Hermite mode creates spurious moments and negative tails.
    encoded_target = contrast*target if contrast else target
    zero_velocity = jnp.zeros((size_v, 1, 1))
    basis = jnp.stack([
        hermval(xi, jnp.eye(modes)[n]) * jnp.exp(-xi**2) * (jnp.pi * alpha[1] * alpha[2])
        / jnp.sqrt(jnp.pi**3 * 2.0**n * jax.scipy.special.factorial(n))
        for n in range(modes)
    ], axis=1)
    if projection == 'svd':
        coefficients = jnp.linalg.pinv(basis, rtol=1e-4) @ encoded_target
    elif projection == 'moments':
        weights = jnp.ones(size_v).at[0].set(0.5).at[-1].set(0.5)
        dual = jnp.stack([
            hermval(xi, jnp.eye(modes)[n]) / jnp.sqrt(2.0**n * jax.scipy.special.factorial(n))
            for n in range(modes)
        ], axis=1) * (2*velocity_range/(size_v-1)/alpha_x) * weights[:, None]
        coefficients = dual.T @ encoded_target
    else:
        raise ValueError("projection must be 'moments' or 'svd'")
    if contrast:
        coefficients = coefficients.at[0].add(1.0/alpha_product)
    density_scale = 1.0 / (alpha_product*jnp.mean(coefficients[0]))
    coefficients = coefficients * density_scale
    coefficient_offset = jnp.zeros((modes, 1), dtype=coefficients.dtype)
    if contrast:
        coefficient_offset = coefficient_offset.at[0, 0].set(density_scale/alpha_product)
    perturbation_coefficients = coefficients-coefficient_offset
    rms = jnp.sqrt(jnp.mean(perturbation_coefficients**2, axis=1))
    scales = jnp.maximum(rms, 1e-3*jnp.max(rms))
    initial = perturbation_coefficients / scales[:, None]
    def decode(image, sample_velocity=velocities):
        if contrast:
            return (image-background_at_velocity(sample_velocity)[:, None])/contrast
        return image
    print(f'Projecting the image perturbation with {projection} Hermite moments and an analytic '
          'Maxwellian background; the differentiable loss penalizes negative spectral values.', flush=True)
    def initial_fourier(parameters):
        hermite_coefficients = coefficient_offset + parameters * scales[:, None]
        spatial_modes = jnp.fft.rfft(hermite_coefficients, axis=-1, norm='forward')
        ck = jnp.zeros((species*modes, 1, size_x//2+1, 1), dtype=spatial_modes.dtype)
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
    initial_fourier.coefficient_offset = coefficient_offset
    initial_fourier.coefficient_scales = scales
    initial_fourier.density_scale = density_scale

    def image_from_parameters(parameters):
        return decode(basis @ (coefficient_offset + parameters * scales[:, None]) / density_scale)

    def image_at_velocity(parameters, sample_velocity):
        sample_xi = jnp.asarray(sample_velocity) / alpha_x
        sample_basis = jnp.stack([
            hermval(sample_xi, jnp.eye(modes)[n]) * jnp.exp(-sample_xi**2)
            * (jnp.pi*alpha[1]*alpha[2])
            / jnp.sqrt(jnp.pi**3*2.0**n*jax.scipy.special.factorial(n))
            for n in range(modes)
        ], axis=1)
        return decode(sample_basis @ (coefficient_offset + parameters * scales[:, None]) / density_scale, sample_velocity)

    def run_simulation(parameters, save_count=snapshots, velocity_samples=None, time_step=None):
        electron_coefficients = coefficient_offset + parameters * scales[:, None]
        electron_modes = jnp.fft.rfft(electron_coefficients, axis=-1, norm='forward')
        wave_numbers = jnp.arange(size_x//2+1)
        safe_wave_numbers = jnp.maximum(wave_numbers, 1)
        electric = (1j*alpha_product*electron_modes[0]*domain_length
                    /(2*jnp.pi*safe_wave_numbers*cyclotron[0]))
        electric = electric.at[0].set(0.0)
        if neutral_initial:
            electric = jnp.zeros_like(electric)
        initial_fields = jnp.zeros((6, 1, size_x//2+1, 1), dtype=jnp.complex128)
        initial_fields = initial_fields.at[0, 0, :, 0].set(electric)
        result = simulation(
            {'Ck_0': initial_fourier(parameters), 'Fk_0': initial_fields,
             'Lx': domain_length, 'Ly': 1.0, 'Lz': 1.0, 'qs': charges,
             'alpha_s': alpha, 'u_s': jnp.zeros(3*species),
             'Omega_cs': cyclotron, 'nu': collision_rate, 'D': 0.0,
             'mi_me': ion_mass, 'Ti_Te': 0.0,
             't_max': final_time, 'ode_tolerance': 1e-5},
            Nx=size_x, Ny=1, Nz=1, Nn=modes, Nm=1, Np=1, Ns=species,
            timesteps=save_count, dt=dt if time_step is None else time_step,
            solver=Dopri5(), adaptive_time_step=False,
        )
        ck = result['Ck']
        coefficients = ck[:, :modes]
        distribution = inverse_HF_transform(
            coefficients, modes, 1, 1, size_x, 1, 1,
            velocities[:, None, None]/alpha_x, zero_velocity, zero_velocity,
        )[:, 0, :, 0, :, 0, 0].transpose(0, 2, 1)
        distribution = distribution * (jnp.pi*alpha_e[1]*alpha_e[2]) / density_scale
        electric_field = jnp.fft.irfft(result['Fk'][:, 0, 0, :, 0],
                                       n=size_x, axis=-1, norm='forward')
        if velocity_samples is not None:
            sample_velocities = jnp.asarray(velocity_samples)
            wide_distribution = inverse_HF_transform(
                coefficients, modes, 1, 1, size_x, 1, 1,
                sample_velocities[:, None, None]/alpha_x,
                jnp.zeros((len(sample_velocities), 1, 1)),
                jnp.zeros((len(sample_velocities), 1, 1)),
            )[:, 0, :, 0, :, 0, 0].transpose(0, 2, 1)
            wide_distribution *= (jnp.pi*alpha_e[1]*alpha_e[2]) / density_scale
            return result['time'], distribution, electric_field, wide_distribution
        return result['time'], distribution, electric_field

    print(
        f'Preparing self-consistent SPECTRAX 1D1V: Nx={size_x}, Nv={size_v}, '
        f'{modes} Hermite modes, '
        f'electrons + {ion_mass:g}×-mass ions, Lx={domain_length:g}, t={final_time:g}, '
        f'ν={collision_rate:g}; image axes x,vx.', flush=True,
    )
    return (initial, run_simulation, initial_fourier, velocities, image_from_parameters,
            image_at_velocity, background_at_velocity)
