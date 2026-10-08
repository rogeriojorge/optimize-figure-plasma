"""Image preparation, verbose optimization, and shared plotting."""
from pathlib import Path
from time import perf_counter
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image
from scipy.optimize import minimize
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
    return jnp.asarray(density/density.mean())


def optimize(loss, initial, iterations, learning_rate):
    """JIT one Adam step; report every iteration, including compilation time."""
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(initial)
    @jax.jit
    def step(parameters, state):
        value, gradient = jax.value_and_grad(loss)(parameters)
        updates, state = optimizer.update(gradient, state, parameters)
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


def save_results(target, initial, frames, history, output, parameters=None, times=None):
    """Save a comparable color scale, loss trace, and all raw snapshots."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target, initial, frames = map(np.asarray, (target, initial, frames))
    if not np.isfinite(frames).all():
        raise FloatingPointError('Nonfinite simulation output.')
    losses = np.mean((frames-target)**2, axis=(-2, -1))
    data = dict(target=target, initial=initial, frames=frames, loss=history,
                frame_mse=losses, frame_mass=frames.mean(axis=(-2, -1)))
    if times is not None:
        data['times'] = np.asarray(times)
    if parameters is not None:
        for i, leaf in enumerate(jax.tree_util.tree_leaves(parameters)):
            data[f'parameters_{i}'] = np.asarray(leaf)
    np.savez_compressed(output/'results.npz', **data)
    indices = np.linspace(0, len(frames)-1, 3).astype(int)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    for ax, data, title in zip(axes, [target, initial, *frames[indices[1:]]],
                               ['Target', 'Optimized initial', *[f't={float(times[i]):.3g}' if times is not None else f'Frame {i}' for i in indices[1:]]]):
        ax.imshow(data, cmap='inferno', vmin=target.min(), vmax=target.max())
        ax.set_title(title)
        ax.axis('off')
    fig.savefig(output/'comparison.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.semilogy(np.maximum(history, 1e-16))
    ax.set(xlabel='Optimization iteration', ylabel='Objective')
    fig.savefig(output/'loss.png', dpi=160)
    plt.close(fig)
    scaled = np.clip((frames-target.min())/(target.max()-target.min()+1e-12), 0, 1)
    rgb = (plt.get_cmap('inferno')(scaled)[..., :3]*255).astype(np.uint8)
    images = [Image.fromarray(frame).resize((384, 384), Image.Resampling.NEAREST) for frame in rgb]
    images[0].save(output/'trajectory.gif', save_all=True, append_images=images[1:],
                   duration=max(40, 1600//len(images)), loop=0)
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


def make_pic(target, particles, steps, length, velocity_range, seed, kernel_width=1.0,
             timestep_ratio=0.65, debye_ratio=0.7):
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
    initial_parameters = jnp.asarray(np.stack((x, vx), axis=1), dtype=jnp.float64)

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
                'vth_over_c_x': 0.035,
                'vth_over_c_y': 0.035,
                'vth_over_c_z': 0.035,
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
        electron_velocities = jnp.zeros_like(electrons).at[:, 0].set(parameters[:, 1] * light_speed)
        return sim.run({'electrons': {'electrons0': {
            'initial_positions': electrons,
            'initial_velocities': electron_velocities,
        }}})

    return initial_parameters, run_simulation, image_from_phase_space
