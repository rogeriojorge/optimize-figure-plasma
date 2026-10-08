"""Image preparation, verbose optimization, and shared plotting."""
from pathlib import Path
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_target(image, size):
    """Letterbox inverted grayscale pixels, floor at 0.05, normalize mean density."""
    image = Image.open(image).convert('L')
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (size, size), color=255)
    canvas.paste(image, ((size-image.width)//2, (size-image.height)//2))
    density = 0.05 + (255-np.asarray(canvas, dtype=float))/255
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
    data = dict(target=target, initial=initial, frames=frames, loss=history, frame_mse=losses)
    if times is not None:
        data['times'] = np.asarray(times)
    if parameters is not None:
        for i, leaf in enumerate(jax.tree_util.tree_leaves(parameters)):
            data[f'parameters_{i}'] = np.asarray(leaf)
    np.savez_compressed(output/'results.npz', **data)
    indices = np.linspace(0, len(frames)-1, 3).astype(int)
    fig, axes = plt.subplots(1, 5, figsize=(14, 3), constrained_layout=True)
    for ax, data, title in zip(axes, [target, initial, *frames[indices]],
                               ['Target', 'Optimized initial', *[f't={float(times[i]):.3g}' if times is not None else f'Frame {i}' for i in indices]]):
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
