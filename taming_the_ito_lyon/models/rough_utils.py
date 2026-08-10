import jax
import jax.numpy as jnp


def compute_disjoint_signature_times(
    ts: jax.Array, signature_window_size: int
) -> jax.Array:
    """Return the regularly strided knot grid required by roughrax."""
    step = int(signature_window_size)
    num_intervals = int(ts.shape[0]) - 1
    if step <= 0:
        raise ValueError("signature_window_size must be a positive integer.")
    if num_intervals < 1 or num_intervals % step:
        raise ValueError(
            "The control grid must contain a whole number of signature windows."
        )
    return ts[::step]


def ito_correction(
    ts: jax.Array,
    control_values: jax.Array,
    signature_knots: jax.Array,
    depth: int,
    *,
    has_time_channel: bool,
) -> jax.Array:
    """Build per-segment Brownian corrections for roughrax/PySigLib."""
    num_windows = int(signature_knots.shape[0]) - 1
    stride = (int(control_values.shape[0]) - 1) // num_windows
    dim = int(control_values.shape[-1])
    correction_dim = sum(dim**level for level in range(2, int(depth) + 1))
    correction = jnp.zeros(
        (int(control_values.shape[0]) - 1, correction_dim),
        dtype=control_values.dtype,
    )
    if depth >= 2:
        if has_time_channel:
            dt = jnp.diff(control_values[:, 0])
            channels = jnp.arange(1, dim)
        else:
            increments = jnp.diff(control_values, axis=0)
            active = jnp.any(increments != 0.0, axis=-1)
            dt = jnp.diff(ts) * active
            channels = jnp.arange(dim)
        diagonal = channels * dim + channels
        correction = correction.at[:, diagonal].set(dt[:, None])
    return correction.reshape(num_windows, stride, correction_dim)
