from __future__ import annotations

import math
from typing import Any

import georax
import jax
import jax.numpy as jnp
import numpy as np


Geometry = georax.Manifold[Any]


def spd_vech(x: jax.Array) -> jax.Array:
    """Flatten the lower triangle of a symmetric matrix."""
    n = int(x.shape[-1])
    rows, cols = np.tril_indices(n)
    return x[..., rows, cols]


def spd_unvech(x: jax.Array) -> jax.Array:
    """Restore symmetric matrices from lower-triangular vectors."""
    size = int(x.shape[-1])
    root = math.isqrt(8 * size + 1)
    if root * root != 8 * size + 1:
        raise ValueError(f"Invalid vech length {size}.")
    n = (root - 1) // 2
    rows, cols = np.tril_indices(n)
    out = jnp.zeros((*x.shape[:-1], n, n), dtype=x.dtype)
    out = out.at[..., rows, cols].set(x)
    diagonal = jnp.diagonal(out, axis1=-2, axis2=-1)
    diagonal_matrix = jnp.eye(n, dtype=x.dtype) * diagonal[..., None, :]
    return out + jnp.swapaxes(out, -1, -2) - diagonal_matrix


def project_to_manifold(geometry: Geometry, x: jax.Array) -> jax.Array:
    """Project a neural-network output onto a supported georax manifold."""
    if isinstance(geometry, georax.Euclidean):
        return x

    if isinstance(geometry, georax.SO):
        if x.ndim < 2 or x.shape[-2:] != geometry.state_shape:
            if geometry.n == 3 and x.shape[-1] == 6:
                a, b = x[..., :3], x[..., 3:]
                a = a / jnp.maximum(jnp.linalg.norm(a, axis=-1, keepdims=True), 1e-7)
                b = b - jnp.sum(a * b, axis=-1, keepdims=True) * a
                b = b / jnp.maximum(jnp.linalg.norm(b, axis=-1, keepdims=True), 1e-7)
                return jnp.stack((a, b, jnp.cross(a, b, axis=-1)), axis=-1)
            if x.shape[-1] == geometry.n**2:
                x = x.reshape((*x.shape[:-1], *geometry.state_shape))
            else:
                raise ValueError(
                    f"SO({geometry.n}) expects (..., {geometry.n}, {geometry.n})"
                    f" or a flattened matrix; got {x.shape}."
                )
        u, _, vh = jnp.linalg.svd(x, full_matrices=False)
        det = jnp.linalg.det(u @ vh)
        u = u.at[..., :, -1].set(
            u[..., :, -1] * jnp.where(det < 0.0, -1.0, 1.0)[..., None]
        )
        return u @ vh

    if isinstance(geometry, georax.SPD):
        if x.ndim < 2 or x.shape[-2:] != geometry.state_shape:
            x = spd_unvech(x)
        sym = 0.5 * (x + jnp.swapaxes(x, -1, -2))
        eigenvalues, eigenvectors = jnp.linalg.eigh(sym)
        eigenvalues = jnp.maximum(eigenvalues, jnp.asarray(1e-6, dtype=x.dtype))
        return (eigenvectors * eigenvalues[..., None, :]) @ jnp.swapaxes(
            eigenvectors, -1, -2
        )

    raise TypeError(f"Unsupported geometry {type(geometry).__name__}.")
