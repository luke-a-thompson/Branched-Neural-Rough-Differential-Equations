from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from stochastax.manifolds.spd import SPDManifold

from taming_the_ito_lyon.config.config import Config


def _maybe_wrap_extrapolation(
    loss: Callable[[jax.Array, jax.Array], jax.Array],
    config: Config,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    if config.experiment_config.extrapolation_scheme is None:
        return loss
    n_recon = config.experiment_config.n_recon

    def extrapolation_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
        pred = pred[n_recon:]
        target = target[n_recon:]
        return loss(pred, target) + loss(target, pred)

    return extrapolation_loss


def mse_loss(
    pred: jax.Array,
    target: jax.Array,
) -> jax.Array:
    assert pred.shape == target.shape, (
        f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
    )
    return jnp.mean((pred - target) ** 2)


def frobenius_loss(
    config: Config,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Frobenius loss between predicted and target rotation matrices."""

    def loss(pred: jax.Array, target: jax.Array) -> jax.Array:
        return jnp.mean(jnp.linalg.norm(pred - target, ord="fro", axis=(-2, -1)))

    return _maybe_wrap_extrapolation(loss, config)


def rotational_geodesic_loss(
    config: Config,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Rotational Geodesic Error: RGE(R1, R2) = 2 * arcsin(||R2 - R1||_F / (2√2))."""

    def loss(pred: jax.Array, target: jax.Array) -> jax.Array:
        assert pred.shape == target.shape, (
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
        assert pred.shape[-1] == pred.shape[-2], "pred/target must be square matrices"
        # Closed-form RGE assumes valid rotations; simulator drift + float error can push
        # the arcsin argument marginally outside [-1, 1]. Clip below 1 to also avoid the
        # arcsin derivative singularity at 1.0 (inf gradients, unstable early training).
        ratio = jnp.linalg.norm(pred - target, ord="fro", axis=(-2, -1)) / (
            2.0 * jnp.sqrt(2.0)
        )
        eps = jnp.asarray(1e-5, dtype=ratio.dtype)
        rge_rad = 2.0 * jnp.arcsin(jnp.clip(ratio, min=0.0, max=1.0 - eps))
        return jnp.mean(rge_rad * (180.0 / jnp.pi))

    return _maybe_wrap_extrapolation(loss, config)


def signature_kernel_score(
    *,
    depth: int = 5,
    value_dim: int = 1,
    use_time: bool = True,
    anchor_at_start: bool = True,
    prepend_zero_basepoint: bool = True,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Signature-kernel score loss, optionally on time-augmented paths.

    Recommended for 1D outputs where plain signatures collapse to increment-only
    information. When `use_time=True`, pySigLib time-augments paths with t in
    [0, 1]; uses truncated signatures as features with a dot-product kernel score.

    Accepts vector paths (B, T, C) or 3x3 matrix paths (B, T, 3, 3). Matrix paths are
    converted inside the loss: SO(3) via log-map to (T, 3) when value_dim=3, SPD via
    vech to (T, 6) when value_dim=6.
    """
    from pysiglib.jax_api import sig
    from taming_the_ito_lyon.utils.so3 import log_map

    depth_i = int(depth)
    value_dim_i = int(value_dim)
    if depth_i <= 0:
        raise ValueError("depth must be >= 1")
    if value_dim_i <= 0:
        raise ValueError("value_dim must be >= 1")

    def _to_euclidean(path: jax.Array) -> jax.Array:
        # Matrix-valued paths are statically disambiguated via `value_dim`.
        if not (path.ndim == 3 and path.shape[-2:] == (3, 3)):
            return path
        if value_dim_i == 3:
            r0_t = jnp.swapaxes(path[:1], -1, -2)
            return log_map(r0_t @ path)  # (T, 3)
        if value_dim_i == 6:
            return SPDManifold.vech(path)  # (T, 6)
        raise ValueError(
            f"Matrix paths require value_dim in (3, 6); got {value_dim_i}."
        )

    def _features(paths: jax.Array) -> jax.Array:
        if anchor_at_start:
            paths = paths - paths[:, :1]
        if prepend_zero_basepoint:
            paths = jnp.concatenate(
                [jnp.zeros((paths.shape[0], 1, value_dim_i), dtype=paths.dtype), paths],
                axis=1,
            )
        return sig(paths, depth_i, time_aug=use_time, end_time=1.0)

    def loss(pred: jax.Array, target: jax.Array) -> jax.Array:
        assert pred.shape == target.shape, (
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
        is_matrix_3x3 = pred.ndim == 4 and pred.shape[-2:] == (3, 3)
        if not is_matrix_3x3:
            assert int(pred.shape[-1]) == value_dim_i, (
                f"Expected value_dim={value_dim_i}, got {int(pred.shape[-1])}."
            )
        if int(pred.shape[1]) < 2:
            return jnp.asarray(0.0, dtype=jnp.float32)

        # pred is (B, T, ...); for matrix paths last two axes are (3, 3).
        phi_pred = _features(jax.vmap(_to_euclidean)(pred))
        phi_target = _features(jax.vmap(_to_euclidean)(target))
        diff = jnp.mean(phi_pred, axis=0) - jnp.mean(phi_target, axis=0)
        return jnp.sum(diff * diff)

    return loss


def branched_signature_kernel_score(
    *,
    depth: int = 2,
    use_planar: bool,
    use_time: bool,
    x_dim: int = 1,
    prepend_zero_basepoint: bool = True,
) -> Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array]:
    """Branched signature-kernel score loss (biased MMD^2 with dot-product kernel).

    Uses pySigLib branched signatures. Quadratic covariation is injected through
    pySigLib's per-segment `correction` parameter.

    Shapes
    ------
    - pred_x / target_x: (B, T), (B, T, C), or SPD matrices (B, T, 3, 3) which are
      converted to vech(X) internally.
    - Optionally pass `target_cov` as a per-step bracket density side-channel
      shaped (B, T, C*C) or (B, T, C, C).

    Multi-channel path Y has shape (T, d) with d = use_time + x_dim. Quadratic
    variation is either supplied by `target_cov` or estimated from x-increments.
    """
    import pysiglib
    from pysiglib.jax_api import branched_sig

    depth_i = int(depth)
    x_dim_i = int(x_dim)
    if depth_i <= 0:
        raise ValueError("depth must be >= 1")
    if x_dim_i <= 0:
        raise ValueError("x_dim must be >= 1")

    path_dim = x_dim_i
    sig_dim = path_dim + (1 if use_time else 0)
    pysiglib.prepare_branched_sig(
        sig_dim,
        depth_i,
        time_aug=False,
        planar=use_planar,
    )

    def _ensure_btc(x: jax.Array, name: str) -> jax.Array:
        # Accept (B,T), (B,T,C), or (B,T,3,3) SPD matrices.
        if x.ndim == 2:
            return x[..., None]
        if x.ndim == 3:
            return x
        if x.ndim == 4 and x.shape[-2:] == (3, 3):
            b, t = int(x.shape[0]), int(x.shape[1])
            return SPDManifold.vech(x.reshape((b * t, 3, 3))).reshape((b, t, 6))
        raise ValueError(
            f"Expected {name} shaped (B,T), (B,T,C), or (B,T,3,3); got {x.shape}"
        )

    def _parse_target_cov(cov: jax.Array, B: int, T: int) -> jax.Array:
        # Wishart dataset stores per-step bracket density for vech(X) as (B,T,C*C).
        if cov.ndim == 3 and cov.shape == (B, T, x_dim_i * x_dim_i):
            return cov.reshape((B, T, x_dim_i, x_dim_i))
        if cov.ndim == 4 and cov.shape == (B, T, x_dim_i, x_dim_i):
            return cov
        raise ValueError(
            f"target cov density must be (B,T,{x_dim_i * x_dim_i}) or "
            f"(B,T,{x_dim_i},{x_dim_i}); got {cov.shape}"
        )

    def loss(
        pred_x: jax.Array,
        target_x: jax.Array,
        target_cov: jax.Array | None = None,
    ) -> jax.Array:
        pred_x_btc = _ensure_btc(pred_x, "pred_x")
        target_x_btc = _ensure_btc(target_x, "target_x")
        if pred_x_btc.shape != target_x_btc.shape:
            raise ValueError(
                f"pred_x / target_x shape mismatch: {pred_x_btc.shape} vs {target_x_btc.shape}"
            )
        if int(pred_x_btc.shape[2]) != x_dim_i:
            raise ValueError(
                f"Expected x_dim={x_dim_i} channels, got {int(pred_x_btc.shape[2])}."
            )

        B, T, _ = pred_x_btc.shape
        if T < 2 or B < 1:  
            return jnp.asarray(0.0, dtype=jnp.float32)

        target_cov = (
            _parse_target_cov(target_cov, B, T) if target_cov is not None else None
        )

        dt = 1.0 / float(T - 1)

        def _path(x: jax.Array) -> jax.Array:
            path = x
            if use_time:
                ts = jnp.linspace(0.0, 1.0, T, dtype=x.dtype)
                t = jnp.broadcast_to(ts[None, :, None], (B, T, 1))
                path = jnp.concatenate([t, x], axis=-1)
            if prepend_zero_basepoint:
                path = jnp.concatenate(
                    [jnp.zeros((B, 1, sig_dim), dtype=x.dtype), path],
                    axis=1,
                )
            return path

        def _correction(x: jax.Array, cov_density: jax.Array | None) -> jax.Array:
            if cov_density is None:
                inc = jnp.diff(x, axis=1)
                dqv = jnp.einsum("btc,btd->btcd", inc, inc)
            else:
                dqv = cov_density[:, :-1] * dt

            corr = jnp.zeros((B, int(dqv.shape[1]), sig_dim, sig_dim), dtype=x.dtype)
            start = 1 if use_time else 0
            corr = corr.at[:, :, start:, start:].set(dqv.astype(x.dtype))
            if prepend_zero_basepoint:
                corr = jnp.concatenate(
                    [jnp.zeros((B, 1, sig_dim, sig_dim), dtype=x.dtype), corr],
                    axis=1,
                )
            return corr.reshape((B, int(corr.shape[1]), sig_dim * sig_dim))

        phi_pred = branched_sig(
            _path(pred_x_btc),
            depth_i,
            planar=use_planar,
            correction=_correction(pred_x_btc, None) if depth_i >= 2 else None,
        )
        phi_target = branched_sig(
            _path(target_x_btc),
            depth_i,
            planar=use_planar,
            correction=(
                _correction(target_x_btc, target_cov) if depth_i >= 2 else None
            ),
        )

        # For dot-product kernels, MMD^2 reduces to the squared distance between mean
        # embeddings — avoids the O(B^2) Gram matrix.
        diff = jnp.mean(phi_pred, axis=0) - jnp.mean(phi_target, axis=0)
        return jnp.sum(diff * diff).astype(jnp.float32)

    return loss


def _maybe_unvech_spd(
    x: jax.Array,
) -> jax.Array:
    if x.ndim >= 2 and x.shape[-2:] == (3, 3):
        return x
    if x.shape[-1] == 6:
        return SPDManifold.unvech(x)
    return x
