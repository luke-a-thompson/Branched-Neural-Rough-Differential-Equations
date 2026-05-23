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
    information. When `use_time=True`, builds augmented paths (t, x_t) with t in
    [0, 1]; uses truncated log-signatures as features with a dot-product kernel score.

    Accepts vector paths (B, T, C) or 3x3 matrix paths (B, T, 3, 3). Matrix paths are
    converted inside the loss: SO(3) via log-map to (T, 3) when value_dim=3, SPD via
    vech to (T, 6) when value_dim=6.
    """
    from stochastax.control_lifts import compute_path_signature
    from stochastax.hopf_algebras import ShuffleHopfAlgebra

    from taming_the_ito_lyon.utils.so3 import log_map

    depth_i = int(depth)
    value_dim_i = int(value_dim)
    ambient_dim = value_dim_i + (1 if use_time else 0)
    hopf = ShuffleHopfAlgebra.build(ambient_dim=ambient_dim, depth=depth_i)

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

    def loss(pred: jax.Array, target: jax.Array) -> jax.Array:
        assert pred.shape == target.shape, (
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
        is_matrix_3x3 = pred.ndim == 4 and pred.shape[-2:] == (3, 3)
        if not is_matrix_3x3:
            assert int(pred.shape[-1]) == value_dim_i, (
                f"Expected value_dim={value_dim_i}, got {int(pred.shape[-1])}."
            )

        # pred is (B, T, ...); for matrix paths last two axes are (3, 3).
        ts_col = (
            jnp.linspace(0.0, 1.0, int(pred.shape[1]), dtype=pred.dtype)[:, None]
            if use_time
            else None
        )

        def _phi(path: jax.Array) -> jax.Array:
            path = _to_euclidean(path)
            if anchor_at_start:
                path = path - path[:1]
            aug = (
                jnp.concatenate([ts_col, path], axis=-1) if use_time else path
            )  # (T, ambient_dim)
            if prepend_zero_basepoint:
                # Signatures depend on increments, so absolute level is invisible.
                # Prepending a zero basepoint makes the first increment equal to x0.
                aug = jnp.concatenate(
                    [jnp.zeros((1, ambient_dim), dtype=aug.dtype), aug], axis=0
                )
            return compute_path_signature(
                path=aug, depth=depth_i, hopf=hopf, mode="full"
            ).flatten()

        phi_pred = jax.vmap(_phi)(pred)
        phi_target = jax.vmap(_phi)(target)
        k_pp = jnp.mean(phi_pred @ phi_pred.T)
        k_tt = jnp.mean(phi_target @ phi_target.T)
        k_pt = jnp.mean(phi_pred @ phi_target.T)
        return k_pp + k_tt - 2.0 * k_pt

    return loss


def branched_signature_kernel_score(
    *,
    depth: int = 4,
    use_planar: bool,
    use_time: bool,
    use_w: bool,
    x_dim: int = 1,
    anchor_at_start: bool = True,
    prepend_zero_basepoint: bool = True,
) -> Callable[[jax.Array, jax.Array, jax.Array | None, jax.Array | None], jax.Array]:
    """Branched signature-kernel score loss (biased MMD^2 with dot-product kernel).

    Mirrors `truncated_sig_loss_time_augmented` but uses a branched Itô signature
    (GL/MKW Hopf algebra). Quadratic covariation is injected as per-step increments
    `Δ⟨Y⟩_k`.

    Shapes
    ------
    - pred_x / target_x: (B, T), (B, T, C), or SPD matrices (B, T, 3, 3) which are
      converted to vech(X) internally.
    - If `use_w=True`, pass `pred_aux` / `target_aux` as the scalar auxiliary path
      (B, T), so the lifted path is (t, w, x).
    - If `use_w=False`, optionally pass `target_cov` as a per-step bracket density
      side-channel shaped (B, T, C*C) or (B, T, C, C).

    Multi-channel path Y has shape (T, d) with d = use_time + use_w + x_dim. The
    auxiliary `w` channel is part of Y only; quadratic variation is either supplied
    by `target_cov` or estimated from x-increments.
    """
    from stochastax.control_lifts import (
        compute_nonplanar_branched_signature,
        compute_planar_branched_signature,
    )
    from stochastax.hopf_algebras import GLHopfAlgebra, MKWHopfAlgebra

    depth_i = int(depth)
    x_dim_i = int(x_dim)
    if depth_i <= 0:
        raise ValueError("depth must be >= 1")
    if x_dim_i <= 0:
        raise ValueError("x_dim must be >= 1")

    d = (1 if use_time else 0) + (1 if use_w else 0) + x_dim_i
    cov_start = (1 if use_time else 0) + (1 if use_w else 0)
    hopf_cls = MKWHopfAlgebra if use_planar else GLHopfAlgebra
    hopf = hopf_cls.build(ambient_dim=d, depth=depth_i)
    compute_sig = (
        compute_planar_branched_signature
        if use_planar
        else compute_nonplanar_branched_signature
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
        pred_aux: jax.Array | None = None,
        target_aux: jax.Array | None = None,
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

        if use_w:
            if pred_aux is None or target_aux is None:
                raise ValueError("use_w=True requires pred_aux and target_aux.")
            pred_w_ = pred_aux if pred_aux.ndim == 2 else pred_aux[..., 0]
            target_w_ = target_aux if target_aux.ndim == 2 else target_aux[..., 0]
            if pred_w_.shape != (B, T) or target_w_.shape != (B, T):
                raise ValueError(
                    f"pred_aux/target_aux must be (B,T)=({B},{T}); "
                    f"got {pred_w_.shape}, {target_w_.shape}"
                )
        else:
            pred_w_ = target_w_ = None
            target_cov = (
                _parse_target_cov(target_cov, B, T) if target_cov is not None else None
            )

        ts = jnp.linspace(0.0, 1.0, T, dtype=pred_x_btc.dtype)
        dt = ts[1:] - ts[:-1]

        def _augment(x_path: jax.Array, w_path: jax.Array | None) -> jax.Array:
            if anchor_at_start:
                x_path = x_path - x_path[:1]
                if w_path is not None:
                    w_path = w_path - w_path[:1]
            cols: list[jax.Array] = []
            if use_time:
                cols.append(ts[:, None])
            if use_w:
                assert w_path is not None
                cols.append(w_path[:, None])
            cols.append(x_path)
            y = jnp.concatenate(cols, axis=1)
            if not prepend_zero_basepoint:
                return y
            return jnp.concatenate([jnp.zeros((1, d), dtype=y.dtype), y], axis=0)

        def _embed_cov(dqv: jax.Array, dtype: jnp.dtype) -> jax.Array:
            # dqv: (T-1, C, C) -> cov_increments embedded in (t, w, x) block coordinates.
            cov = jnp.zeros((int(dqv.shape[0]), d, d), dtype=dtype)
            cov = cov.at[:, cov_start:, cov_start:].set(dqv)
            if not prepend_zero_basepoint:
                return cov
            return jnp.concatenate([jnp.zeros((1, d, d), dtype=dtype), cov], axis=0)

        def _phi(
            x_path: jax.Array, w_path: jax.Array | None, cov_density: jax.Array | None
        ) -> jax.Array:
            # cov_density=None => estimate QV from x increments; else use provided density.
            if cov_density is None:
                inc = jnp.diff(x_path, axis=0)
                dqv = jnp.einsum("tc,td->tcd", inc, inc)
            else:
                dqv = cov_density[:-1] * dt[:, None, None]
            cov_inc = _embed_cov(dqv, x_path.dtype)
            path = _augment(x_path, w_path)
            return (
                compute_sig(path, depth_i, hopf, "full", cov_increments=cov_inc)
                .log()
                .flatten()
            )

        if use_w:
            assert pred_w_ is not None and target_w_ is not None
            phi_pred = jax.vmap(lambda x, w: _phi(x, w, None))(pred_x_btc, pred_w_)
            phi_target = jax.vmap(lambda x, w: _phi(x, w, None))(
                target_x_btc, target_w_
            )
        else:
            phi_pred = jax.vmap(lambda x: _phi(x, None, None))(pred_x_btc)
            if target_cov is None:
                phi_target = jax.vmap(lambda x: _phi(x, None, None))(target_x_btc)
            else:
                phi_target = jax.vmap(lambda x, cov: _phi(x, None, cov))(
                    target_x_btc, target_cov
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
