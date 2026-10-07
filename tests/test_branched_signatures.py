import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taming_the_ito_lyon.training.losses import branched_signature_kernel_score
from taming_the_ito_lyon.training.results_gathering_fns import (
    _make_branched_ito_signature_moment_gap_metrics,
)


@pytest.mark.parametrize("use_cov_density", [False, True])
def test_branched_loss_matches_scalar_ito_signature(use_cov_density: bool) -> None:
    loss = branched_signature_kernel_score(
        depth=2,
        use_planar=False,
        use_time=False,
        x_dim=1,
        prepend_zero_basepoint=False,
    )
    pred = jnp.array([[[0.0], [1.0], [2.0]]], dtype=jnp.float32)
    target = jnp.zeros_like(pred)
    cov = jnp.zeros((1, 3, 1)) if use_cov_density else None

    value = jax.jit(loss)(pred, target, cov)

    # The scalar Itô signature is (2, (2**2 - (1**2 + 1**2))/2) = (2, 1).
    np.testing.assert_allclose(value, 5.0, atol=1e-6)


@pytest.mark.parametrize("use_planar", [False, True])
def test_branched_loss_has_finite_multichannel_gradients(use_planar: bool) -> None:
    loss = branched_signature_kernel_score(
        depth=3,
        use_planar=use_planar,
        use_time=True,
        x_dim=2,
    )
    pred = jnp.array([[[0.0, 0.0], [0.1, -0.2], [0.3, 0.1]]])
    target = jnp.zeros_like(pred)

    value, grad = jax.jit(jax.value_and_grad(loss))(pred, target)

    assert jnp.isfinite(value) and value > 0.0
    assert jnp.all(jnp.isfinite(grad)) and jnp.any(grad != 0.0)
    np.testing.assert_allclose(loss(pred, pred), 0.0, atol=1e-6)


def test_branched_metrics_use_ito_second_level() -> None:
    metric = _make_branched_ito_signature_moment_gap_metrics()
    pred = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    target = np.zeros_like(pred)

    result = metric(pred_paths=pred, target_paths=target)

    # With time augmentation, the second-level gaps are (0, 1, 1, 1).
    np.testing.assert_allclose(result["branched_sig_gap_lvl1_l2"], 2.0, atol=1e-6)
    np.testing.assert_allclose(
        result["branched_sig_gap_lvl2_full_l2"],
        np.sqrt(3.0),
        atol=1e-6,
    )
