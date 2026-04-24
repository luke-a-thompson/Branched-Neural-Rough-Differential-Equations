import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
from stochastax.manifolds.spd import SPDManifold

from taming_the_ito_lyon.config.config import load_toml_config
from taming_the_ito_lyon.training.factories import create_grad_batch_loss_fns
from taming_the_ito_lyon.training.losses import (
    initial_step_log_eigenvalue_loss,
    signature_kernel_score,
)


def test_sigker_factory_adds_explicit_initial_step_loss() -> None:
    config = load_toml_config("configs/spd_covariance/nrde.toml")
    _, _, loss_on_preds_fn = create_grad_batch_loss_fns(config, output_path_dim=6)

    preds = jnp.array(
        [
            [
                [0.20, -0.10, 0.30, 0.40, -0.20, 0.10],
                [0.25, -0.05, 0.28, 0.42, -0.18, 0.12],
                [0.30, 0.00, 0.26, 0.44, -0.16, 0.14],
            ]
        ],
        dtype=jnp.float32,
    )
    target = jnp.array(
        [
            [
                [0.10, -0.20, 0.20, 0.30, -0.10, 0.00],
                [0.24, -0.04, 0.27, 0.41, -0.17, 0.11],
                [0.31, 0.01, 0.25, 0.45, -0.15, 0.15],
            ]
        ],
        dtype=jnp.float32,
    )
    dummy_control = jnp.zeros((1, 3, 2), dtype=jnp.float32)
    dummy_driver = jnp.zeros((1, 3, 36), dtype=jnp.float32)

    sig_loss = signature_kernel_score(
        value_dim=6,
        anchor_at_start=False,
        prepend_zero_basepoint=True,
    )(preds, target)
    ic_loss = initial_step_log_eigenvalue_loss(
        SPDManifold.unvech(preds), SPDManifold.unvech(target)
    )

    actual = loss_on_preds_fn(preds, target, dummy_control, dummy_driver)

    assert jnp.allclose(actual, sig_loss + ic_loss, atol=1e-4, rtol=1e-4)
