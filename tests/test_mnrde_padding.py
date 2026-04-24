import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import diffrax
import jax.numpy as jnp
import jax.random as jr

from stochastax.manifolds import EuclideanSpace

from taming_the_ito_lyon.config.config_options import HopfAlgebraType
from taming_the_ito_lyon.models import MNDRE


def test_mnrde_zero_basepoint_padding_uses_uniform_synthetic_grid() -> None:
    model = MNDRE(
        input_path_dim=3,
        cde_state_dim=8,
        output_path_dim=2,
        initial_hidden_dim=8,
        initial_cond_mlp_depth=2,
        vf_hidden_dim=8,
        vf_mlp_depth=2,
        signature_depth=2,
        signature_window_size=2,
        data_manifold=EuclideanSpace,
        hidden_manifold=EuclideanSpace,
        hopf_algebra_type=HopfAlgebraType.SHUFFLE,
        solver=diffrax.Tsit5(),
        stepsize_controller=diffrax.ConstantStepSize(),
        evolving_out=True,
        prepend_zero_basepoint=True,
        key=jr.PRNGKey(0),
    )

    ts = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float32)
    values = jnp.arange(15, dtype=jnp.float32).reshape(5, 3)

    ts_aug, values_aug = model._maybe_prepend_zero_basepoint(ts, values)

    expected_ts = jnp.array([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25], dtype=jnp.float32)
    assert jnp.allclose(ts_aug, expected_ts)
    assert values_aug.shape == (7, 3)
    assert jnp.allclose(values_aug[0], jnp.zeros((3,), dtype=jnp.float32))
    assert jnp.allclose(values_aug[-1], values[-1])
