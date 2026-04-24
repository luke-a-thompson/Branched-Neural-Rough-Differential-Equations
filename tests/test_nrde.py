import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import diffrax
import jax.numpy as jnp
import jax.random as jr

from stochastax.manifolds import EuclideanSpace

from taming_the_ito_lyon.config.config_options import HopfAlgebraType
from taming_the_ito_lyon.models import MNDRE, NeuralRDE


def test_nrde_prepend_zero_basepoint_preserves_output_shape() -> None:
    model = NeuralRDE(
        input_path_dim=3,
        cde_state_dim=8,
        output_path_dim=2,
        vf_hidden_dim=8,
        init_hidden_dim=8,
        initial_cond_mlp_depth=2,
        vf_mlp_depth=2,
        signature_depth=2,
        signature_window_size=2,
        manifold=EuclideanSpace,
        solver=diffrax.Tsit5(),
        stepsize_controller=diffrax.ConstantStepSize(),
        evolving_out=True,
        prepend_zero_basepoint=True,
        key=jr.PRNGKey(0),
    )

    control_values = jnp.array(
        [
            [0.2, -0.1, 0.3],
            [0.4, 0.0, 0.1],
            [0.5, 0.2, -0.2],
            [0.8, 0.1, 0.0],
            [1.0, -0.2, 0.4],
        ],
        dtype=jnp.float32,
    )

    outputs = model(control_values)

    assert outputs.shape == (5, 2)


def test_mnrde_prepend_zero_basepoint_preserves_output_shape() -> None:
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
        hopf_algebra_type=HopfAlgebraType.SHUFFLE,
        solver=diffrax.Tsit5(),
        stepsize_controller=diffrax.ConstantStepSize(),
        evolving_out=True,
        prepend_zero_basepoint=True,
        key=jr.PRNGKey(1),
    )

    control_values = jnp.array(
        [
            [0.2, -0.1, 0.3],
            [0.4, 0.0, 0.1],
            [0.5, 0.2, -0.2],
            [0.8, 0.1, 0.0],
            [1.0, -0.2, 0.4],
        ],
        dtype=jnp.float32,
    )

    outputs = model(control_values)

    assert outputs.shape == (5, 2)
