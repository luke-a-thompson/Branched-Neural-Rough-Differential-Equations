import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import diffrax
import georax
import jax
import jax.numpy as jnp
import jax.random as jr

from taming_the_ito_lyon.config.config import load_toml_config
from taming_the_ito_lyon.models import ManifoldNeuralODE
from taming_the_ito_lyon.training.factories import create_model
from taming_the_ito_lyon.utils.so3 import rodrigues


def test_manifold_neural_ode_euclidean_smoke() -> None:
    model = ManifoldNeuralODE(
        local_dim=3,
        anchor_dim=3,
        vf_hidden_dim=8,
        vf_mlp_depth=2,
        manifold=georax.Euclidean(),
        key=jr.PRNGKey(0),
        stepsize_controller=diffrax.ConstantStepSize(),
        dt0=0.1,
    )

    control_values = jnp.array(
        [
            [0.1, -0.2, 0.3],
            [0.2, -0.1, 0.4],
            [0.4, 0.1, 0.5],
            [0.6, 0.2, 0.6],
            [0.7, 0.3, 0.7],
        ]
    )
    xs = model(control_values)

    assert xs.shape == (5, 3)
    assert jnp.allclose(xs[0], control_values[0])
    assert model.future_only_loss is True


def test_manifold_neural_ode_so3_smoke() -> None:
    model = ManifoldNeuralODE(
        local_dim=3,
        anchor_dim=9,
        vf_hidden_dim=8,
        vf_mlp_depth=2,
        manifold=georax.SO(3),
        key=jr.PRNGKey(1),
        stepsize_controller=diffrax.ConstantStepSize(),
        dt0=0.1,
    )

    rotations = jnp.stack(
        [
            jnp.eye(3),
            rodrigues(jnp.array([0.05, 0.0, 0.0])),
            rodrigues(jnp.array([0.10, 0.0, 0.0])),
            rodrigues(jnp.array([0.15, 0.0, 0.0])),
        ],
        axis=0,
    )
    control_values = rotations.reshape(4, 9)
    xs = model(control_values)

    assert xs.shape == (4, 3, 3)
    assert jnp.allclose(xs[0], rotations[0])

    identity = jnp.eye(3)
    gram = xs[-1].T @ xs[-1]
    assert jnp.allclose(gram, identity, atol=1e-5)
    assert jnp.all(jnp.linalg.det(xs) > 0.0)


def test_m_ode_config_and_factory_smoke() -> None:
    config = load_toml_config("configs/sg_so3_sim/m_ode.toml")
    model = create_model(
        config,
        input_path_dim=9,
        output_path_dim=6,
        key=jax.random.PRNGKey(0),
    )
    assert isinstance(model, ManifoldNeuralODE)
