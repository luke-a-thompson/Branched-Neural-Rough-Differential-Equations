import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import diffrax
import georax
import jax.numpy as jnp
import jax.random as jr

from taming_the_ito_lyon.config.config_options import HiddenStateMode, RoughSolution
from taming_the_ito_lyon.models import BNRDE
from taming_the_ito_lyon.models.rough_utils import ito_correction


def test_bnrde_zero_basepoint_padding_uses_uniform_synthetic_grid() -> None:
    model = BNRDE(
        input_path_dim=3,
        initial_state_param_dim=8,
        output_path_dim=2,
        initial_hidden_dim=8,
        initial_cond_mlp_depth=2,
        vf_hidden_dim=8,
        vf_mlp_depth=2,
        signature_depth=2,
        signature_window_size=2,
        data_geometry=georax.Euclidean(),
        hidden_state_mode=HiddenStateMode.EUCLIDEAN,
        rough_solution=RoughSolution.ITO,
        solver=diffrax.Tsit5(),
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

    outputs = model(values)
    assert outputs.shape == (5, 2)
    assert jnp.all(jnp.isfinite(outputs))


def test_ito_correction_covers_every_brownian_channel() -> None:
    ts = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float32)
    values = jnp.stack(
        [ts, jnp.sqrt(ts), -jnp.sqrt(ts)],
        axis=-1,
    )

    correction = ito_correction(
        ts,
        values,
        ts[::2],
        depth=3,
        has_time_channel=True,
    )

    assert correction.shape == (2, 2, 36)
    assert jnp.allclose(correction[..., 0], 0.0)
    assert jnp.allclose(correction[..., 4], 0.25)
    assert jnp.allclose(correction[..., 8], 0.25)
    assert jnp.allclose(correction[..., 9:], 0.0)


def test_ito_correction_uses_solver_time_without_time_channel() -> None:
    ts = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float32)
    values = jnp.stack([jnp.sqrt(ts), -jnp.sqrt(ts)], axis=-1)

    correction = ito_correction(
        ts,
        values,
        ts[::2],
        depth=2,
        has_time_channel=False,
    )

    assert correction.shape == (2, 2, 4)
    assert jnp.allclose(correction[..., 0], 0.25)
    assert jnp.allclose(correction[..., 3], 0.25)
    assert jnp.allclose(correction[..., 1:3], 0.0)
