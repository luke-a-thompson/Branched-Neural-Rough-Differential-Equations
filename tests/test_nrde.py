import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import diffrax
import georax
import jax.numpy as jnp
import jax.random as jr
import pytest

from stochastax.manifolds import EuclideanSpace, SO3
from stochastax.manifolds.spd import SPDManifold

from taming_the_ito_lyon.config.config import Config
from taming_the_ito_lyon.config.config_options import HiddenStateMode, HopfAlgebraType
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


def test_geometric_mnrde_so3_stays_on_manifold() -> None:
    model = MNDRE(
        input_path_dim=3,
        initial_state_param_dim=6,
        output_path_dim=9,
        initial_hidden_dim=8,
        initial_cond_mlp_depth=1,
        vf_hidden_dim=8,
        vf_mlp_depth=1,
        signature_depth=3,
        signature_window_size=2,
        data_manifold=SO3,
        hopf_algebra_type=HopfAlgebraType.SHUFFLE,
        hidden_state_mode=HiddenStateMode.PROBLEM_MANIFOLD,
        solver=georax.CG2(),
        stepsize_controller=diffrax.ConstantStepSize(),
        evolving_out=True,
        prepend_zero_basepoint=False,
        key=jr.PRNGKey(2),
    )

    control_values = 0.1 * jr.normal(jr.PRNGKey(3), (5, 3), dtype=jnp.float32)
    outputs = model(control_values)

    ident = jnp.eye(3, dtype=outputs.dtype)
    gram = jnp.swapaxes(outputs, -1, -2) @ outputs
    assert outputs.shape == (5, 3, 3)
    assert jnp.allclose(gram, ident, rtol=1e-3, atol=1e-3)
    assert jnp.all(jnp.linalg.det(outputs) > 0.0)


def test_geometric_mnrde_rejects_gl_lift() -> None:
    with pytest.raises(ValueError, match="GL lift"):
        MNDRE(
            input_path_dim=3,
            initial_state_param_dim=6,
            output_path_dim=9,
            initial_hidden_dim=8,
            initial_cond_mlp_depth=1,
            vf_hidden_dim=8,
            vf_mlp_depth=1,
            signature_depth=1,
            signature_window_size=2,
            data_manifold=SO3,
            hopf_algebra_type=HopfAlgebraType.GL,
            hidden_state_mode=HiddenStateMode.PROBLEM_MANIFOLD,
            solver=georax.CG2(),
            stepsize_controller=diffrax.ConstantStepSize(),
            key=jr.PRNGKey(4),
        )


def test_geometric_mnrde_rejects_non_georax_solver() -> None:
    with pytest.raises(ValueError, match="requires solver"):
        MNDRE(
            input_path_dim=3,
            initial_state_param_dim=6,
            output_path_dim=9,
            initial_hidden_dim=8,
            initial_cond_mlp_depth=1,
            vf_hidden_dim=8,
            vf_mlp_depth=1,
            signature_depth=1,
            signature_window_size=2,
            data_manifold=SO3,
            hopf_algebra_type=HopfAlgebraType.SHUFFLE,
            hidden_state_mode=HiddenStateMode.PROBLEM_MANIFOLD,
            solver=diffrax.Tsit5(),
            stepsize_controller=diffrax.ConstantStepSize(),
            key=jr.PRNGKey(7),
        )


def test_geometric_mnrde_spd_stays_on_manifold() -> None:
    model = MNDRE(
        input_path_dim=2,
        initial_state_param_dim=6,
        output_path_dim=6,
        initial_hidden_dim=8,
        initial_cond_mlp_depth=1,
        vf_hidden_dim=8,
        vf_mlp_depth=1,
        signature_depth=1,
        signature_window_size=2,
        data_manifold=SPDManifold,
        hopf_algebra_type=HopfAlgebraType.MKW,
        hidden_state_mode=HiddenStateMode.PROBLEM_MANIFOLD,
        solver=georax.CG2(),
        stepsize_controller=diffrax.ConstantStepSize(),
        evolving_out=True,
        prepend_zero_basepoint=False,
        key=jr.PRNGKey(5),
    )

    control_values = 0.1 * jr.normal(jr.PRNGKey(6), (5, 2), dtype=jnp.float32)
    outputs = model(control_values)

    assert outputs.shape == (5, 3, 3)
    assert jnp.allclose(outputs, jnp.swapaxes(outputs, -1, -2), rtol=1e-5, atol=1e-5)
    assert jnp.all(jnp.linalg.eigvalsh(outputs) > 0.0)


def test_geometric_mnrde_same_count_control_stays_aligned() -> None:
    model = MNDRE(
        input_path_dim=3,
        initial_state_param_dim=6,
        output_path_dim=6,
        initial_hidden_dim=8,
        initial_cond_mlp_depth=1,
        vf_hidden_dim=8,
        vf_mlp_depth=1,
        signature_depth=1,
        signature_window_size=1,
        data_manifold=SPDManifold,
        hopf_algebra_type=HopfAlgebraType.MKW,
        hidden_state_mode=HiddenStateMode.PROBLEM_MANIFOLD,
        solver=georax.CG2(),
        stepsize_controller=diffrax.ConstantStepSize(),
        evolving_out=True,
        prepend_zero_basepoint=True,
        virtual_brownian_refinement=2,
        key=jr.PRNGKey(8),
    )

    ts = jnp.linspace(0.0, 1.0, 9, dtype=jnp.float32)
    drivers = 0.1 * jr.normal(jr.PRNGKey(9), (9, 2), dtype=jnp.float32)
    drivers = drivers - drivers[:1]
    control_values = jnp.concatenate([ts[:, None], drivers], axis=-1)

    outputs = model(control_values)

    assert outputs.shape == (9, 3, 3)
    assert jnp.allclose(outputs, jnp.swapaxes(outputs, -1, -2), rtol=1e-5, atol=1e-5)
    assert jnp.all(jnp.linalg.eigvalsh(outputs) > 0.0)


def test_geometric_mnrde_rejects_spd_latent_decoder_shape() -> None:
    with pytest.raises(ValueError, match="integrated manifold state as the output"):
        MNDRE(
            input_path_dim=3,
            initial_state_param_dim=10,
            output_path_dim=6,
            initial_hidden_dim=8,
            initial_cond_mlp_depth=1,
            vf_hidden_dim=8,
            vf_mlp_depth=1,
            signature_depth=1,
            signature_window_size=1,
            data_manifold=SPDManifold,
            hopf_algebra_type=HopfAlgebraType.MKW,
            hidden_state_mode=HiddenStateMode.PROBLEM_MANIFOLD,
            solver=georax.CG2(),
            stepsize_controller=diffrax.ConstantStepSize(),
            evolving_out=True,
            prepend_zero_basepoint=False,
            key=jr.PRNGKey(10),
        )


def _geometric_mnrde_config(solver: str) -> dict:
    return {
        "experiment_config": {
            "model_type": "mnrde",
            "dataset_name": "synthetic_gbm",
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "loss": "mse",
            "seed": 1,
            "batch_size": 8,
            "epochs": 1,
            "early_stopping_patience": 1,
            "manifold": "so3",
            "hidden_state_mode": "problem_manifold",
            "evolving_out": True,
        },
        "solver_config": {
            "stepsize_controller": "constant",
            "solver": solver,
            "adjoint": "recursive_checkpoint",
            "rtol": 1e-3,
            "atol": 1e-3,
            "dtmin": 1e-4,
        },
        "mnrde_config": {
            "initial_state_param_dim": 9,
            "init_hidden_dim": 8,
            "vf_hidden_dim": 8,
            "initial_cond_mlp_depth": 1,
            "vf_mlp_depth": 1,
            "out_size": 9,
            "signature_depth": 1,
            "signature_window_size": 2,
            "hopf_algebra": "shuffle",
        },
    }


def test_problem_manifold_config_requires_georax_solver() -> None:
    with pytest.raises(ValueError, match="requires solver"):
        Config.model_validate(_geometric_mnrde_config("tsit5"))


def test_problem_manifold_config_accepts_georax_solver() -> None:
    config = Config.model_validate(_geometric_mnrde_config("cfees25"))

    assert config.solver_config.solver.value == "cfees25"


def test_mnrde_prepend_zero_basepoint_preserves_output_shape() -> None:
    model = MNDRE(
        input_path_dim=3,
        initial_state_param_dim=8,
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
