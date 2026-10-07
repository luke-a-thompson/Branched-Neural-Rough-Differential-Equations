import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import diffrax
import equinox as eqx
import georax
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from taming_the_ito_lyon.config.config import Config
from taming_the_ito_lyon.config.config_options import HiddenStateMode, RoughSolution
from taming_the_ito_lyon.models import BNRDE, NeuralRDE


@pytest.mark.parametrize("signature_depth", [1, 2])
def test_nrde_prepend_zero_basepoint_preserves_output_shape(
    signature_depth: int,
) -> None:
    model = NeuralRDE(
        input_path_dim=3,
        cde_state_dim=8,
        output_path_dim=2,
        vf_hidden_dim=8,
        init_hidden_dim=8,
        initial_cond_mlp_depth=2,
        vf_mlp_depth=2,
        signature_depth=signature_depth,
        signature_window_size=2,
        manifold=georax.Euclidean(),
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

    outputs = eqx.filter_jit(model)(control_values)

    assert outputs.shape == (5, 2)
    assert jnp.all(jnp.isfinite(outputs))


@pytest.mark.parametrize("adaptive", [False, True])
def test_bnrde_so3_stays_on_manifold(adaptive: bool) -> None:
    model = BNRDE(
        input_path_dim=3,
        initial_state_param_dim=6,
        output_path_dim=9,
        initial_hidden_dim=8,
        initial_cond_mlp_depth=1,
        vf_hidden_dim=8,
        vf_mlp_depth=1,
        signature_depth=3,
        signature_window_size=2,
        data_geometry=georax.SO(3),
        hidden_state_mode=HiddenStateMode.PROBLEM_MANIFOLD,
        rough_solution=RoughSolution.STRATONOVICH,
        solver=georax.CFEES25() if adaptive else georax.CG2(),
        stepsize_controller=(
            diffrax.PIDController(rtol=1e-3, atol=1e-3)
            if adaptive
            else diffrax.ConstantStepSize()
        ),
        evolving_out=True,
        prepend_zero_basepoint=False,
        key=jr.PRNGKey(2),
    )

    control_values = 0.1 * jr.normal(jr.PRNGKey(3), (5, 3), dtype=jnp.float32)
    outputs = model(control_values)

    ident = jnp.eye(3, dtype=outputs.dtype)
    gram = jnp.swapaxes(outputs, -1, -2) @ outputs
    assert outputs.shape == (5, 3, 3)
    assert jnp.allclose(gram, ident, rtol=3e-3, atol=3e-3)
    assert jnp.all(jnp.linalg.det(outputs) > 0.0)


def test_bnrde_rejects_non_georax_solver_for_problem_manifold() -> None:
    with pytest.raises(ValueError, match="requires solver"):
        BNRDE(
            input_path_dim=3,
            initial_state_param_dim=6,
            output_path_dim=9,
            initial_hidden_dim=8,
            initial_cond_mlp_depth=1,
            vf_hidden_dim=8,
            vf_mlp_depth=1,
            signature_depth=1,
            signature_window_size=2,
            data_geometry=georax.SO(3),
            hidden_state_mode=HiddenStateMode.PROBLEM_MANIFOLD,
            rough_solution=RoughSolution.STRATONOVICH,
            solver=diffrax.Tsit5(),
            key=jr.PRNGKey(7),
        )


@pytest.mark.parametrize("signature_depth", [1, 2])
def test_bnrde_spd_stays_on_manifold(signature_depth: int) -> None:
    model = BNRDE(
        input_path_dim=2,
        initial_state_param_dim=6,
        output_path_dim=6,
        initial_hidden_dim=8,
        initial_cond_mlp_depth=1,
        vf_hidden_dim=8,
        vf_mlp_depth=1,
        signature_depth=signature_depth,
        signature_window_size=2,
        data_geometry=georax.SPD(3),
        hidden_state_mode=HiddenStateMode.PROBLEM_MANIFOLD,
        rough_solution=RoughSolution.ITO,
        solver=georax.CG2(),
        evolving_out=True,
        prepend_zero_basepoint=False,
        key=jr.PRNGKey(5),
    )

    control_values = 0.1 * jr.normal(jr.PRNGKey(6), (5, 2), dtype=jnp.float32)
    outputs = model(control_values)

    assert outputs.shape == (5, 3, 3)
    assert jnp.allclose(outputs, jnp.swapaxes(outputs, -1, -2), rtol=1e-5, atol=1e-5)
    assert jnp.all(jnp.linalg.eigvalsh(outputs) > 0.0)


@pytest.mark.parametrize("model_type", ["nrde", "bnrde"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_rough_models_have_finite_training_gradients(
    model_type: str,
    adaptive: bool,
) -> None:
    controller = (
        diffrax.PIDController(rtol=1e-3, atol=1e-3)
        if adaptive
        else diffrax.ConstantStepSize()
    )
    if model_type == "nrde":
        model = NeuralRDE(
            input_path_dim=2,
            cde_state_dim=2,
            output_path_dim=1,
            vf_hidden_dim=4,
            init_hidden_dim=4,
            initial_cond_mlp_depth=1,
            vf_mlp_depth=1,
            signature_depth=2,
            signature_window_size=2,
            manifold=georax.Euclidean(),
            solver=diffrax.Tsit5(),
            stepsize_controller=controller,
            key=jr.PRNGKey(12),
        )
    else:
        model = BNRDE(
            input_path_dim=2,
            initial_state_param_dim=2,
            output_path_dim=1,
            initial_hidden_dim=4,
            initial_cond_mlp_depth=1,
            vf_hidden_dim=4,
            vf_mlp_depth=1,
            signature_depth=2,
            signature_window_size=2,
            data_geometry=georax.Euclidean(),
            hidden_state_mode=HiddenStateMode.EUCLIDEAN,
            rough_solution=RoughSolution.ITO,
            solver=diffrax.Tsit5(),
            stepsize_controller=controller,
            key=jr.PRNGKey(12),
        )
    values = 0.1 * jr.normal(jr.PRNGKey(13), (5, 2))

    loss_fn = eqx.filter_jit(
        eqx.filter_value_and_grad(lambda m: jnp.mean(m(values) ** 2))
    )
    loss, grads = loss_fn(model)

    assert jnp.isfinite(loss)
    leaves = jax.tree.leaves(grads)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(leaf != 0.0) for leaf in leaves)


def test_bnrde_same_count_control_stays_aligned() -> None:
    model = BNRDE(
        input_path_dim=3,
        initial_state_param_dim=6,
        output_path_dim=6,
        initial_hidden_dim=8,
        initial_cond_mlp_depth=1,
        vf_hidden_dim=8,
        vf_mlp_depth=1,
        signature_depth=1,
        signature_window_size=1,
        data_geometry=georax.SPD(3),
        hidden_state_mode=HiddenStateMode.PROBLEM_MANIFOLD,
        rough_solution=RoughSolution.ITO,
        solver=georax.CG2(),
        evolving_out=True,
        prepend_zero_basepoint=True,
        control_has_time_channel=True,
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


def test_bnrde_rejects_spd_latent_decoder_shape() -> None:
    with pytest.raises(ValueError, match="integrated manifold state as the output"):
        BNRDE(
            input_path_dim=3,
            initial_state_param_dim=6,
            output_path_dim=10,
            initial_hidden_dim=8,
            initial_cond_mlp_depth=1,
            vf_hidden_dim=8,
            vf_mlp_depth=1,
            signature_depth=1,
            signature_window_size=1,
            data_geometry=georax.SPD(3),
            hidden_state_mode=HiddenStateMode.PROBLEM_MANIFOLD,
            rough_solution=RoughSolution.ITO,
            solver=georax.CG2(),
            evolving_out=True,
            prepend_zero_basepoint=False,
            key=jr.PRNGKey(10),
        )


def _bnrde_config(solver: str) -> dict:
    return {
        "experiment_config": {
            "model_type": "bnrde",
            "dataset_name": "synthetic_gbm",
            "optimizer": "adam",
            "learning_rate": 5e-3,
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
        "bnrde_config": {
            "initial_state_param_dim": 9,
            "init_hidden_dim": 8,
            "vf_hidden_dim": 8,
            "initial_cond_mlp_depth": 1,
            "vf_mlp_depth": 1,
            "out_size": 9,
            "signature_depth": 1,
            "signature_window_size": 2,
            "rough_solution": "stratonovich",
        },
    }


def test_problem_manifold_config_requires_georax_solver() -> None:
    with pytest.raises(ValueError, match="requires solver"):
        Config.model_validate(_bnrde_config("tsit5"))


def test_problem_manifold_config_accepts_georax_solver() -> None:
    config = Config.model_validate(_bnrde_config("cfees25"))

    assert config.solver_config.solver.value == "cfees25"


def test_bnrde_prepend_zero_basepoint_preserves_output_shape() -> None:
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
        rough_solution=RoughSolution.STRATONOVICH,
        solver=diffrax.Tsit5(),
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
