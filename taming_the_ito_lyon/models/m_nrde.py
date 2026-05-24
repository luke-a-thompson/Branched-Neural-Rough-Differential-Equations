"""
A reimplementation of Log-NCDE using Jax and Stochastax (https://arxiv.org/abs/2402.18512).
"""

import math
from collections.abc import Callable

import diffrax
import equinox as eqx
import georax
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import roughrax
from stochastax.hopf_algebras import (
    GLHopfAlgebra,
    HopfAlgebra,
    MKWHopfAlgebra,
    ShuffleHopfAlgebra,
)
from stochastax.manifolds import SO3, EuclideanSpace, Manifold
from stochastax.manifolds.spd import SPDManifold
from stochastax.vector_field_lifts.gl_lift import form_gl_bracket_functions
from stochastax.vector_field_lifts.lie_lift import form_lyndon_bracket_functions
from stochastax.vector_field_lifts.mkw_lift import form_mkw_bracket_functions
from stochastax.vector_field_lifts.vector_field_lift_types import (
    VectorFieldBracketFunctionLift,
)

from taming_the_ito_lyon.config.config_options import HiddenStateMode, HopfAlgebraType

from .extrapolation import ExtrapolationScheme
from .logsig_cde_solve import (
    solve_cde_from_windowed_logsigs_piecewise,
)
from .logsignatures import (
    compute_disjoint_signature_times,
    compute_windowed_logsignatures_from_values,
)


def lipswish(x: jax.Array) -> jax.Array:
    return 0.909 * jnn.silu(x)


def _spd_n_from_dim(dim: int) -> int:
    disc = 8 * int(dim) + 1
    root = math.isqrt(disc)
    if root * root != disc:
        raise ValueError(f"Expected triangular SPD dimension, got {dim}.")
    return (root - 1) // 2


def _georax_geometry_for_manifold(
    manifold: type[Manifold],
    state_param_dim: int,
) -> georax.Manifold:
    if manifold is SO3:
        if int(state_param_dim) not in (6, 9):
            raise ValueError(
                "SO3 problem-manifold mode initializes from a 6D representation "
                "or a direct 3x3 state, so initial_state_param_dim must be 6 or "
                f"9; got {state_param_dim}."
            )
        return georax.SO(3)
    if manifold is SPDManifold:
        return georax.SPD(_spd_n_from_dim(state_param_dim))
    raise ValueError(
        "hidden_state_mode='problem_manifold' currently supports SO3 and SPD only."
    )


def _state_shape_for_geometry(geometry: georax.Manifold) -> tuple[int, ...]:
    if hasattr(geometry, "state_shape"):
        return tuple(int(i) for i in geometry.state_shape)
    if isinstance(geometry, (georax.SO, georax.SPD)):
        return (int(geometry.n), int(geometry.n))
    raise ValueError(f"Unsupported georax geometry: {type(geometry).__name__}.")


def _geometry_dim(geometry: georax.Manifold) -> int:
    if hasattr(geometry, "coordinate_shape"):
        return math.prod(int(i) for i in geometry.coordinate_shape)
    if isinstance(geometry, georax.SO):
        n = int(geometry.n)
        return n * (n - 1) // 2
    if isinstance(geometry, georax.SPD):
        n = int(geometry.n)
        return n * (n + 1) // 2
    raise ValueError(f"Unsupported georax geometry: {type(geometry).__name__}.")


class MNRDEFunc(eqx.Module):
    """Vector field for a MNDRE, expressed as a CDE over (flattened) log-signatures.

    This returns a matrix of shape (cde_state_dim, logsig_size) which multiplies the
    time-derivative of a cumulative log-signature control path.
    """

    input_path_dim: int
    vf_mlp: eqx.nn.MLP
    cde_state_dim: int
    hidden_manifold: type[Manifold] = eqx.field(static=True)
    hopf_algebra: HopfAlgebra = eqx.field(static=True)
    vf_lift: VectorFieldBracketFunctionLift

    def __init__(
        self,
        *,
        input_path_dim: int,
        cde_state_dim: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        hopf_algebra: HopfAlgebra,
        vf_lift: VectorFieldBracketFunctionLift,
        hidden_manifold: type[Manifold],
        key: jax.Array,
    ) -> None:
        # Vector field
        self.cde_state_dim = cde_state_dim
        self.input_path_dim = input_path_dim
        self.vf_mlp = eqx.nn.MLP(
            in_size=cde_state_dim,
            out_size=input_path_dim * cde_state_dim,
            width_size=vf_hidden_dim,
            depth=vf_mlp_depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=key,
        )

        # Rough paths
        self.hidden_manifold = hidden_manifold
        self.hopf_algebra = hopf_algebra
        self.vf_lift = vf_lift

    def __call__(self, t: jax.typing.ArrayLike, y: jax.Array, args: None) -> jax.Array:
        del t, args

        y_retracted = self.hidden_manifold.retract(y)

        def batched_field(z: jax.Array) -> jax.Array:
            return self.vf_mlp(z).reshape(self.input_path_dim, self.cde_state_dim)

        # Form the bracket functions for the batched vector field
        bracket_functions = self.vf_lift(
            batched_field,
            self.hopf_algebra,
            self.hidden_manifold(),
        )

        # Flatten the bracket functions to a single list of length m (logsig_size)
        flat_bracket_functions = [bf for level in bracket_functions for bf in level]
        # Back to tangent space. THIS EVALUATES vf_mlp(y)
        cols = [
            self.hidden_manifold.project_to_tangent(y_retracted, bf(y_retracted))
            for bf in flat_bracket_functions
        ]
        return jnp.stack(cols, axis=1)


class GeometricMNRDEFunc(eqx.Module):
    """Geometric MNRDE vector field returning frame coefficients."""

    input_path_dim: int
    vf_mlp: eqx.nn.MLP
    geometry: georax.Manifold
    manifold_dim: int = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        input_path_dim: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        geometry: georax.Manifold,
        key: jax.Array,
    ) -> None:
        self.input_path_dim = int(input_path_dim)
        self.geometry = geometry
        self.manifold_dim = _geometry_dim(geometry)
        self.state_shape = _state_shape_for_geometry(geometry)
        self.vf_mlp = eqx.nn.MLP(
            in_size=math.prod(self.state_shape),
            out_size=self.input_path_dim * self.manifold_dim,
            width_size=vf_hidden_dim,
            depth=vf_mlp_depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, y: jax.Array) -> jax.Array:
        return self.vf_mlp(jnp.ravel(y)).reshape(self.input_path_dim, self.manifold_dim)


class MNDRE(eqx.Module):
    """
    Discrete log-ODE version of NCDE that enforces Lyndon Lie polynomials.

    Usage:
    - Standard mode (n_recon=None): model(ts, coeffs) -> outputs
    - Extrapolation mode (n_recon set): model(ts, x) -> outputs
      where ts covers both reconstruction and future times
    """

    initial_cond_mlp: eqx.nn.MLP
    cde_func: MNRDEFunc | GeometricMNRDEFunc
    readout_layer: eqx.nn.Linear | None

    # Extrapolation scheme
    extrapolation_scheme: ExtrapolationScheme | None
    n_recon: int | None = eqx.field(static=True)

    # Static configuration
    hopf_algebra: HopfAlgebra | None = eqx.field(static=True)
    vf_lift: VectorFieldBracketFunctionLift | None = eqx.field(static=True)
    data_manifold: type[Manifold] = eqx.field(static=True)
    hidden_manifold: type[Manifold] = eqx.field(static=True)
    hidden_state_mode: HiddenStateMode = eqx.field(static=True)
    geometry: georax.Manifold | None
    rough_solution: str = eqx.field(static=True)
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    signature_depth: int = eqx.field(static=True)
    signature_window_size: int = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)
    prepend_zero_basepoint: bool = eqx.field(static=True)
    brownian_channels: tuple[int, ...] | None = eqx.field(static=True)
    brownian_corr: float | None = eqx.field(static=True)
    virtual_brownian_refinement: int = eqx.field(static=True)

    # Solver configuration (matches NeuralCDE/NeuralRDE pattern)
    solver: diffrax.AbstractSolver = eqx.field(static=True)
    adjoint: diffrax.AbstractAdjoint = eqx.field(static=True)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(static=True)

    def __init__(
        self,
        input_path_dim: int,
        initial_state_param_dim: int,
        output_path_dim: int,
        initial_hidden_dim: int,
        initial_cond_mlp_depth: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        signature_depth: int,
        signature_window_size: int,
        *,
        key: jax.Array,
        data_manifold: type[Manifold],
        hopf_algebra_type: HopfAlgebraType,
        hidden_state_mode: HiddenStateMode = HiddenStateMode.EUCLIDEAN,
        solver: diffrax.AbstractSolver = diffrax.Tsit5(),
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller: diffrax.AbstractStepSizeController,
        # IMPORTANT: default to identity to avoid artificially symmetrising/skew-clipping
        # the output distribution (e.g. tanh produces symmetric outputs about 0).
        # This matches the NCDE/LogNCDE defaults in this repo.
        readout_activation: Callable[[jax.Array], jax.Array] = lambda x: x,
        evolving_out: bool = True,
        prepend_zero_basepoint: bool = True,
        extrapolation_scheme: ExtrapolationScheme | None = None,
        n_recon: int | None = None,
        brownian_channels: list[int] | None = None,
        brownian_corr: float | None = None,
        virtual_brownian_refinement: int = 1,
    ) -> None:
        num_keys = 3
        keys = jr.split(key, num_keys)
        k1 = keys[0]
        k_readout = keys[1]
        vf_key = keys[2]

        # Rough paths
        if hidden_state_mode == HiddenStateMode.PROBLEM_MANIFOLD and not isinstance(
            solver, (georax.CG2, georax.CFEES25)
        ):
            raise ValueError(
                "hidden_state_mode='problem_manifold' requires solver to be "
                "georax.CG2() or georax.CFEES25()."
            )
        self.data_manifold = data_manifold
        self.hidden_state_mode = hidden_state_mode
        self.hidden_manifold = (
            EuclideanSpace
            if hidden_state_mode == HiddenStateMode.EUCLIDEAN
            else data_manifold
        )
        self.signature_depth = signature_depth
        self.signature_window_size = signature_window_size
        self.brownian_channels = (
            tuple(int(i) for i in brownian_channels)
            if brownian_channels is not None
            else None
        )
        self.brownian_corr = float(brownian_corr) if brownian_corr is not None else None
        self.virtual_brownian_refinement = int(virtual_brownian_refinement)
        self.prepend_zero_basepoint = prepend_zero_basepoint
        if hidden_state_mode == HiddenStateMode.PROBLEM_MANIFOLD:
            if hopf_algebra_type == HopfAlgebraType.GL:
                raise ValueError(
                    "GL lift currently supports only Euclidean hidden states."
                )
            self.hopf_algebra = None
            self.vf_lift = None
        else:
            match hopf_algebra_type:
                case HopfAlgebraType.SHUFFLE:
                    self.hopf_algebra = ShuffleHopfAlgebra.build(
                        input_path_dim, signature_depth
                    )
                    self.vf_lift = form_lyndon_bracket_functions
                case HopfAlgebraType.GL:
                    self.hopf_algebra = GLHopfAlgebra.build(
                        input_path_dim, signature_depth
                    )
                    self.vf_lift = form_gl_bracket_functions
                case HopfAlgebraType.MKW:
                    self.hopf_algebra = MKWHopfAlgebra.build(
                        input_path_dim, signature_depth
                    )
                    self.vf_lift = form_mkw_bracket_functions
                case _:
                    raise ValueError(
                        f"Unsupported Hopf algebra type: {hopf_algebra_type}"
                    )

        # Module
        self.initial_cond_mlp = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=initial_state_param_dim,
            width_size=initial_hidden_dim,
            depth=initial_cond_mlp_depth,
            activation=lipswish,
            key=k1,
        )
        self.geometry = (
            None
            if hidden_state_mode == HiddenStateMode.EUCLIDEAN
            else _georax_geometry_for_manifold(data_manifold, initial_state_param_dim)
        )
        self.rough_solution = (
            "ito"
            if hidden_state_mode == HiddenStateMode.PROBLEM_MANIFOLD
            and hopf_algebra_type == HopfAlgebraType.MKW
            else "stratonovich"
        )
        if self.geometry is not None:
            expected_output_dim = (
                _geometry_dim(self.geometry)
                if isinstance(self.geometry, georax.SPD)
                else math.prod(_state_shape_for_geometry(self.geometry))
            )
            if int(output_path_dim) != expected_output_dim:
                raise ValueError(
                    "hidden_state_mode='problem_manifold' uses the integrated "
                    "manifold state as the output, so output_path_dim must match "
                    f"the geometry output dimension ({expected_output_dim}); got "
                    f"{output_path_dim}."
                )
        if self.geometry is None:
            assert self.hopf_algebra is not None
            assert self.vf_lift is not None
            self.cde_func = MNRDEFunc(
                input_path_dim=input_path_dim,
                cde_state_dim=initial_state_param_dim,
                vf_hidden_dim=vf_hidden_dim,
                vf_mlp_depth=vf_mlp_depth,
                hopf_algebra=self.hopf_algebra,
                vf_lift=self.vf_lift,
                hidden_manifold=self.hidden_manifold,
                key=vf_key,
            )
        else:
            self.cde_func = GeometricMNRDEFunc(
                input_path_dim=input_path_dim,
                vf_hidden_dim=vf_hidden_dim,
                vf_mlp_depth=vf_mlp_depth,
                geometry=self.geometry,
                key=vf_key,
            )
        self.readout_layer = (
            None
            if self.geometry is not None
            else eqx.nn.Linear(
                in_features=initial_state_param_dim,
                out_features=output_path_dim,
                use_bias=True,
                key=k_readout,
            )
        )

        self.readout_activation = readout_activation

        # Static configuration
        self.extrapolation_scheme = extrapolation_scheme
        self.n_recon = n_recon
        self.evolving_out = evolving_out
        self.solver = solver
        self.adjoint = adjoint
        self.stepsize_controller = stepsize_controller

    def _maybe_prepend_zero_basepoint(
        self, ts: jax.Array, control_values: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if not self.prepend_zero_basepoint:
            return ts, control_values

        if int(ts.shape[0]) < 2:
            raise ValueError(
                "Expected at least two timestamps when prepending a basepoint."
            )
        dt = ts[1] - ts[0]
        zero0 = jnp.zeros(
            (1, int(control_values.shape[-1])), dtype=control_values.dtype
        )
        ts0 = ts[:1] - dt
        ts_aug = jnp.concatenate([ts0, ts], axis=0)
        values_aug = jnp.concatenate([zero0, control_values], axis=0)

        # Keep the disjoint-window partition valid after the synthetic prefix point.
        step = int(self.signature_window_size)
        remainder = (int(values_aug.shape[0]) - 1) % step
        if remainder == 0:
            return ts_aug, values_aug

        pad_points = step - remainder
        ts_pad = ts_aug[-1] + dt * jnp.arange(
            1,
            pad_points + 1,
            dtype=ts.dtype,
        )
        values_pad = jnp.repeat(values_aug[-1:], pad_points, axis=0)
        return (
            jnp.concatenate([ts_aug, ts_pad], axis=0),
            jnp.concatenate([values_aug, values_pad], axis=0),
        )

    def _apply_readout(self, hidden_states: jax.Array) -> jax.Array:
        """Apply readout to hidden states."""
        if self.hidden_state_mode == HiddenStateMode.PROBLEM_MANIFOLD:
            return hidden_states

        assert self.readout_layer is not None

        def apply_single(y: jax.Array) -> jax.Array:
            activation = self.readout_activation(self.readout_layer(y))
            if self.data_manifold is SPDManifold:
                matrix = SPDManifold.unvech(activation)
                return SPDManifold.retract(matrix)
            return self.data_manifold.retract(activation)

        return jax.vmap(apply_single)(hidden_states)

    def _solve_geometric_from_values(
        self,
        ts: jax.Array,
        control_values: jax.Array,
        y0: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        assert self.geometry is not None
        assert isinstance(self.cde_func, GeometricMNRDEFunc)

        driver = diffrax.LinearInterpolation(ts=ts, ys=control_values)
        signature_ts = compute_disjoint_signature_times(
            ts, int(self.signature_window_size)
        )
        control = roughrax.SignatureInterpolation(
            driver,
            signature_ts,
            depth=int(self.signature_depth),
            solution=self.rough_solution,
        )

        def vector_field(y: jax.Array) -> jax.Array:
            return self.cde_func(y)

        term = roughrax.RoughTerm(vector_field, control, self.geometry)
        solution = diffrax.diffeqsolve(
            term,
            roughrax.LogODE(self.solver),
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.StepTo(signature_ts),
            saveat=diffrax.SaveAt(ts=ts),
            adjoint=self.adjoint,
            max_steps=int(signature_ts.shape[0]) + 4,
        )
        assert solution.ys is not None
        return solution.ys, solution.stats

    def _initial_hidden(self, x0: jax.Array) -> jax.Array:
        raw = self.initial_cond_mlp(x0)
        if self.hidden_state_mode == HiddenStateMode.EUCLIDEAN:
            return raw
        if self.data_manifold is SO3:
            if raw.shape[-1] == math.prod(self.cde_func.state_shape):
                return SO3.retract(raw.reshape(self.cde_func.state_shape))
            return SO3.retract(raw)
        if self.data_manifold is SPDManifold:
            sym = SPDManifold.unvech(raw)
            sym = 0.5 * (sym + jnp.swapaxes(sym, -1, -2))
            evals, evecs = jnp.linalg.eigh(sym)
            evals = jnp.clip(evals, -8.0, 8.0)
            return (evecs * jnp.exp(evals)[..., None, :]) @ jnp.swapaxes(
                evecs, -1, -2
            )
        raise ValueError(
            "Could not map initial condition to the problem manifold. "
            f"Got raw initial shape {raw.shape}."
        )

    def _forward_with_values(
        self,
        ts: jax.Array,
        control_values: jax.Array,
    ) -> jax.Array:
        """Core forward pass given sampled control values (standard mode fast path)."""
        x0 = control_values[0]
        h0 = self._initial_hidden(x0)

        if self.geometry is not None:
            ys, _ = self._solve_geometric_from_values(ts, control_values, h0)
            return ys

        # Single-window mode
        logsigs = compute_windowed_logsignatures_from_values(
            control_values,
            self.hopf_algebra,
            self.signature_depth,
            self.signature_window_size,
            brownian_channels=(
                list(self.brownian_channels)
                if self.brownian_channels is not None
                else None
            ),
            brownian_corr=self.brownian_corr,
        )
        ys, _ = solve_cde_from_windowed_logsigs_piecewise(
            ts,
            logsigs,
            signature_window_size=int(self.signature_window_size),
            cde_func=self.cde_func,
            y0=h0,
            solver=self.solver,
            adjoint=self.adjoint,
            stepsize_controller=self.stepsize_controller,
        )
        return ys

    def _integration_steps_with_values(
        self,
        ts: jax.Array,
        control_values: jax.Array,
    ) -> jax.Array:
        x0 = control_values[0]
        h0 = self._initial_hidden(x0)

        if self.geometry is not None:
            _, stats = self._solve_geometric_from_values(ts, control_values, h0)
            return jnp.asarray(stats["num_steps"], dtype=jnp.float32)

        logsigs = compute_windowed_logsignatures_from_values(
            control_values,
            self.hopf_algebra,
            self.signature_depth,
            self.signature_window_size,
            brownian_channels=(
                list(self.brownian_channels)
                if self.brownian_channels is not None
                else None
            ),
            brownian_corr=self.brownian_corr,
        )
        _, stats = solve_cde_from_windowed_logsigs_piecewise(
            ts,
            logsigs,
            signature_window_size=int(self.signature_window_size),
            cde_func=self.cde_func,
            y0=h0,
            solver=self.solver,
            adjoint=self.adjoint,
            stepsize_controller=self.stepsize_controller,
        )
        return jnp.asarray(stats["num_steps"], dtype=jnp.float32)

    def __call__(
        self,
        control_values: jax.Array,
    ) -> jax.Array:
        """
        Forward pass.

        Standard mode (self.extrapolation_scheme=None):
            model(control_values) -> outputs

        Extrapolation mode (self.extrapolation_scheme is set):
            model(control_values) -> outputs
        """
        length = control_values.shape[0]
        ts = jnp.linspace(0.0, 1.0, length, dtype=control_values.dtype)  # (T,)
        if self.extrapolation_scheme is not None:
            assert self.n_recon is not None, (
                "n_recon must be set when using extrapolation_scheme"
            )
            control, _ = self.extrapolation_scheme.create_control(
                ts, control_values, self.n_recon
            )
            control_values = jax.vmap(control.evaluate)(ts)
            ts_aug, control_values_aug = self._maybe_prepend_zero_basepoint(
                ts, control_values
            )
            hidden = self._forward_with_values(ts_aug, control_values_aug)
            outputs = self._apply_readout(hidden)

            if self.prepend_zero_basepoint:
                outputs = outputs[1 : 1 + length]
            return outputs
        else:
            # Standard mode
            ts_aug, control_values_aug = self._maybe_prepend_zero_basepoint(
                ts, control_values
            )
            hidden = self._forward_with_values(ts_aug, control_values_aug)

            if self.evolving_out:
                outputs = self._apply_readout(hidden)
                if self.prepend_zero_basepoint:
                    outputs = outputs[1 : 1 + length]
                return outputs

            return self._apply_readout(hidden[-1:])[0]

    def integration_steps(self, control_values: jax.Array) -> jax.Array:
        length = control_values.shape[0]
        ts = jnp.linspace(0.0, 1.0, length, dtype=control_values.dtype)
        if self.extrapolation_scheme is not None:
            assert self.n_recon is not None
            control, _ = self.extrapolation_scheme.create_control(
                ts, control_values, self.n_recon
            )
            control_values = jax.vmap(control.evaluate)(ts)
        ts_aug, control_values_aug = self._maybe_prepend_zero_basepoint(
            ts, control_values
        )
        return self._integration_steps_with_values(ts_aug, control_values_aug)
