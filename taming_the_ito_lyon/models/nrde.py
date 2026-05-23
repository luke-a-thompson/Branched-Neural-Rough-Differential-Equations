"""
A reimplementation of Neural Rough Differential Equations (NRDE) using Jax and Stochastax (https://arxiv.org/abs/2009.08295).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
from collections.abc import Callable
import diffrax

from stochastax.hopf_algebras import ShuffleHopfAlgebra
from stochastax.manifolds import Manifold
from stochastax.manifolds.spd import SPDManifold
from .logsignatures import (
    compute_windowed_logsignatures_from_values,
)
from .extrapolation import ExtrapolationScheme
from .logsig_cde_solve import (
    solve_cde_from_windowed_logsigs,
)


class NRDEFunc(eqx.Module):
    """
    Vector field for a Neural RDE in log-ODE form.

    Given hidden state y in R^{cde_state_dim}, returns matrix in
    R^{cde_state_dim x logsig_size} which multiplies the log-signature vector
    on each interval.
    """

    vf_mlp: eqx.nn.MLP
    cde_state_dim: int
    shuffle_hopf_algebra: ShuffleHopfAlgebra = eqx.field(static=True)
    logsig_size: int

    def __init__(
        self,
        *,
        input_path_dim: int,
        cde_state_dim: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        shuffle_hopf_algebra: ShuffleHopfAlgebra,
        key: jax.Array,
    ) -> None:
        self.cde_state_dim = cde_state_dim
        self.shuffle_hopf_algebra = shuffle_hopf_algebra
        self.logsig_size = shuffle_hopf_algebra.basis_size()
        self.vf_mlp = eqx.nn.MLP(
            in_size=cde_state_dim,
            out_size=cde_state_dim
            * self.logsig_size,  # NRDE outputs one element per log-signature coefficient
            width_size=vf_hidden_dim,
            depth=vf_mlp_depth,
            activation=jnn.softplus,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, t: jax.typing.ArrayLike, y: jax.Array, args: None) -> jax.Array:
        del t, args
        out = self.vf_mlp(y)
        return out.reshape(self.cde_state_dim, self.logsig_size)


class NeuralRDE(eqx.Module):
    """
    Neural Rough Differential Equation (log-ODE) model.

    Usage
    - Provide `ts` and either a `diffrax` control path or cubic interpolation coeffs.
    - The model computes per-interval log-signatures and applies discrete
      log-ODE updates with a readout on the hidden state.
    """

    # Modules
    initial: eqx.nn.MLP
    cde_func: NRDEFunc
    readout_layer: eqx.nn.Linear

    # Static configuration
    shuffle_hopf_algebra: ShuffleHopfAlgebra = eqx.field(static=True)
    manifold: type[Manifold] = eqx.field(static=True)
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    signature_depth: int = eqx.field(static=True)
    signature_window_size: int = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)
    prepend_zero_basepoint: bool = eqx.field(static=True)

    # Extrapolation scheme
    extrapolation_scheme: ExtrapolationScheme | None = eqx.field(static=True)
    n_recon: int | None = eqx.field(static=True)

    # Solver configuration (matches NeuralCDE pattern)
    solver: diffrax.AbstractAdaptiveSolver = eqx.field(static=True)
    adjoint: diffrax.AbstractAdjoint = eqx.field(static=True)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(static=True)
    dt0: float | None = eqx.field(static=True)

    def __init__(
        self,
        input_path_dim: int,
        cde_state_dim: int,
        output_path_dim: int,
        vf_hidden_dim: int,
        init_hidden_dim: int,
        initial_cond_mlp_depth: int,
        vf_mlp_depth: int,
        signature_depth: int,
        signature_window_size: int,
        *,
        key: jax.Array,
        manifold: type[Manifold],
        readout_activation: Callable[[jax.Array], jax.Array] = lambda x: x,
        solver: diffrax.AbstractAdaptiveSolver = diffrax.Tsit5(),
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller: diffrax.AbstractStepSizeController,
        dt0: float | None = None,
        evolving_out: bool = True,
        prepend_zero_basepoint: bool = False,
        extrapolation_scheme: ExtrapolationScheme | None = None,
        n_recon: int | None = None,
    ) -> None:
        k1, k2, k3 = jr.split(key, 3)
        self.shuffle_hopf_algebra = ShuffleHopfAlgebra.build(
            input_path_dim, signature_depth
        )
        # Initial state from initial control value (matches NCDE style)
        self.initial = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=cde_state_dim,
            width_size=init_hidden_dim,
            depth=initial_cond_mlp_depth,
            activation=jnn.softplus,
            key=k1,
        )
        self.cde_func = NRDEFunc(
            input_path_dim=input_path_dim,
            cde_state_dim=cde_state_dim,
            vf_hidden_dim=vf_hidden_dim,
            vf_mlp_depth=vf_mlp_depth,
            shuffle_hopf_algebra=self.shuffle_hopf_algebra,
            key=k2,
        )
        self.readout_layer = eqx.nn.Linear(
            in_features=cde_state_dim,
            out_features=output_path_dim,
            use_bias=True,
            key=k3,
        )
        self.readout_activation = readout_activation
        self.manifold = manifold
        self.signature_depth = signature_depth
        self.signature_window_size = signature_window_size
        self.evolving_out = evolving_out
        self.prepend_zero_basepoint = prepend_zero_basepoint
        self.extrapolation_scheme = extrapolation_scheme
        self.n_recon = n_recon

        self.solver = solver
        self.adjoint = adjoint
        self.stepsize_controller = stepsize_controller
        self.dt0 = dt0

    def _maybe_prepend_zero_basepoint(
        self, ts: jax.Array, control_values: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if not self.prepend_zero_basepoint:
            return ts, control_values

        zero0 = jnp.zeros(
            (1, int(control_values.shape[-1])), dtype=control_values.dtype
        )
        ts_aug = jnp.concatenate([ts[:1], ts], axis=0)
        values_aug = jnp.concatenate([zero0, control_values], axis=0)

        # Keep the disjoint-window partition valid after the synthetic prefix point.
        step = int(self.signature_window_size)
        remainder = (int(values_aug.shape[0]) - 1) % step
        if remainder == 0:
            return ts_aug, values_aug

        pad_points = step - remainder
        ts_pad = jnp.repeat(ts_aug[-1:], pad_points, axis=0)
        values_pad = jnp.repeat(values_aug[-1:], pad_points, axis=0)
        return (
            jnp.concatenate([ts_aug, ts_pad], axis=0),
            jnp.concatenate([values_aug, values_pad], axis=0),
        )

    def _project_readout(self, activation: jax.Array) -> jax.Array:
        if issubclass(self.manifold, SPDManifold):
            matrix = SPDManifold.unvech(activation)
            return SPDManifold.retract(matrix)
        if activation.shape[-1] == 9:
            return self.manifold.retract(jnp.reshape(activation, (3, 3)))
        return self.manifold.retract(activation)

    def _forward_with_values(
        self,
        ts: jax.Array,
        control_values: jax.Array,
    ) -> jax.Array:
        """Core forward pass given sampled control values.

        This avoids constructing a control path and avoids evaluating it at `ts`.
        """
        x0 = control_values[0]
        h0 = self.initial(x0)

        logsigs = compute_windowed_logsignatures_from_values(
            control_values,
            self.shuffle_hopf_algebra,
            self.signature_depth,
            self.signature_window_size,
        )  # (num_windows, logsig_size)
        ys, _ = solve_cde_from_windowed_logsigs(
            ts,
            logsigs,
            signature_window_size=int(self.signature_window_size),
            cde_func=self.cde_func,
            y0=h0,
            solver=self.solver,
            adjoint=self.adjoint,
            stepsize_controller=self.stepsize_controller,
            dt0=self.dt0,
        )
        return ys

    def _integration_steps_with_values(
        self,
        ts: jax.Array,
        control_values: jax.Array,
    ) -> jax.Array:
        x0 = control_values[0]
        h0 = self.initial(x0)

        logsigs = compute_windowed_logsignatures_from_values(
            control_values,
            self.shuffle_hopf_algebra,
            self.signature_depth,
            self.signature_window_size,
        )
        _, stats = solve_cde_from_windowed_logsigs(
            ts,
            logsigs,
            signature_window_size=int(self.signature_window_size),
            cde_func=self.cde_func,
            y0=h0,
            solver=self.solver,
            adjoint=self.adjoint,
            stepsize_controller=self.stepsize_controller,
            dt0=self.dt0,
        )
        return jnp.asarray(stats["num_steps"], dtype=jnp.float32)

    def _forward_with_control(
        self,
        ts: jax.Array,
        control: diffrax.AbstractPath,
    ) -> jax.Array:
        control_values = jax.vmap(control.evaluate)(ts)
        return self._forward_with_values(ts, control_values)

    def _apply_readout(self, hidden_states: jax.Array) -> jax.Array:
        """Apply readout to hidden states with manifold-aware output projection."""

        def apply_single(y: jax.Array) -> jax.Array:
            activation = self.readout_activation(self.readout_layer(y))
            return self._project_readout(activation)

        return jax.vmap(apply_single)(hidden_states)

    def __call__(
        self,
        control_values: jax.Array,
    ) -> jax.Array:
        """
        Forward pass.

        Parameters
        - control_values: shape (T, C). Control values.
        - control_or_coeffs: either a diffrax control path (e.g. CubicInterpolation)
          or a tuple of cubic coefficients compatible with diffrax.CubicInterpolation.

        Returns
        - If self.evolving_out is False: shape (out_size,)
        - If self.evolving_out is True: shape (T, out_size)
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
            hidden_over_time = self._forward_with_values(ts_aug, control_values_aug)
        else:
            ts_aug, control_values_aug = self._maybe_prepend_zero_basepoint(
                ts, control_values
            )
            hidden_over_time = self._forward_with_values(ts_aug, control_values_aug)

        if self.evolving_out:
            if self.prepend_zero_basepoint:
                hidden_over_time = hidden_over_time[1 : 1 + length]
            return self._apply_readout(hidden_over_time)
        else:
            return self._apply_readout(hidden_over_time[-1:])[0]

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
