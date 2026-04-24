from collections.abc import Callable
from typing import ClassVar, cast

import numpy as np
from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from diffrax._local_interpolation import ThirdOrderHermitePolynomialInterpolation
from diffrax._solution import RESULTS
from diffrax._solver.base import AbstractReversibleSolver
from diffrax._solver.runge_kutta import AbstractERK, ButcherTableau
from diffrax._term import AbstractTerm
from equinox.internal import ω
from jaxtyping import PyTree

ω = cast(Callable, ω)
_SolverState = Y


_ees25_tableau = ButcherTableau(
    a_lower=(np.array([1.0 / 3]), np.array([-5.0 / 48, 15.0 / 16])),
    b_sol=np.array([1.0 / 10, 1.0 / 2, 2.0 / 5]),
    b_error=np.array([1.0 / 10 + 1.0 / 3, 1.0 / 2 - 2.0 / 3, 2.0 / 5 - 2.0 / 3]),
    c=np.array([1.0 / 3, 5.0 / 6]),
)


class EES25(AbstractERK, AbstractReversibleSolver):
    """Explicit and Effectively Symmetric (EES) Runge-Kutta scheme of order 2
    and antisymmetric order 5, with parameter :math:`x = 1/10`.
    """

    tableau: ClassVar[ButcherTableau] = _ees25_tableau
    interpolation_cls: ClassVar[
        Callable[..., ThirdOrderHermitePolynomialInterpolation]
    ] = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        del terms
        return 2

    def antisymmetric_order(self, terms):
        del terms
        return 5

    def backward_step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y1: Y,
        args: Args,
        ts_state: PyTree[RealScalarLike],
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, DenseInfo, _SolverState, RESULTS]:
        del ts_state
        y0, _, dense_info, solver_state, result = self.step(
            terms, t1, t0, y1, args, solver_state, made_jump
        )
        return y0, dense_info, solver_state, result
