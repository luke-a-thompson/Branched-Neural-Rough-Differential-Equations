"""Compatibility import for the current roughrax/georax HEAD revisions."""

import os

import georax
import jax
import pysiglib
import pysiglib.jax_api


if not hasattr(georax, "post_lie_bracket"):
    from georax._geometry.base import post_lie_bracket

    georax.post_lie_bracket = post_lie_bracket

_prepare_log_sig = pysiglib.prepare_log_sig


def _prepare_log_sig_for_backend(*args, **kwargs):
    if os.environ.get("JAX_PLATFORMS") == "cpu" or jax.default_backend() == "cpu":
        kwargs["device"] = "cpu"
    return _prepare_log_sig(*args, **kwargs)


pysiglib.prepare_log_sig = _prepare_log_sig_for_backend
pysiglib.jax_api.prepare_log_sig = _prepare_log_sig_for_backend

import roughrax as roughrax  # noqa: E402


__all__ = ["roughrax"]
