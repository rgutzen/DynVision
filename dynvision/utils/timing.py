"""Validation of biological durations against the integration step ``dt``.

DynVision expresses propagation delays in milliseconds (``t_feedforward``,
``t_recurrence``, ``t_feedback``, ``t_skip``) and the loss reaction window in
milliseconds (``loss_reaction_time``). Models integrate those durations on a
fixed grid of ``dt`` millisecond steps, so every one of them has to be an
integer number of steps: the delay buffers and offset indices that implement
them are integer-indexed.

A duration that is not an exact multiple cannot be represented on that grid. It
used to be converted silently, and the conversion sites disagreed with each
other -- ``int(value / dt)`` truncated toward zero for the feedforward and
recurrent delays while ``math.ceil`` rounded up for the reaction time, so
``dt=3, t_recurrence=4, loss_reaction_time=4`` ran with a 3 ms recurrence and a
6 ms reaction window. These functions are the single authority for the
constraint: :func:`is_integer_multiple` decides it, :func:`to_timesteps`
converts a validated duration, and :func:`multiple_error_message` explains a
violation.
"""

import math
from typing import Final

# Tolerance for the multiple test. Durations are milliseconds in the 0.1-100
# range, so 1e-9 sits far above float64 division error (~1e-16 relative) and far
# below any fraction of a step that carries biological meaning.
_RTOL: Final[float] = 1e-9
_ATOL: Final[float] = 1e-9


def is_integer_multiple(
    value: float,
    dt: float,
    *,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> bool:
    """Return True when ``value`` is a whole number of ``dt`` steps.

    The test is tolerant because it is applied to user-supplied floats.
    ``0.3 % 0.1`` evaluates to ``0.09999999999999998`` in binary floating point,
    so an exact remainder test would reject the valid configuration
    ``dt=0.1, t_recurrence=0.3``. Comparing ``value / dt`` against its nearest
    integer avoids that without accepting a materially different duration.

    Args:
        value: Duration in milliseconds. Must be non-negative.
        dt: Integration step in milliseconds. Must be positive.
        rtol: Relative tolerance passed to :func:`math.isclose`.
        atol: Absolute tolerance passed to :func:`math.isclose`.

    Returns:
        True if ``value`` is an integer number of ``dt`` steps.

    Raises:
        ValueError: If ``dt`` is not positive, or ``value`` is negative.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive to test for multiples, got dt={dt:g}")
    if value < 0:
        raise ValueError(f"duration must be non-negative, got value={value:g}")
    ratio = value / dt
    return math.isclose(ratio, round(ratio), rel_tol=rtol, abs_tol=atol)


def to_timesteps(value: float, dt: float) -> int:
    """Convert a millisecond duration to a whole number of ``dt`` steps.

    Callers should have accepted ``value`` via :func:`is_integer_multiple`
    first, in which case the rounding is exact. Rounding to nearest (rather than
    truncating toward zero, as a bare ``int(value / dt)`` does) keeps the
    conversion correct for entry points that construct model classes directly
    and so bypass parameter validation.

    Args:
        value: Duration in milliseconds.
        dt: Integration step in milliseconds. Must be positive.

    Returns:
        The duration expressed in integration steps.

    Raises:
        ValueError: If ``dt`` is not positive, or ``value`` is negative.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive to convert to timesteps, got dt={dt:g}")
    if value < 0:
        raise ValueError(f"duration must be non-negative, got value={value:g}")
    return int(round(value / dt))


def multiple_error_message(name: str, value: float, dt: float) -> str:
    """Build the error message for a duration that is not a multiple of ``dt``.

    The message names the offending parameter, reports the ratio that could not
    be represented, and gives both neighbouring valid values so the caller can
    pick one without recomputing them.

    Args:
        name: Parameter name as it appears in the configuration, e.g.
            ``"t_recurrence"``.
        value: The offending duration in milliseconds.
        dt: Integration step in milliseconds.

    Returns:
        A message suitable for raising as a ``ValueError``.
    """
    ratio = value / dt
    lower_steps = math.floor(ratio)
    upper_steps = math.ceil(ratio)
    return (
        f"{name}={value:g} ms is not an integer multiple of dt={dt:g} ms "
        f"({value:g}/{dt:g} = {ratio:.6g} timesteps). Durations are integrated "
        f"on the dt grid, so they must be whole numbers of steps. "
        f"Use {lower_steps * dt:g} ms ({lower_steps} step(s)) or "
        f"{upper_steps * dt:g} ms ({upper_steps} step(s))."
    )
