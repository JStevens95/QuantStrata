"""
numba kernel warm-up

Called once in each worker process before real computation begins.
Prevents the first pricing call paying JIT compliation cost.
"""
from __future__ import annotations

import logging
import numpy as np

# define module level logging.
logger = logging.getLogger(__name__)


def warmup_kernels() -> None:
    """
    Call all @njit pricing kernels with dummy scalar inputs to trigger JIT compilation.

    Safe to call multiple times - numab caches compiled functions.

    :return:
    """
    try:
        from src.static_replication.pricing.kernels._math import (
            norm_cdf, norm_pdf, df_from_rate, linear_interp_1d, bilinear_interp_2d, compute_annuity,
            compute_forward_swap_rate
        )
        from src.static_replication.pricing.kernels.fx import (
            price_fx_vanilla_option, greeks_fx_vanilla_option, price_fx_digital_option, greeks_fx_digital_option,
            price_fx_quanto_option, greeks_fx_quanto_option
        )
        from src.static_replication.pricing.kernels.rates import (
            price_ir_swap, price_ir_swpation_bachelier, greeks_ir_swaption_bachelier
        )

        _t = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
        _r = np.array([0.04, 0.042, 0.045, 0.047, 0.05])

        # warmup math kernels.
        _ = norm_cdf(0.0)
        _ = norm_pdf(0.0)
        _ = df_from_rate(0.05, 1.0)
        _ = linear_interp_1d(np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5)
        _ = bilinear_interp_2d(
            np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.2, 0.2], [0.2, 0.2]), 0.5
        )
        _ = compute_annuity(_t, _r, 0.0, 1.0, 0.5)
        _ = compute_forward_swap_rate(_t, _r, 0.0, 1.0, 0.5)

        # warm up FX kernels.
        _ = price_fx_vanilla_option(1.0, 1.0, 0.25, 0.05, 0.03, 0.10, 1.0, True)
        _ = greeks_fx_vanilla_option(1.0, 1.0, 0.25, 0.05, 0.03, 0.10, 1.0, True)
        _ = price_fx_digital_option(1.0, 1.0, 0.25, 0.05, 0.03, 0.10, 1.0, True)
        _ = price_fx_quanto_option(1.0, 1.0, 0.25, 0.05, 0.03, 0.10, 1.0, 1.0, True)

        # warm up IR kernels.
        _ = price_ir_swap(_t, _r, 0.0, 5.0, 0.045, 0.5, 1.0, True)
        _ = price_ir_swpation_bachelier(_t, _r, 1.0, 5.0, 0.045, 0.005, 0.5, 1.0, True)
        _ = greeks_ir_swaption_bachelier(_t, _r, 1.0, 5.0, 0.045, 0.005, 0.5, 1.0, True)

        logger.debug("warmup_kernels: all kernels warmed up successfully")
    except Exception as exc:
        logger.warning("warmup_kernels: failed to warm up kernels: %s", exc)