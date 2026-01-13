from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from src.marketdata.surfaces.fx.quotes import FxSmileSliceQuotes


def plot_fx_smile_slice_nodes(
    slice_quotes: FxSmileSliceQuotes,
    *,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot FX smile slice in delta-space as node vols: put(Δ), ATM, call(Δ).
    """
    ds = slice_quotes.deltas()
    ds = [float(d) for d in ds]

    put_vols = [float(slice_quotes.vol_put(d)) for d in ds]
    call_vols = [float(slice_quotes.vol_call(d)) for d in ds]
    atm = float(slice_quotes.atm_vol)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(ds, put_vols, marker="o", linestyle="-", label="put wing σ_put(Δ)")
    ax.plot(ds, call_vols, marker="o", linestyle="-", label="call wing σ_call(Δ)")
    ax.axhline(atm, linestyle="--", label="ATM")

    ax.set_title(title or f"FX Smile Slice Nodes @ T={float(slice_quotes.expiry)}")
    ax.set_xlabel("|Δ|")
    ax.set_ylabel("Implied Vol")
    ax.grid(True)
    ax.legend()
    return fig


def plot_fx_smile_roundtrip_errors(
    orig: FxSmileSliceQuotes,
    recon: FxSmileSliceQuotes,
    *,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot RR/BF errors (recon - orig) per delta for one expiry.
    """
    ds = sorted(set(orig.deltas()) | set(recon.deltas()))
    ds = [float(d) for d in ds]

    rr_err = []
    bf_err = []
    for d in ds:
        rr0 = float(orig.rr_by_delta.get(d, 0.0))
        bf0 = float(orig.bf_by_delta.get(d, 0.0))
        rr1 = float(recon.rr_by_delta.get(d, 0.0))
        bf1 = float(recon.bf_by_delta.get(d, 0.0))
        rr_err.append(rr1 - rr0)
        bf_err.append(bf1 - bf0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ds, rr_err, marker="o", label="RR error")
    ax.plot(ds, bf_err, marker="o", label="BF error")
    ax.axhline(0.0, linestyle="--")
    ax.set_title(title or f"Smile Roundtrip Errors @ T={float(orig.expiry)}")
    ax.set_xlabel("|Δ|")
    ax.set_ylabel("error (recon - orig)")
    ax.grid(True)
    ax.legend()
    return fig