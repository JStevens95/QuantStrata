from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote


def plot_quotes(quotes: Mapping[MarketId, Quote], title: str = "Quotes") -> plt.Figure:
    labels = [mid.key() for mid in quotes.keys()]
    values = [float(q.value) for q in quotes.values()]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(range(len(values)), values)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, axis="y")
    fig.tight_layout()
    return fig