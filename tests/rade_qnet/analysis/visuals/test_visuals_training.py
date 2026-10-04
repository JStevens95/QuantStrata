"""
Tests for the training figures.

The gradient-norm figure is the one worth arguing for. A loss curve cannot
distinguish a converged network from a dead one: both go flat and stay flat.
The gradient norm separates them immediately -- a collapse towards zero is a
dead network, and unbounded growth is a run that is about to diverge rather
than one that already has.

Two behaviours here come from problems found while building the phase. An
absent metric produces a NaN rather than a zero, so a gap in the history
renders as a gap in the line instead of as a plunge to zero -- which in a
gradient-norm panel is a meaningful and alarming value. And a log-scaled axis
is only applied when there is something positive to plot, because
``set_yscale('log')`` over all-zero data warns and renders an empty panel.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.rade_qnet.analysis.visuals.training import (
    epoch_timing_figure,
    gradient_norm_figure,
    learning_rate_figure,
    training_diagnostics_figure,
)
from src.rade_qnet.core.contract.result import EpochRecord, FitOutcome
from src.rade_qnet.core.lifecycle.errors import ContractError


def outcome(
    *,
    n_epochs: int = 5,
    with_validation: bool = True,
    metrics_per_epoch: list[dict[str, float]] | None = None,
    learning_rates: list[float] | None = None,
) -> FitOutcome:
    """
    Build a fit outcome with a plausible history.

    Parameters
    ----------
    n_epochs
        How many epoch records to produce.
    with_validation
        Whether each record carries a validation loss.
    metrics_per_epoch
        Per-epoch metric mappings, defaulting to empty ones.
    learning_rates
        Per-epoch learning rates, defaulting to a constant.

    Returns
    -------
    FitOutcome
        The outcome.
    """
    metrics = metrics_per_epoch or [{} for _ in range(n_epochs)]
    rates = learning_rates or [0.01] * n_epochs
    history = tuple(
        EpochRecord(
            epoch=index,
            train_loss=1.0 / (index + 1),
            val_loss=1.2 / (index + 1) if with_validation else None,
            metrics=metrics[index],
            learning_rate=rates[index],
            seconds=0.5 + index * 0.1,
        )
        for index in range(n_epochs)
    )
    return FitOutcome(
        history=history,
        monitor="val_loss" if with_validation else "train_loss",
        best_epoch=n_epochs - 1,
        best_monitor_value=history[-1].val_loss or history[-1].train_loss,
        stopped_early=False,
        restored_best=True,
        total_seconds=sum(record.seconds for record in history),
    )


class TestGradientNormFigure:
    """The diagnostic a loss curve cannot provide."""

    def test_both_the_mean_and_the_maximum_are_drawn(self):
        """
        Because they fail differently.

        A rising maximum against a flat mean is a handful of exploding
        samples; a collapsing mean is a dead network. One line would conflate
        the two.
        """
        figure = gradient_norm_figure(
            outcome(
                n_epochs=3,
                metrics_per_epoch=[
                    {"grad_norm_mean": 1.0, "grad_norm_max": 3.0},
                    {"grad_norm_mean": 0.8, "grad_norm_max": 5.0},
                    {"grad_norm_mean": 0.6, "grad_norm_max": 9.0},
                ],
            )
        )
        assert len(figure.axes[0].lines) >= 2

    def test_the_axis_is_log_scaled_when_the_norms_are_positive(self):
        """
        Because the interesting changes are multiplicative.

        A norm going from 1 to 100 and one going from 0.01 to 1 are the same
        hundred-fold change, and on a linear axis the second is invisible.
        """
        figure = gradient_norm_figure(
            outcome(
                n_epochs=2,
                metrics_per_epoch=[
                    {"grad_norm_mean": 0.01, "grad_norm_max": 0.02},
                    {"grad_norm_mean": 1.0, "grad_norm_max": 100.0},
                ],
            )
        )
        assert figure.axes[0].get_yscale() == "log"

    def test_all_zero_norms_do_not_get_a_log_axis(self):
        """
        Because a log scale over zeros warns and renders an empty panel.

        And this is exactly the case worth looking at: an all-zero norm is a
        dead network, so the panel that would show it must not be the one that
        comes out blank.
        """
        figure = gradient_norm_figure(
            outcome(
                n_epochs=2,
                metrics_per_epoch=[
                    {"grad_norm_mean": 0.0, "grad_norm_max": 0.0},
                    {"grad_norm_mean": 0.0, "grad_norm_max": 0.0},
                ],
            )
        )
        assert figure.axes[0].get_yscale() == "linear"

    def test_an_absent_epoch_renders_as_a_gap(self):
        """
        Not as a plunge to zero.

        Zero is a meaningful and alarming value in this panel, so an epoch
        that simply did not record a norm must not look like one that recorded
        a collapse. A NaN leaves the line broken, which reads correctly.
        """
        figure = gradient_norm_figure(
            outcome(
                n_epochs=3,
                metrics_per_epoch=[
                    {"grad_norm_mean": 1.0, "grad_norm_max": 2.0},
                    {},
                    {"grad_norm_mean": 1.0, "grad_norm_max": 2.0},
                ],
            )
        )
        values = figure.axes[0].lines[0].get_ydata()
        assert np.isnan(values[1])

    def test_a_history_with_no_norms_at_all_is_refused(self):
        """
        Rather than drawing an empty panel.

        An empty figure in a report is read as "the run had no gradient
        problem", which is a much stronger claim than "nobody measured".
        """
        with pytest.raises(ContractError, match="no gradient norms"):
            gradient_norm_figure(outcome(n_epochs=3))

    def test_the_clipped_fraction_is_drawn_when_it_was_recorded(self):
        """
        Because a clip that binds on every step is not clipping outliers.

        It is rescaling every update, which caps the effective learning rate
        at something other than the configured one -- and nothing else in the
        output says so.
        """
        figure = gradient_norm_figure(
            outcome(
                n_epochs=2,
                metrics_per_epoch=[
                    {"grad_norm_mean": 1.0, "grad_norm_max": 2.0, "grad_clipped_fraction": 1.0},
                    {"grad_norm_mean": 1.0, "grad_norm_max": 2.0, "grad_clipped_fraction": 1.0},
                ],
            )
        )
        assert len(figure.axes) > 1 or len(figure.axes[0].lines) >= 3


class TestLearningRateFigure:
    """The schedule as it actually ran."""

    def test_the_recorded_rates_are_drawn(self):
        """
        Rather than the formula recomputed from the spec.

        A recomputed curve agrees with reality only while the formula and the
        scheduler agree, and the figure would then be showing a rate the
        optimiser never used.
        """
        figure = learning_rate_figure(outcome(n_epochs=3, learning_rates=[0.1, 0.05, 0.01]))
        assert figure.axes[0].lines[0].get_ydata().tolist() == [0.1, 0.05, 0.01]

    def test_an_unrecorded_rate_is_a_gap_rather_than_a_zero(self):
        """
        Because a flat line at zero would read as a frozen optimiser.

        That is a real failure mode, so the figure must not be able to
        fabricate its appearance from missing data. An epoch with no recorded
        rate leaves the line broken instead.
        """
        history = tuple(
            EpochRecord(epoch=index, train_loss=1.0, learning_rate=None, seconds=0.1)
            for index in range(3)
        )
        figure = learning_rate_figure(
            FitOutcome(
                history=history,
                monitor="train_loss",
                stopped_early=False,
                restored_best=False,
                total_seconds=0.3,
            )
        )
        assert np.all(np.isnan(figure.axes[0].lines[0].get_ydata()))

    def test_an_empty_history_is_refused(self):
        """
        Because there is no schedule to show.

        An empty panel in a report reads as a constant rate, which is a claim
        about the run rather than an absence of data.
        """
        with pytest.raises(ContractError):
            learning_rate_figure(
                FitOutcome(
                    history=(),
                    monitor="train_loss",
                    stopped_early=False,
                    restored_best=False,
                    total_seconds=0.0,
                )
            )


class TestEpochTimingFigure:
    """Where the time went, which a total cannot show."""

    def test_each_epoch_s_duration_is_drawn(self):
        """
        Because a run that slows down over time is leaking.

        A total divided by the epoch count would average the leak away
        completely.
        """
        figure = epoch_timing_figure(outcome(n_epochs=4))
        assert len(figure.axes[0].patches or figure.axes[0].lines[0].get_ydata()) == 4

    def test_an_empty_history_is_refused(self):
        """
        Rather than drawing an empty figure.

        An empty history means the fit produced nothing, which is worth an
        error at the point it is noticed rather than a blank panel in a report
        someone has to interpret.
        """
        with pytest.raises(ContractError, match="at least one epoch"):
            epoch_timing_figure(
                FitOutcome(
                    history=(),
                    monitor="train_loss",
                    stopped_early=False,
                    restored_best=False,
                    total_seconds=0.0,
                )
            )


class TestTrainingDiagnosticsFigure:
    """The three panels on a shared epoch axis."""

    def test_three_panels_are_drawn(self):
        """
        Loss, learning rate and gradient norm, which is the diagnostic set.

        Separate figures would have to be read side by side with their x-axes
        lined up by eye, and the whole value here is in the alignment.
        """
        figure = training_diagnostics_figure(
            outcome(
                n_epochs=3,
                metrics_per_epoch=[{"grad_norm_mean": 1.0, "grad_norm_max": 2.0}] * 3,
            )
        )
        assert len(figure.axes) == 3

    def test_the_epoch_axis_is_shared(self):
        """
        So a spike in the gradient norm lines up with the epoch it happened in.

        Without sharing, a schedule change and a loss plateau can appear one
        epoch apart purely from independent axis autoscaling.
        """
        figure = training_diagnostics_figure(
            outcome(
                n_epochs=3,
                metrics_per_epoch=[{"grad_norm_mean": 1.0, "grad_norm_max": 2.0}] * 3,
            )
        )
        limits = {axes.get_xlim() for axes in figure.axes}
        assert len(limits) == 1

    def test_the_figure_renders_without_gradient_norms(self):
        """
        Because tracking is optional and a report must not be load-bearing.

        The panel that cannot be filled is left empty or annotated rather than
        taking the whole figure down.
        """
        assert isinstance(training_diagnostics_figure(outcome(n_epochs=3)), Figure)

    def test_the_figure_renders_without_a_validation_split(self):
        """
        Which is a legitimate configuration, not an error.

        A run with no validation fraction still has a loss curve worth
        plotting, and the missing series must not be drawn as zeros.
        """
        assert isinstance(
            training_diagnostics_figure(outcome(n_epochs=3, with_validation=False)),
            Figure,
        )
