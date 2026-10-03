"""
Tests for weight serialisation -- defect 10.

Defect 10 is that checkpoints were loaded with ``weights_only=False``, which
makes ``torch.load`` run the pickle machinery and therefore execute arbitrary
code from the file. A model artifact is a *data* file: it travels between
machines, sits in object storage, and is fetched by a serving process. Giving
it the privileges of a Python script is a straightforward remote code
execution path, and it is enabled by a default rather than by a decision.

Loading with ``weights_only=True`` closes it, and the consequence is that a
checkpoint containing pickled *modules* -- rather than a plain state
dictionary -- can no longer be read. That is the correct trade: such a
checkpoint cannot be loaded safely at all, and the fix is to regenerate it
from the bundle rather than to re-enable the unsafe path.

The second group of tests here is about the snapshot being a *copy*. A state
dictionary is a dictionary of views onto the live parameters, so a snapshot
taken by reference is not a snapshot: the next optimiser step rewrites it in
place, and "restore the best epoch" restores the last epoch instead. Nothing
about that is visible in the training history, which still reports the best
epoch correctly.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.rade_xl.core.runtime.errors import EngineError
from src.rade_xl.engines.base import ModelHandle
from src.rade_xl.engines.torch.checkpoint import (
    load_weights,
    restore_state_dict,
    save_weights,
    snapshot_state_dict,
)


def make_model(seed: int = 0) -> nn.Module:
    """
    Build a small deterministic model.

    Parameters
    ----------
    seed
        Seed for the initial weights, so two models can be made to differ.

    Returns
    -------
    torch.nn.Module
        The model.
    """
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


class TestSnapshotIsACopy:
    """A snapshot by reference is not a snapshot."""

    def test_a_snapshot_survives_a_parameter_update(self):
        """
        The defect that makes "restore the best epoch" restore the last one.

        ``state_dict`` returns views onto the live tensors, so a snapshot
        taken without detaching and copying is rewritten in place by the next
        optimiser step. The history still names the right best epoch, so
        nothing reports the problem.
        """
        model = make_model()
        snapshot = snapshot_state_dict(model)
        before = snapshot["0.weight"].clone()

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)

        assert torch.equal(snapshot["0.weight"], before)

    def test_a_snapshot_is_on_the_cpu(self):
        """
        So a device-resident snapshot does not hold accelerator memory.

        A tuning sweep keeping one snapshot per trial on the GPU runs out of
        memory at a trial count that looks arbitrary.
        """
        model = make_model()
        snapshot = snapshot_state_dict(model)
        assert all(tensor.device.type == "cpu" for tensor in snapshot.values())

    def test_a_snapshot_carries_no_gradients(self):
        """
        It is weights, not a point in an optimisation.

        A snapshot holding its gradient graph keeps the whole forward pass's
        activations alive, which turns one retained snapshot into a memory
        leak proportional to the batch size.
        """
        model = make_model()
        inputs = torch.ones(2, 4)
        model(inputs).sum().backward()
        snapshot = snapshot_state_dict(model)
        assert all(not tensor.requires_grad for tensor in snapshot.values())


class TestRestore:
    """Putting a snapshot back, exactly."""

    def test_a_restored_model_predicts_identically(self):
        """
        Which is the property early stopping depends on.

        Restoring approximately would mean the model that gets saved is not
        the model whose validation score chose it.
        """
        model = make_model()
        inputs = torch.ones(3, 4)
        expected = model(inputs)

        snapshot = snapshot_state_dict(model)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.5)
        restore_state_dict(model, snapshot)

        assert torch.equal(model(inputs), expected)

    def test_a_mismatched_snapshot_is_refused(self):
        """
        Strictly, so a partial restore cannot happen quietly.

        A non-strict load silently leaves any unmatched parameter at its
        current value, which produces a model that is part best-epoch and part
        last-epoch -- and scores somewhere between the two.
        """
        wide = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
        with pytest.raises(Exception, match=r"size mismatch|shape|Error"):
            restore_state_dict(wide, snapshot_state_dict(make_model()))


class TestFileRoundTrip:
    """Save, reload into a fresh model, predict the same."""

    def test_weights_round_trip_bit_for_bit(self, tmp_path):
        """
        Not approximately: a served model and a scored model are the same model.

        A round trip that lost precision would make the difference show up as
        a discrepancy between a backtest and live trading, which is the last
        place anyone looks for a serialisation bug.
        """
        model = make_model(seed=1)
        inputs = torch.ones(3, 4)
        expected = model(inputs)

        path = tmp_path / "weights.pt"
        save_weights(ModelHandle(model=model, unwrapped=model), path)

        fresh = make_model(seed=2)
        assert not torch.equal(fresh(inputs), expected)
        load_weights(fresh, path)
        assert torch.equal(fresh(inputs), expected)

    def test_a_wrapper_prefix_is_stripped(self, tmp_path):
        """
        So a bundle written from a distributed run reopens without one.

        ``DistributedDataParallel`` prefixes every key with ``module.`` and
        ``torch.compile`` with ``_orig_mod.``. A bundle carrying those keys
        can only be loaded by reconstructing the same wrapper -- which the
        reader of the bundle has no way to know about.
        """
        model = make_model()
        wrapped = nn.Sequential(model)  # Stands in for a wrapper's nesting.
        path = tmp_path / "weights.pt"
        save_weights(ModelHandle(model=wrapped, unwrapped=model), path)

        fresh = make_model(seed=3)
        load_weights(fresh, path)
        assert torch.equal(fresh(torch.ones(2, 4)), model(torch.ones(2, 4)))

    def test_a_manifest_is_written_beside_the_weights(self, tmp_path):
        """
        So a weights file can be identified without loading it.

        Which matters because loading it is the operation the format exists to
        make safe to refuse.
        """
        model = make_model()
        path = tmp_path / "weights.pt"
        save_weights(ModelHandle(model=model, unwrapped=model), path)
        assert (tmp_path / "weights.json").exists()


class TestDefectTenPickledCheckpointsAreRefused:
    """``weights_only=True``, and a message that says what to do instead."""

    def test_a_checkpoint_containing_a_module_is_refused(self, tmp_path):
        """
        The remote code execution path, closed.

        A pickled module in a checkpoint means loading the file executes code
        from it. A model artifact travels between machines and is fetched by a
        serving process, so that is not a theoretical concern.
        """
        path = tmp_path / "pickled.pt"
        torch.save(make_model(), path)
        with pytest.raises(EngineError):
            load_weights(make_model(), path)

    def test_the_message_points_at_regeneration(self, tmp_path):
        """
        Rather than at the flag that would re-open the hole.

        Torch's own error helpfully suggests ``weights_only=False``, which is
        precisely the thing not to do -- so the message is replaced rather
        than passed through.
        """
        path = tmp_path / "pickled.pt"
        torch.save(make_model(), path)
        with pytest.raises(EngineError) as caught:
            load_weights(make_model(), path)
        assert "weights_only=False" not in str(caught.value)
        assert "regenerat" in str(caught.value).lower()
