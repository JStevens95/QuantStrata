"""
How a single job uses its machine.

:class:`HardwareSpec` answers *how does this one job use the hardware it has* --
device, precision, compilation, distribution, determinism. It deliberately
does **not** answer *where do jobs run*, which is placement and belongs to the
job-set spec.

Keeping them apart matters because they are configured by different people for
different reasons. "Use bfloat16" is a modelling decision made once. "Run
forty of these across four GPUs" is an operational decision made per run.
Conflating them is why "use the GPU" and "run forty in parallel" so often end
up as the same overloaded flag.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..lifecycle.errors import SpecError
from ..provenance.seeding import Determinism
from .base import Spec

__all__ = ["HardwareSpec"]


class HardwareSpec(Spec):
    """
    Device, precision and determinism settings for one job.

    Parameters
    ----------
    device
        ``auto`` selects the best available accelerator and falls back to the
        CPU. Naming a device explicitly is a request, not a guarantee: an
        engine that cannot honour it warns and degrades rather than failing a
        run that would otherwise have succeeded.
    device_index
        Which device of the chosen type to use. ``None`` means the engine
        chooses, which is the correct setting under a job set -- the executor
        has already pinned device visibility per worker, so a job that also
        picks an index would fight it.
    precision
        Compute precision. Reduced precision is a throughput choice with
        numerical consequences, so it is explicit rather than inferred from
        the device.
    compile_model
        Whether to apply the engine's graph compiler. Off by default because
        compilation perturbs floating-point results, and a refactor being
        verified against a golden fixture needs that off.
    distributed
        Distribution strategy for a single job across several devices.
    determinism
        How hard to work for bit-for-bit reproducibility. See
        :mod:`rade_qnet.core.provenance.seeding` for what each level costs.
    threads_per_worker
        Intra-op thread budget. ``None`` leaves the library's default, which
        is correct for a single job and wrong under a process pool -- there
        the executor sets it, because N workers each claiming every core
        collapses throughput.
    """

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    device_index: int | None = Field(default=None, ge=0)
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    compile_model: bool = False
    distributed: Literal["none", "ddp"] = "none"
    determinism: Determinism = "off"
    threads_per_worker: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _reject_impossible_combinations(self) -> HardwareSpec:
        """
        Reject settings that cannot be honoured on the requested device.

        Checked here rather than in the engine so the failure arrives before
        any data is read. A four-hour job that dies on an unsupported
        precision setting in its first optimiser step has wasted the data
        build for nothing.

        Returns
        -------
        HardwareSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if the combination cannot work.
        """
        # float16 on a CPU is not merely slow: most CPU kernels have no fp16
        # path, so the result is either an error deep in the engine or a
        # silent upcast that delivers none of the expected speed-up. bfloat16
        # is genuinely supported on modern CPUs and so is permitted.
        if self.precision == "fp16" and self.device == "cpu":
            raise SpecError(
                "precision='fp16' is not supported on device='cpu'; "
                "use 'bf16' for reduced precision on a CPU, or 'fp32'"
            )

        # Pinning an index while also asking for automatic device selection is
        # contradictory, and the two settings would be resolved by different
        # layers -- which is exactly how a job ends up on an unexpected device.
        if self.device == "auto" and self.device_index is not None:
            raise SpecError(
                "device_index cannot be set when device='auto'; "
                "name the device explicitly, or leave device_index unset"
            )
        return self
