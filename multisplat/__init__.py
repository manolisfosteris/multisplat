"""MultiSplat package.

Compatibility shim (torch >= 2.6):
    PyTorch 2.6 flipped ``torch.load``'s default to ``weights_only=True``, which
    refuses to unpickle the numpy scalars / objects stored in NeRFStudio 1.0.0
    checkpoints. NeRFStudio 1.0.0 predates that change and calls ``torch.load``
    without overriding the flag, so loading a splatfacto/multisplat checkpoint
    (rendering, or ``--load-checkpoint`` during editing) fails with
    ``UnpicklingError: Weights only load failed``.

    We restore the pre-2.6 behavior process-wide by defaulting ``weights_only``
    to ``False``. This runs on first import of any ``multisplat.*`` module, which
    happens before any checkpoint load for both ``ns-train multisplat`` (via
    ``multisplat.config``) and ``ns-multisplat-render`` (via ``multisplat.render``).
    Loading pickled checkpoints already implies trusting their source.
"""

import functools as _functools

import torch as _torch

if not getattr(_torch.load, "_multisplat_weights_only_shim", False):
    _original_torch_load = _torch.load

    @_functools.wraps(_original_torch_load)
    def _torch_load_weights_only_false(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)

    _torch_load_weights_only_false._multisplat_weights_only_shim = True
    _torch.load = _torch_load_weights_only_false
