import copy
import numbers
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from math import gcd

import torch
import torch.nn.functional as F
from philtorch.lti import cubic_spline
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

INTERPOLATION_MODES: tuple[str, ...] = ("linear", "nearest", "cubic")

interpolation_mode_var: ContextVar[str] = ContextVar(
    "interpolation_mode", default="linear"
)


@contextmanager
def interpolation_mode(mode: str):
    if mode not in INTERPOLATION_MODES:
        raise ValueError(
            f"mode must be one of {', '.join(INTERPOLATION_MODES)}, got {mode!r}"
        )
    token = interpolation_mode_var.set(mode)
    try:
        yield
    finally:
        interpolation_mode_var.reset(token)


def upsample(x: Tensor, hop_length: int, mode: str) -> Tensor:
    """Upsample the last dim of `x` by `hop_length` with an explicit `mode`."""
    match mode:
        case "nearest":
            return F.interpolate(
                x.reshape(-1, 1, x.size(-1)),
                scale_factor=hop_length,
                mode="nearest",
            ).view(*x.shape[:-1], -1)[..., hop_length // 2 :]
        case "linear":
            return F.interpolate(
                x.reshape(-1, 1, x.size(-1)),
                (x.size(-1) - 1) * hop_length + 1,
                mode="linear",
                align_corners=True,
            ).view(*x.shape[:-1], -1)
        case "cubic":
            return cubic_spline(x.flatten(0, -2), hop_length).view(*x.shape[:-1], -1)
        case _:
            raise ValueError(f"Unknown interpolation mode: {mode}")


def _require_time_axis(t, name: str) -> None:
    if t.hop_length < 0:
        raise ValueError(f"Cannot call {name} on an AudioTensor with hop_length < 0")


def check_hop_length(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        _require_time_axis(self, func.__name__)
        return func(self, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Helpers operating on plain tensors (no __torch_function__ re-entrance).
# Callers must hold `torch._C.DisableTorchFunctionSubclass()` or pass plain
# tensors.
# ---------------------------------------------------------------------------


def _upsample_time(x: Tensor, factor: int, mode: str) -> Tensor:
    """Upsample the time axis (dim 1) of a plain tensor by `factor`."""
    if factor == 1:
        return x
    # `upsample` works on the last dim: swap time to the last position
    if x.ndim > 2:
        x = x.transpose(1, -1)
    x = upsample(x, factor, mode)
    if x.ndim > 2:
        x = x.transpose(1, -1)
    return x


def _unsqueeze_trailing(x: Tensor, ndim: int) -> Tensor:
    n = ndim - x.ndim
    if n <= 0:
        return x
    return x[(...,) + (None,) * n]


def _narrow_time(x: Tensor, steps: int) -> Tensor:
    """Keep the first `steps` frames of the time axis (dim 1) of a plain tensor."""
    return x if x.shape[1] <= steps else x.narrow(1, 0, steps)


def _wrap(t: Tensor, hop_length: int, cls) -> "AudioTensor":
    """Wrap a plain tensor as `cls` (keeps autograd graph / storage)."""
    out = t.as_subclass(cls)
    out.hop_length = hop_length
    return out


def _rewrap(t: "AudioTensor", fn, hop_length: int) -> "AudioTensor":
    """Apply `fn` to the plain view of `t` and wrap the result with `hop_length`."""
    with torch._C.DisableTorchFunctionSubclass():
        data = fn(t.as_subclass(Tensor))
    return _wrap(data, hop_length, type(t))


# ---------------------------------------------------------------------------
# Helpers for __torch_function__
# ---------------------------------------------------------------------------

_INPLACE_DUNDERS = frozenset(
    {
        "__setitem__",
        "__iadd__",
        "__isub__",
        "__imul__",
        "__itruediv__",
        "__ifloordiv__",
        "__imod__",
        "__ipow__",
        "__iand__",
        "__ior__",
        "__ixor__",
        "__ilshift__",
        "__irshift__",
        "__imatmul__",
    }
)


def _is_inplace(name: str, kwargs: dict) -> bool:
    return (
        name in _INPLACE_DUNDERS
        or (name.endswith("_") and not name.startswith("__"))
        or "out" in kwargs
    )


def _arg(args, kwargs, pos: int, *names, default=None):
    if len(args) > pos:
        return args[pos]
    for n in names:
        if n in kwargs:
            return kwargs[n]
    return default


def _reconcile_inplace(
    name, args, kwargs, flatten, idx, hops, mode
) -> tuple[dict[int, Tensor], int]:
    """Broadcast the operands of an in-place call to its target (never the reverse).

    Pure: returns `({index in flatten: replacement}, output hop)` without
    mutating its arguments. Runs under `DisableTorchFunctionSubclass`.
    """
    if "out" in kwargs:
        out = kwargs["out"]
        targets = list(out) if isinstance(out, (tuple, list)) else [out]
    else:
        targets = args[:1]
    tgt_idx = [i for i in idx if any(flatten[i] is t for t in targets)]
    op_idx = [i for i in idx if i not in tgt_idx]
    output_hop = hops[0]
    repl = {}
    if len(set(hops)) > 1:
        tgt_hops = {flatten[i].hop_length for i in tgt_idx}
        if name == "__setitem__" or len(tgt_hops) != 1:
            raise ValueError(
                f"In-place operation `{name}` on AudioTensors with "
                f"different hop_length ({sorted(set(hops))}) is not "
                "supported; resample them to a common hop_length "
                "first (e.g. with set_hop_length)."
            )
        output_hop = tgt_hops.pop()
        if gcd(*hops) != output_hop:
            raise ValueError(
                f"In-place operation `{name}`: the destination has "
                f"hop_length {output_hop} but an operand has a finer "
                f"hop_length (gcd {gcd(*hops)}). The destination "
                "cannot be resampled in place; call set_hop_length "
                "on it first (out of place)."
            )
        # operands longer than the target are truncated; shorter
        # ones are left to torch's own shape error
        steps = flatten[tgt_idx[0]].shape[1]
        for i in op_idx:
            t = flatten[i]
            factor = t.hop_length // output_hop
            repl[i] = _narrow_time(
                _upsample_time(t.as_subclass(Tensor), factor, mode), steps
            )
    if name != "__setitem__":
        # align operand ndim (a lower-ndim target is torch's error)
        max_ndim = max(repl.get(i, flatten[i]).ndim for i in idx)
        for i in op_idx:
            repl[i] = _unsqueeze_trailing(repl.get(i, flatten[i]), max_ndim)
    return repl, output_hop


def _reconcile_resampled(flatten, idx, hops, mode) -> tuple[dict[int, Tensor], int]:
    """Resample, ndim-align and truncate all AudioTensors of an out-of-place call.

    Pure: returns `({index in flatten: replacement}, output (gcd) hop)` without
    mutating its arguments. Runs under `DisableTorchFunctionSubclass`.
    """
    hop_gcd = gcd(*hops)
    plains = [
        _upsample_time(flatten[i].as_subclass(Tensor), h // hop_gcd, mode)
        for i, h in zip(idx, hops)
    ]
    max_ndim = max(p.ndim for p in plains)
    plains = [_unsqueeze_trailing(p, max_ndim) for p in plains]
    steps = min(p.shape[1] for p in plains)
    return {i: _narrow_time(p, steps) for i, p in zip(idx, plains)}, hop_gcd


class AudioTensor(Tensor):
    def __new__(cls, data, hop_length: int = 1, *, requires_grad: bool | None = None):
        if isinstance(hop_length, bool) or not isinstance(hop_length, numbers.Integral):
            raise ValueError(  # noqa: TRY004
                f"hop_length must be an int, got {hop_length!r}"
            )
        hop_length = int(hop_length)
        with torch._C.DisableTorchFunctionSubclass():
            t = torch.as_tensor(data)
            if hop_length != -1 and (hop_length < 1 or t.ndim < 2):
                raise ValueError(
                    "hop_length must be -1 (no time axis) or >= 1 with ndim >= 2; "
                    f"got hop_length={hop_length} for a tensor with ndim={t.ndim}"
                )
            if requires_grad is None:
                self = t.as_subclass(cls)
            else:
                self = Tensor._make_subclass(cls, t, requires_grad)
        self.hop_length = hop_length
        return self

    def __repr__(self):
        return f"Hop-length: {self.hop_length}\n" + super().__repr__()

    def __deepcopy__(self, memo):
        # torch's default Tensor.__deepcopy__ fails for subclasses (`new_empty`
        # returns a plain Tensor under DisableTorchFunctionSubclass), so deep
        # copy the plain tensor (storage, conj/neg bits, memo) and re-wrap it.
        with torch._C.DisableTorchFunctionSubclass():
            if not self.is_leaf:
                raise RuntimeError(
                    "Only Tensors created explicitly by the user "
                    "(graph leaves) support the deepcopy protocol at the moment."
                )
            plain = copy.deepcopy(self.detach(), memo)
            new = Tensor._make_subclass(type(self), plain, self.requires_grad)
            memo[id(self)] = new
            if self.grad is not None:
                new.grad = copy.deepcopy(self.grad, memo)
            new.__dict__ = copy.deepcopy(self.__dict__, memo)
            return new

    @check_hop_length
    def set_hop_length(self, hop_length: int):
        if hop_length < 1:
            raise ValueError(f"hop_length must be >= 1, got {hop_length}")
        if hop_length > self.hop_length:
            if hop_length % self.hop_length:
                raise ValueError(
                    f"hop_length {hop_length} is not a multiple of the current "
                    f"hop_length {self.hop_length}"
                )
            return self.increase_hop_length(hop_length // self.hop_length)
        elif hop_length < self.hop_length:
            if self.hop_length % hop_length:
                raise ValueError(
                    f"hop_length {hop_length} does not divide the current "
                    f"hop_length {self.hop_length}"
                )
            return self.reduce_hop_length(self.hop_length // hop_length)
        return self

    @check_hop_length
    def increase_hop_length(self, factor: int):
        if factor < 1:
            raise ValueError(f"factor must be positive, got {factor}")
        if factor == 1:
            return self
        return _rewrap(self, lambda x: x[:, ::factor].clone(), self.hop_length * factor)

    @check_hop_length
    def reduce_hop_length(self, factor: int | None = None):
        if factor is None:
            factor = self.hop_length
        elif factor < 1 or self.hop_length % factor:
            raise ValueError(
                f"factor {factor} must be a positive divisor of the current "
                f"hop_length {self.hop_length}"
            )
        if factor == 1:
            return self
        mode = interpolation_mode_var.get()
        return _rewrap(
            self, lambda x: _upsample_time(x, factor, mode), self.hop_length // factor
        )

    def as_tensor(self):
        with torch._C.DisableTorchFunctionSubclass():
            return self.as_subclass(Tensor)

    @property
    @check_hop_length
    def steps(self):
        with torch._C.DisableTorchFunctionSubclass():
            return self.shape[1]

    @check_hop_length
    def truncate(self, steps: int):
        if steps >= self.steps:
            return self
        return _rewrap(self, lambda x: _narrow_time(x, steps), self.hop_length)

    def unfold(self, *args, **kwargs):
        """`unfold(dimension, size, step)` as in torch, or the shorthand
        `unfold(size, step=1)` which always unfolds the time axis (dim 1)."""
        if "dimension" in kwargs or len(args) + len(kwargs) == 3:
            return Tensor.unfold(self, *args, **kwargs)
        _require_time_axis(self, "unfold")
        size = _arg(args, kwargs, 0, "size")
        step = _arg(args, kwargs, 1, "step", default=1)
        return Tensor.unfold(self, 1, size, step)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        name = getattr(func, "__name__", "")
        inplace = _is_inplace(name, kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            flatten, spec = tree_flatten((args, kwargs))
            # objects that exist before the call: never relabel them
            inputs = {id(t) for t in flatten if isinstance(t, cls)}
            idx = [
                i
                for i, t in enumerate(flatten)
                if isinstance(t, cls) and t.hop_length > 0
            ]
            if not idx:
                output_hop = -1
            elif len(idx) == 1:
                output_hop = flatten[idx[0]].hop_length
            else:
                mode = interpolation_mode_var.get()
                hops = [flatten[i].hop_length for i in idx]
                if inplace:
                    repl, output_hop = _reconcile_inplace(
                        name, args, kwargs, flatten, idx, hops, mode
                    )
                else:
                    repl, output_hop = _reconcile_resampled(flatten, idx, hops, mode)
                flatten = [repl.get(i, t) for i, t in enumerate(flatten)]
                args, kwargs = tree_unflatten(flatten, spec)

        ret = super().__torch_function__(func, types, args, kwargs)

        def post_process(t):
            if not isinstance(t, cls) or id(t) in inputs:
                return t
            hop = output_hop
            if t.ndim < 2:
                # the only invalidation rule: no time axis left
                hop = -1
            elif (
                hop > 0
                and func is Tensor.unfold
                # time stays dim 1; unfolding it with `step` coarsens the hop
                and _arg(args, kwargs, 1, "dimension") % args[0].ndim == 1
            ):
                hop *= _arg(args, kwargs, 3, "step")
            t.hop_length = hop
            return t

        with torch._C.DisableTorchFunctionSubclass():
            if isinstance(ret, Tensor):
                return post_process(ret)
            if ret is None or isinstance(ret, (bool, int, float, str, torch.Size)):
                return ret
            return tree_map(post_process, ret)


# allow `torch.load(..., weights_only=True)` (the default) to restore AudioTensors
torch.serialization.add_safe_globals([AudioTensor])
