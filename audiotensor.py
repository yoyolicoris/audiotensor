import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
import numpy as np
from torch import Tensor
from typing import Union, Iterable
from functools import reduce
from math import gcd
from contextvars import ContextVar
from contextlib import contextmanager
from philtorch.lti import cubic_spline

INTERPOLATION_MODES: Iterable[str] = ["linear", "nearest", "cubic"]

interpolation_mode_var: ContextVar[str] = ContextVar(
    "interpolation_mode", default="linear"
)


@contextmanager
def interpolation_mode(mode: str):
    assert mode in INTERPOLATION_MODES, f"mode must be one of {INTERPOLATION_MODES}"
    token = interpolation_mode_var.set(mode)
    try:
        yield
    finally:
        interpolation_mode_var.reset(token)


def upsample(x: Tensor, hop_length: int) -> Tensor:
    mode = interpolation_mode_var.get()
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


def check_hop_length(func):
    def wrapper(self, *args, **kwargs):
        if self.hop_length < 0:
            raise ValueError(
                "Cannot call {} on an AudioTensor with hop_length < 0".format(
                    func.__name__
                )
            )
        return func(self, *args, **kwargs)

    return wrapper


class AudioTensor(Tensor):
    def __new__(
        cls,
        x,
        *args,
        requires_grad=None,
        **kwargs,
    ):
        if requires_grad is None:
            return super().__new__(cls, x)
        return cls._make_subclass(cls, x, requires_grad)

    def __init__(
        self,
        data: Union[Tensor, np.ndarray],
        hop_length: int = 1,
    ):
        super().__init__()
        assert data.ndim > 1, "AudioTensor must have at least 2 dimensions"
        self.hop_length = hop_length

    def __repr__(self):
        return f"Hop-length: {self.hop_length}\n" + super().__repr__()

    @check_hop_length
    def set_hop_length(self, hop_length: int):
        if hop_length > self.hop_length:
            assert hop_length % self.hop_length == 0
            return self.increase_hop_length(hop_length // self.hop_length)
        elif hop_length < self.hop_length:
            assert self.hop_length % hop_length == 0
            return self.reduce_hop_length(self.hop_length // hop_length)
        return self

    @check_hop_length
    def increase_hop_length(self, factor: int):
        assert factor > 0, "factor must be positive"
        if factor == 1:
            return self

        data = self[:, ::factor].clone()
        data.hop_length = self.hop_length * factor
        return data

    @check_hop_length
    def reduce_hop_length(self, factor: int = None):
        if factor is None:
            factor = self.hop_length
        else:
            assert self.hop_length % factor == 0 and factor <= self.hop_length

        if factor == 1:
            return self

        self_copy = self.clone()
        # swap the time dimension to the last
        if self.ndim > 2:
            self_copy = self_copy.transpose(1, -1)
        expand_self_copy = upsample(self_copy, factor)

        # swap the time dimension back
        if self.ndim > 2:
            expand_self_copy = expand_self_copy.transpose(1, -1)

        expand_self_copy.hop_length = self.hop_length // factor

        return expand_self_copy

    def as_tensor(self):
        return torch.Tensor(self)

    @property
    @check_hop_length
    def steps(self):
        if self.ndim < 2:
            return torch.iinfo(torch.int32).max
        return self.size(1)

    @check_hop_length
    def truncate(self, steps: int):
        if steps >= self.steps or self.ndim < 2:
            return self
        return self.narrow(1, 0, steps)

    def new_tensor(self, data: Tensor):
        return AudioTensor(data, hop_length=self.hop_length)

    @check_hop_length
    def unfold(self, size: int, step: int = 1):
        assert self.ndim == 2
        data = super().unfold(1, size, step)
        data.hop_length = self.hop_length * step
        return data

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        flatten, spec = tree_flatten((args, kwargs))
        mask = tuple(
            map(lambda t: isinstance(t, AudioTensor) and t.hop_length > 0, flatten)
        )

        if sum(mask) > 1:
            audio_tensors, others = reduce(
                lambda x, y: (x[0] + (y[1],), x[1]) if y[0] else (x[0], x[1] + (y[1],)),
                zip(mask, flatten),
                ((), ()),
            )

            audio_tensors = cls.broadcasting(*audio_tensors)
            min_steps = min(a.steps for a in audio_tensors)
            audio_tensors = tuple(a.truncate(min_steps) for a in audio_tensors)
            flatten, *_ = reduce(
                lambda x, is_audio: (
                    (x[0] + x[1][:1], x[1][1:], x[2])
                    if is_audio
                    else (x[0] + x[2][:1], x[1], x[2][1:])
                ),
                mask,
                ((), audio_tensors, others),
            )

        def get_output_hop(cur, xs):
            if len(xs) == 0 or cur > 0:
                return cur
            x, *xs = xs
            if isinstance(x, AudioTensor):
                return get_output_hop(max(cur, x.hop_length), xs)
            else:
                return get_output_hop(cur, xs)

        output_hop = get_output_hop(-1, flatten)
        broadcasted = tree_unflatten(flatten, spec)

        def post_process(t):
            if isinstance(t, cls):
                t.hop_length = output_hop
                if t.ndim == 1:
                    t.hop_length = -1
            return t

        return tree_map(
            post_process, super().__torch_function__(func, types, *broadcasted)
        )

    @classmethod
    def broadcasting(cls, *tensors):
        assert len(tensors) > 0
        # check hop lengths are divisible by each other
        hop_lengths = tuple(t.hop_length for t in tensors)
        hop_length_gcd = gcd(*hop_lengths)
        ret = tuple(
            t.reduce_hop_length(t.hop_length // hop_length_gcd) for t in tensors
        )
        max_ndim = max(t.ndim for t in ret)
        ret = tuple(
            (
                reduce(lambda x, _: x.unsqueeze(-1), [None] * (max_ndim - t.ndim), t)
                if t.ndim < max_ndim
                else t
            )
            for t in ret
        )
        return ret
