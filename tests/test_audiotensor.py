import copy
import io
import pickle
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from philtorch.lti import cubic_spline

from audiotensor import AudioTensor, interpolation_mode
from audiotensor.audiotensor import interpolation_mode_var, upsample


# ---------------------------------------------------------------- helpers
@contextmanager
def count_torch_function():
    calls = []
    orig = AudioTensor.__torch_function__.__func__

    def counting(cls, func, types, args=(), kwargs=None):
        calls.append(getattr(func, "__name__", str(func)))
        return orig(cls, func, types, args, kwargs)

    AudioTensor.__torch_function__ = classmethod(counting)
    try:
        yield calls
    finally:
        AudioTensor.__torch_function__ = classmethod(orig)


def at(*shape, hop=1, **kw):
    return AudioTensor(torch.randn(*shape), hop, **kw)


# ---------------------------------------------------------------- basics
def test_hop_length_propagates():
    a = AudioTensor(torch.randn(2, 5), hop_length=2)
    assert (a * 2).hop_length == 2


def test_mixed_hop_lengths_resample_to_gcd():
    a = AudioTensor(torch.randn(2, 5), hop_length=2)
    b = AudioTensor(torch.randn(2, 9), hop_length=1)
    out = a + b
    assert out.hop_length == 1
    assert out.shape == (2, 9)


def test_repr_first_line():
    assert repr(at(2, 5, hop=2)).splitlines()[0] == "Hop-length: 2"
    assert repr(at(2, 5, hop=2).sum(1)).splitlines()[0] == "Hop-length: -1"


def test_invalid_interpolation_mode():
    with pytest.raises(ValueError, match="mode"), interpolation_mode("bogus"):
        pass


def test_unknown_mode_set_directly_on_contextvar():
    token = interpolation_mode_var.set("bogus")
    try:
        with pytest.raises(ValueError, match="Unknown interpolation mode"):
            AudioTensor(torch.randn(2, 5), 2).reduce_hop_length()
        with pytest.raises(ValueError, match="Unknown interpolation mode"):
            upsample(torch.randn(2, 5), 2, "bogus")
    finally:
        interpolation_mode_var.reset(token)


# ---------------------------------------------------------------- resampling
def test_upsample_is_pure_explicit_mode():
    x = torch.arange(5.0).repeat(2, 1)
    expected = upsample(x, 2, "linear")
    with interpolation_mode("nearest"):  # must not influence `upsample`
        assert torch.equal(upsample(x, 2, "linear"), expected)
        assert not torch.equal(upsample(x, 2, "nearest"), expected)
    assert expected[0].tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def test_context_mode_is_read_at_the_boundaries():
    a = AudioTensor(torch.arange(5.0).repeat(2, 1), 2)
    b = AudioTensor(torch.zeros(2, 9), 1)
    with interpolation_mode("nearest"):
        via_method = a.reduce_hop_length()
        via_op = a + b
    assert via_method[0].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    assert torch.equal(via_op.as_tensor(), via_method.as_tensor())
    assert a.reduce_hop_length()[0, 1] == 0.5  # back to linear outside


@pytest.mark.parametrize("mode", ["linear", "nearest", "cubic"])
def test_reduce_hop_length_modes(mode):
    a = AudioTensor(torch.randn(2, 5), hop_length=2)
    with interpolation_mode(mode):
        out = a.reduce_hop_length()
    assert out.hop_length == 1
    assert out.shape[0] == 2


def _cubic_reference(x: torch.Tensor, hop: int) -> torch.Tensor:
    """cubic_spline applied directly on a plain tensor, time moved last."""
    if x.ndim > 2:
        x = x.transpose(1, -1)
    y = cubic_spline(x.flatten(0, -2), hop).view(*x.shape[:-1], -1)
    if x.ndim > 2:
        y = y.transpose(1, -1)
    return y


@pytest.mark.parametrize(
    "shape,hop",
    [
        ((2, 60), 2),
        ((3, 64), 3),
        ((2, 55), 4),
        ((2, 60, 3), 2),
        ((2, 52, 3), 4),
        ((2, 60, 3, 2), 3),
        ((1, 50, 2, 2), 4),
    ],
)
def test_cubic_reduce_hop_length_matches_plain_cubic_spline(shape, hop):
    # Regression: AudioTensor inputs used to make philtorch's internal ops
    # truncate to the minimum time length, corrupting the last output frames.
    x = torch.randn(*shape)
    a = AudioTensor(x, hop)
    with interpolation_mode("cubic"):
        out = a.reduce_hop_length()
        ref = _cubic_reference(x, hop)
        mixed = a + AudioTensor(torch.zeros_like(ref), 1)  # mixed-hop path
    assert out.hop_length == 1 and out.shape == ref.shape
    assert torch.allclose(out.as_tensor(), ref)
    # the last frames specifically
    assert torch.allclose(out.as_tensor()[:, -10:], ref[:, -10:])
    assert mixed.hop_length == 1 and mixed.shape == ref.shape
    assert torch.allclose(mixed.as_tensor(), ref)


def test_nearest_mode_values():
    a = AudioTensor(torch.arange(5.0).repeat(2, 1), 2)
    with interpolation_mode("nearest"):
        out = a.reduce_hop_length()
    assert out[0].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]


def test_resampling_methods_values_and_hop():
    a = AudioTensor(torch.arange(10.0).view(2, 5), 2)
    r = a.reduce_hop_length()
    assert r.hop_length == 1 and r.shape == (2, 9)
    assert r[0].tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    i = a.increase_hop_length(2)
    assert i.hop_length == 4 and i.shape == (2, 3)
    assert a.set_hop_length(4).hop_length == 4
    assert a.set_hop_length(1).hop_length == 1
    assert a.set_hop_length(2) is a
    assert a.truncate(3).shape == (2, 3) and a.truncate(10) is a
    assert a.steps == 5
    assert a.increase_hop_length(1) is a
    assert a.reduce_hop_length(1) is a
    r3 = at(2, 5, 3, hop=2).reduce_hop_length()
    assert r3.shape == (2, 9, 3) and r3.hop_length == 1


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda a: a.set_hop_length(5), "not a multiple"),  # hop 4 -> 5
        (lambda a: a.set_hop_length(6), "not a multiple"),
        (lambda a: a.set_hop_length(3), "does not divide"),
        (lambda a: a.set_hop_length(0), "must be >= 1"),
        (lambda a: a.set_hop_length(-1), "must be >= 1"),
        (lambda a: a.increase_hop_length(0), "factor must be positive"),
        (lambda a: a.increase_hop_length(-2), "factor must be positive"),
        (lambda a: a.reduce_hop_length(3), "divisor"),
        (lambda a: a.reduce_hop_length(8), "divisor"),
        (lambda a: a.reduce_hop_length(0), "divisor"),
        (lambda a: a.reduce_hop_length(-2), "divisor"),
    ],
)
def test_invalid_resampling_arguments_raise_and_leave_tensor_unchanged(call, match):
    data = torch.arange(10.0).view(2, 5)
    a = AudioTensor(data.clone(), 4)
    with pytest.raises(ValueError, match=match):
        call(a)
    assert a.hop_length == 4 and torch.equal(a.as_tensor(), data)


def test_aliasing_contract():
    a = AudioTensor(torch.arange(10.0).view(2, 5), 2)
    # no-ops return self
    assert a.increase_hop_length(1) is a and a.reduce_hop_length(1) is a
    assert a.set_hop_length(2) is a and a.truncate(5) is a and a.truncate(9) is a
    # truncate: a view sharing storage
    t = a.truncate(3)
    assert t is not a and t.data_ptr() == a.data_ptr()
    # increase_hop_length (clone) and reduce_hop_length: fresh storage
    assert a.increase_hop_length(2).data_ptr() != a.data_ptr()
    assert a.reduce_hop_length().data_ptr() != a.data_ptr()


def test_methods_on_negative_hop_raise():
    a = AudioTensor(torch.randn(3, 2), -1)
    for call in (
        lambda: a.steps,
        lambda: a.truncate(1),
        lambda: a.reduce_hop_length(),
        lambda: a.increase_hop_length(2),
        lambda: a.set_hop_length(2),
        lambda: a.unfold(2),
    ):
        with pytest.raises(ValueError):
            call()


# ---------------------------------------------------------------- constructor
def test_constructor_accepts_lists_and_ndarrays():
    a = AudioTensor([[1, 2], [3, 4]], 3)
    assert a.hop_length == 3 and a.dtype == torch.int64
    b = AudioTensor(np.zeros((2, 3)), hop_length=2)
    assert isinstance(b, AudioTensor) and b.hop_length == 2


def test_constructor_shares_storage_and_keeps_graph():
    w = torch.randn(2, 5, requires_grad=True)
    y = AudioTensor(w * 2, 1)
    assert y.grad_fn is not None and y.requires_grad
    y.sum().backward()
    assert torch.equal(w.grad, torch.full((2, 5), 2.0))
    x = torch.randn(2, 3)
    assert AudioTensor(x, 1).data_ptr() == x.data_ptr()


def test_constructor_requires_grad_leaf_backward_and_sgd():
    p = AudioTensor(torch.randn(2, 5), 1, requires_grad=True)
    assert p.is_leaf and p.requires_grad
    p.sum().backward()
    assert p.grad is not None and p.grad.shape == (2, 5)
    before = p.detach().clone()
    p.grad = None
    opt = torch.optim.SGD([p], lr=0.1)
    (p**2).sum().backward()
    opt.step()
    assert not torch.equal(p.detach(), before)
    assert isinstance(p, AudioTensor) and p.hop_length == 1


def test_constructor_requires_grad_false_and_subclass():
    p = AudioTensor(torch.randn(2, 5, requires_grad=True), 1, requires_grad=False)
    assert not p.requires_grad

    class Sub(AudioTensor):
        pass

    s = Sub(torch.ones(2, 5), 2, requires_grad=True)
    assert type(s) is Sub and s.hop_length == 2 and s.requires_grad
    out = s * 2
    assert type(out) is Sub and out.hop_length == 2


@pytest.mark.parametrize(
    "shape,hop",
    [
        ((2, 3), 0),
        ((2, 3), -2),
        ((5,), 1),
        ((), 1),
        ((2, 3), 1.5),
        ((2, 3), "1"),
        ((2, 3), True),
    ],
)
def test_constructor_rejects_invalid(shape, hop):
    with pytest.raises(ValueError):
        AudioTensor(torch.randn(shape), hop)


@pytest.mark.parametrize("shape", [(5,), (2, 3), (), (2, 3, 4)])
def test_constructor_minus_one_any_ndim(shape):
    assert AudioTensor(torch.randn(shape), -1).hop_length == -1


# ---------------------------------------------------------------- in-place
def _iadd(x, y):
    # `+=` rebinds the name: the returned object must still be the target
    z = x
    z += y
    return z


def _finest_target():
    x = AudioTensor(torch.randn(2, 9), 1)
    y = AudioTensor(torch.randn(2, 5), 2)
    return x, y, x.as_tensor().clone(), y.reduce_hop_length().as_tensor()


@pytest.mark.parametrize(
    "op,ref",
    [
        (lambda x, y: x.add_(y), lambda x0, yu: x0 + yu),
        (lambda x, y: x.mul_(y), lambda x0, yu: x0 * yu),
        (lambda x, y: x.__iadd__(y), lambda x0, yu: x0 + yu),
        (_iadd, lambda x0, yu: x0 + yu),
    ],
)
def test_inplace_target_finest_resamples_operand(op, ref):
    x, y, x0, yu = _finest_target()
    r = op(x, y)
    assert r is x and x.hop_length == 1 and x.shape == (2, 9)
    assert isinstance(x, AudioTensor)
    assert torch.allclose(x.as_tensor(), ref(x0, yu))
    assert y.shape == (2, 5) and y.hop_length == 2  # operand untouched


def test_inplace_longer_operand_truncated_to_target():
    x = AudioTensor(torch.zeros(2, 5), 1)
    y = AudioTensor(torch.arange(2 * 9.0).view(2, 9), 2)  # upsampled: T=17
    r = x.add_(y)
    assert r is x and x.shape == (2, 5)
    assert torch.allclose(x.as_tensor(), y.reduce_hop_length().as_tensor()[:, :5])
    # same hop but longer operand: no truncation, torch's error surfaces
    x = AudioTensor(torch.zeros(2, 5), 1)
    with pytest.raises(RuntimeError):
        x.add_(AudioTensor(torch.ones(2, 9), 1))
    assert x.shape == (2, 5) and x.sum() == 0


def test_inplace_shorter_operand_not_padded():
    x = AudioTensor(torch.zeros(2, 20), 1)
    y = AudioTensor(torch.ones(2, 5), 2)  # upsampled: T=9
    with pytest.raises(RuntimeError):
        x.add_(y)
    assert x.shape == (2, 20) and x.sum() == 0


def test_inplace_target_coarser_raises_and_untouched():
    x = AudioTensor(torch.zeros(2, 5), 2)
    y = AudioTensor(torch.ones(2, 9), 1)
    with pytest.raises(ValueError, match="set_hop_length"):
        x.add_(y)
    with pytest.raises(ValueError, match="hop_length"):
        x.__iadd__(y)
    with pytest.raises(ValueError, match="hop_length"):
        x.mul_(y)
    # target hop 2, operand hop 3: gcd 1 < 2
    with pytest.raises(ValueError, match="hop_length"):
        x.add_(AudioTensor(torch.ones(2, 4), 3))
    assert x.sum() == 0 and x.hop_length == 2 and x.shape == (2, 5)


def test_inplace_three_operands():
    x = AudioTensor(torch.zeros(2, 13), 1)
    y = AudioTensor(torch.randn(2, 5), 2)
    z = AudioTensor(torch.randn(2, 5), 3)
    # hop 2 -> T=9; hop 3 -> T=13
    with pytest.raises(RuntimeError):
        x.add_(y)  # shorter operand (9 < 13) is not padded
    x = AudioTensor(torch.zeros(2, 9), 1)
    r = x.addcmul_(y, z)
    assert r is x and x.hop_length == 1 and x.shape == (2, 9)
    yu = y.reduce_hop_length().as_tensor()
    zu = z.reduce_hop_length().as_tensor()[:, :9]
    assert torch.allclose(x.as_tensor(), yu * zu)


def test_inplace_gradient_flows_through_resampled_operand():
    x = AudioTensor(torch.randn(2, 9), 1)
    y = AudioTensor(torch.randn(2, 5), 2, requires_grad=True)
    r = x.add_(y)
    assert r is x and r.requires_grad
    r.sum().backward()
    assert y.grad is not None and y.grad.shape == (2, 5)
    assert torch.allclose(y.grad.sum(), torch.tensor(18.0))
    # `+=` on a non-leaf clone
    w = AudioTensor(torch.randn(2, 5), 1, requires_grad=True)
    v = AudioTensor(torch.randn(2, 3), 2, requires_grad=True)
    t = w * 1
    t += v
    t.sum().backward()
    assert v.grad.shape == (2, 3) and w.grad.shape == (2, 5)


def test_inplace_setitem():
    x = AudioTensor(torch.zeros(2, 9), 1)
    y = AudioTensor(torch.ones(2, 5), 2)
    with pytest.raises(ValueError, match="hop_length"):
        x[:, :2] = y[:, :2]
    assert x.sum() == 0
    # same hop works
    x[:, :2] = AudioTensor(torch.ones(2, 2), 1)
    assert x.sum() == 4 and x.hop_length == 1


def test_inplace_out_kwarg():
    x = AudioTensor(torch.zeros(2, 9), 1)
    y = AudioTensor(torch.ones(2, 5), 2)
    out = AudioTensor(torch.zeros(2, 9), 1)
    r = torch.add(x, y, out=out)
    assert r is out and out.hop_length == 1
    assert torch.allclose(out.as_tensor(), y.reduce_hop_length().as_tensor())
    coarse = AudioTensor(torch.zeros(2, 5), 2)
    with pytest.raises(ValueError, match="hop_length"):
        torch.add(x, y, out=coarse)
    # tuple of outputs, plain `out` tensors
    v, i = AudioTensor(torch.zeros(2), -1), torch.zeros(2, dtype=torch.long)
    vals, idx = torch.max(at(2, 5), 1, out=(v, i))
    assert vals is v and torch.equal(idx.as_tensor(), i)
    with pytest.raises(ValueError, match="hop_length"):
        torch.add(x, y, out=torch.zeros(2, 9))  # plain destination


def test_inplace_mixed_hops_with_plain_target_raises():
    plain = torch.zeros(2, 9)
    assert plain.add_(at(2, 9, hop=2)).shape == (2, 9)  # single operand is fine
    with pytest.raises(ValueError, match="hop_length"):
        plain.addcmul_(at(2, 9, hop=1), at(2, 5, hop=2))


def test_inplace_with_plain_tensor_and_scalar():
    x = AudioTensor(torch.zeros(2, 5), 2)
    assert x.add_(torch.ones(2, 5)).hop_length == 2
    assert x.mul_(2).sum() == 20


@pytest.mark.parametrize("y_hop,y_steps", [(1, 5), (2, 3)])
def test_inplace_lower_ndim_operand_is_aligned(y_hop, y_steps):
    x = AudioTensor(torch.zeros(2, 5, 3), 1)
    y = AudioTensor(torch.ones(2, y_steps), y_hop)
    r = x.add_(y)
    assert r is x and x.shape == (2, 5, 3) and x.hop_length == 1
    assert x.sum() == 2 * 5 * 3


def test_inplace_lower_ndim_target_raises_untouched():
    x = AudioTensor(torch.zeros(2, 5), 1)
    y = AudioTensor(torch.ones(2, 5, 3), 1)
    with pytest.raises(RuntimeError):
        x.add_(y)
    assert x.sum() == 0 and x.shape == (2, 5)
    x = AudioTensor(torch.zeros(2, 9), 1)
    y = AudioTensor(torch.ones(2, 5, 3), 2)
    with pytest.raises(RuntimeError):
        x.add_(y)
    assert x.sum() == 0 and x.shape == (2, 9)


def test_inplace_never_relabels_the_target():
    x = AudioTensor(torch.zeros(2, 5), -1)
    r = x.add_(AudioTensor(torch.ones(2, 5), 3))
    assert r is x and x.hop_length == -1 and x.sum() == 10
    out = AudioTensor(torch.zeros(2, 5), -1)
    r = torch.add(at(2, 5, hop=3), 1, out=out)
    assert r is out and out.hop_length == -1
    # same-hop and finest-target cases keep identity and hop
    x = AudioTensor(torch.zeros(2, 5), 2)
    y = AudioTensor(torch.ones(2, 5), 2)
    assert x.add_(y) is x and x.hop_length == 2 and x.sum() == 10
    x += y
    assert x.sum() == 20 and isinstance(x, AudioTensor) and x.hop_length == 2
    x[:, 0] = 5.0
    assert x[:, 0].tolist() == [5.0, 5.0]
    x = AudioTensor(torch.zeros(2, 9), 1)
    assert x.add_(AudioTensor(torch.ones(2, 5), 2)) is x and x.hop_length == 1
    out = AudioTensor(torch.zeros(2, 9), 1)
    assert torch.add(x, at(2, 5, hop=2), out=out) is out and out.hop_length == 1


def test_ops_returning_self_keep_the_object_and_hop():
    a = at(2, 5, hop=2)
    assert a.contiguous() is a and a.hop_length == 2
    assert a.to(a.dtype) is a and a.hop_length == 2
    assert a.requires_grad_(False) is a and a.hop_length == 2
    n = AudioTensor(torch.randn(2, 5), -1)
    assert n.contiguous() is n and n.hop_length == -1


# ---------------------------------------------------------------- out-of-place
def test_cat_many_tensors_no_recursion():
    xs = [AudioTensor(torch.ones(1, 3), -1)] + [torch.ones(1, 3)] * 3000
    out = torch.cat(xs, 0)
    assert out.shape == (3001, 3)


def test_output_hop_is_first_positive_hop():
    a = AudioTensor(torch.ones(2, 4), -1)
    b = AudioTensor(torch.ones(2, 4), 3)
    assert (a + b).hop_length == 3
    assert (a + torch.ones(2, 4)).hop_length == -1
    # a hop -1 tensor is ignored when reconciling
    t = AudioTensor(torch.ones(5, 2), -1)
    assert (t + torch.ones(5, 2)).hop_length == -1
    assert (t + at(5, 2, hop=3)).hop_length == 3


def test_torch_function_called_once_and_helpers_do_not_reenter():
    a = at(2, 5, hop=2)
    a2 = at(2, 5, hop=2)
    b = at(2, 9, hop=1)
    with count_torch_function() as calls:
        a + 1
        assert len(calls) == 1
        calls.clear()
        a + a2
        assert len(calls) == 1
        calls.clear()
        out = a + b
        assert len(calls) == 1
        # helpers work on plain tensors and never re-enter
        calls.clear()
        c = at(2, 5, 3, hop=2)
        c.reduce_hop_length()
        c.increase_hop_length(2)
        c.set_hop_length(1)
        c.truncate(3)
        _ = c.steps
        c.as_tensor()
        assert calls == []
    assert out.hop_length == 1


def test_non_tensor_and_tuple_results():
    a = at(2, 5, hop=2)
    assert a.shape == (2, 5) and a.dim() == 2 and a.size(1) == 5
    assert a.is_floating_point() is True
    vals, _ = a.max(dim=1)
    assert vals.hop_length == -1 and type(vals) is AudioTensor
    s, _ = torch.sort(a, dim=1)
    assert s.hop_length == 2 and s.shape == (2, 5)
    parts = a.split(2, dim=1)
    assert [p.hop_length for p in parts] == [2, 2, 2]
    assert a.grad is None


def test_foreign_subclass_defers_before_any_preprocessing():
    class Other(torch.Tensor):
        pass

    o = torch.ones(2, 5).as_subclass(Other)
    with pytest.raises(TypeError):
        torch.add(at(2, 5, hop=2), o)
    # without the NotImplemented guard our in-place hop check would raise
    # ValueError before torch gets to try `Other.__torch_function__`
    x = AudioTensor(torch.zeros(2, 5), 2)
    with pytest.raises(TypeError):
        x.addcmul_(at(2, 9, hop=1), o)
    assert x.sum() == 0


def test_as_tensor_shares_storage_and_autograd():
    p = AudioTensor(torch.randn(2, 5), 1, requires_grad=True)
    t = (p * 2).as_tensor()
    assert type(t) is torch.Tensor and t.grad_fn is not None
    assert p.as_tensor().data_ptr() == p.data_ptr()
    t.sum().backward()
    assert torch.equal(p.grad, torch.full((2, 5), 2.0))


def test_broadcast_aligns_ndim_and_truncates():
    a = at(2, 5, hop=2)
    b = at(2, 9, 3, hop=1)
    out = a * b
    assert out.shape == (2, 9, 3) and out.hop_length == 1
    c = at(2, 4, hop=4) + at(2, 3, hop=6)
    assert c.hop_length == 2 and c.shape == (2, 7)
    short = at(2, 3, hop=1) + at(2, 9, hop=1)
    assert short.shape == (2, 3)


# ---------------------------------------------------------------- time-axis contract
def test_low_ndim_results_get_negative_hop():
    a = at(2, 5, hop=2)
    s = a.sum()
    assert s.ndim == 0 and s.hop_length == -1
    assert a.sum(1).hop_length == -1
    assert a[0].hop_length == -1
    assert a.sum(0).hop_length == -1
    assert a.flatten().hop_length == -1


def test_time_axis_contract_ops_keep_hop():
    # The hop describes dim 1. Only results with ndim < 2 get hop -1; ops that
    # move/remove the time axis keep the hop and the caller must handle it
    # (escape hatch: `.as_tensor()`). Do not "fix" this without a decision.
    a = at(2, 5, hop=2)
    assert a.T.hop_length == 2 and a.T.shape == (5, 2)
    assert a.transpose(0, 1).hop_length == 2
    assert a.reshape(5, 2).hop_length == 2
    assert a.view(2, 5, 1).hop_length == 2
    assert a.unsqueeze(0).hop_length == 2


# ---------------------------------------------------------------- unfold
def test_unfold_legacy_2d():
    a = at(2, 5, hop=2)
    u = a.unfold(2, 1)
    assert u.shape == (2, 4, 2) and u.hop_length == 2
    u = a.unfold(2, 2)
    assert u.shape == (2, 2, 2) and u.hop_length == 4
    u = a.unfold(2)
    assert u.shape == (2, 4, 2) and u.hop_length == 2
    u = a.unfold(size=2, step=2)
    assert u.hop_length == 4


@pytest.mark.parametrize(
    "shape,call,out_shape,out_hop",
    [
        ((2, 5, 1), lambda a: a.unfold(1, 2, 2), (2, 2, 1, 2), 4),  # standard form
        ((2, 5), lambda a: a.unfold(1, 2, 2), (2, 2, 2), 4),
        ((2, 5), lambda a: a.unfold(dimension=-1, size=2, step=2), (2, 2, 2), 4),
        ((2, 5), lambda a: a.unfold(0, 2, 1), (1, 5, 2), 2),  # batch dim: unchanged
        ((2, 5), lambda a: torch.Tensor.unfold(a, 1, 2, 2), (2, 2, 2), 4),  # bypass
        ((2, 6, 3), lambda a: a.unfold(2, 1), (2, 5, 3, 2), 2),  # legacy, 3D
        ((2, 6, 3), lambda a: a.unfold(1, 2, 2), (2, 3, 3, 2), 4),
        ((2, 6, 3), lambda a: a.unfold(2, 2, 1), (2, 6, 2, 2), 2),  # dim 2: unchanged
        ((2, 6, 3), lambda a: a.unfold(-2, 2, 3), (2, 2, 3, 2), 6),
    ],
)
def test_unfold_standard_forms(shape, call, out_shape, out_hop):
    u = call(at(*shape, hop=2))
    assert isinstance(u, AudioTensor)
    assert tuple(u.shape) == out_shape and u.hop_length == out_hop


def test_functional_unfold_not_confused():
    # F.unfold (im2col) has the same __name__ but is not Tensor.unfold
    x = AudioTensor(torch.randn(1, 2, 4, 4), 2)
    out = torch.nn.functional.unfold(x, 2)
    assert out.hop_length == 2


# ---------------------------------------------------------------- new_tensor
def test_new_tensor_standard_signature():
    a = at(2, 5, hop=2)
    n = a.new_tensor([[1.0, 2.0]], dtype=torch.float64)
    assert isinstance(n, AudioTensor)
    assert n.dtype == torch.float64 and n.hop_length == 2
    assert n.shape == (1, 2)
    n1 = a.new_tensor([1.0, 2.0])
    assert n1.hop_length == -1
    assert a.new_tensor([[1.0]], requires_grad=True).requires_grad


# ---------------------------------------------------------------- deepcopy / serialization
def test_deepcopy_preserves_everything():
    a = AudioTensor(torch.arange(10.0).view(2, 5), 3)
    a.extra = {"k": [1, 2]}
    b = copy.deepcopy(a)
    assert isinstance(b, AudioTensor) and b.hop_length == 3
    assert torch.equal(b.as_tensor(), a.as_tensor())
    assert b.dtype == a.dtype
    assert b.data_ptr() != a.data_ptr()
    assert b.extra == a.extra and b.extra is not a.extra
    b[0, 0] = 100.0
    assert a[0, 0] == 0.0


def test_deepcopy_requires_grad_leaf():
    a = AudioTensor(torch.randn(2, 5, dtype=torch.float64), 2, requires_grad=True)
    b = copy.deepcopy(a)
    assert b.requires_grad and b.is_leaf and b.dtype == torch.float64
    assert b.hop_length == 2
    b.sum().backward()
    assert b.grad is not None and a.grad is None


def test_deepcopy_copies_grad():
    a = AudioTensor(torch.randn(2, 5), 2, requires_grad=True)
    (a * 3).sum().backward()
    b = copy.deepcopy(a)
    assert torch.equal(b.grad, a.grad) and b.grad is not a.grad
    assert b.grad.data_ptr() != a.grad.data_ptr()


def test_deepcopy_conj_and_neg_bits():
    z = torch.randn(2, 3, dtype=torch.complex64)
    a = AudioTensor(z.conj(), 2)
    assert a.is_conj()
    b = copy.deepcopy(a)
    assert b.hop_length == 2  # torch materializes the conj bit when copying
    assert torch.equal(b.as_tensor(), a.as_tensor())
    n = AudioTensor(torch._neg_view(torch.randn(2, 3)), 2)
    assert n.is_neg()
    m = copy.deepcopy(n)
    assert torch.equal(m.as_tensor(), n.as_tensor())


def test_deepcopy_non_leaf_raises():
    a = AudioTensor(torch.randn(2, 5), 2, requires_grad=True)
    with pytest.raises(RuntimeError):
        copy.deepcopy(a * 2)


def test_deepcopy_memo_and_shared_storage():
    a = AudioTensor(torch.randn(2, 5), 2)
    v = a.as_tensor()[:, :3]
    c = copy.deepcopy({"a": a, "a2": a, "v": v})
    assert c["a"] is c["a2"]
    assert c["v"].data_ptr() == c["a"].data_ptr()


def test_deepcopy_module_with_audiotensor_attribute():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            self.buf = AudioTensor(torch.randn(2, 5), 4)
            self.reg = AudioTensor(torch.randn(2, 5), 4, requires_grad=True)

    m = M()
    m2 = copy.deepcopy(m)
    assert m2.buf.hop_length == 4 and isinstance(m2.buf, AudioTensor)
    assert m2.buf.data_ptr() != m.buf.data_ptr()
    assert m2.reg.requires_grad and m2.reg.is_leaf


def test_pickle_and_torch_save_roundtrip():
    a = AudioTensor(torch.arange(10.0).view(2, 5), 2)
    b = pickle.loads(pickle.dumps(a))
    assert isinstance(b, AudioTensor) and b.hop_length == 2
    f = io.BytesIO()
    torch.save(a, f)
    f.seek(0)
    c = torch.load(f)  # weights_only=True by default
    assert isinstance(c, AudioTensor) and c.hop_length == 2
    assert torch.equal(c.as_tensor(), a.as_tensor())


def test_copy_copy_keeps_hop():
    a = at(2, 5, hop=2)
    assert copy.copy(a).hop_length == 2
