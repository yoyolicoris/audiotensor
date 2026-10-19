# AudioTensor

`AudioTensor` is a `torch.Tensor` subclass that carries a `hop_length` attribute (samples per frame) and keeps it consistent across torch operations.

## Data model

- For `ndim > 1` the shape is `(B, T, ...)`: `B` is the batch size and `T` the number of frames (time is always dim 1).
- `hop_length = -1` means "no time axis". Such tensors may have any `ndim` (for example a 1D `(B,)` tensor, or a 0-d scalar). Methods that need a time axis (`steps`, `truncate`, `set_hop_length`, ...) raise `ValueError` on them, and they are ignored when operands are reconciled.

## Constructor

```python
AudioTensor(data, hop_length=1, *, requires_grad=None)
```

- `data` is anything `torch.as_tensor` accepts (tensor, list, `numpy` array).
- `hop_length` must be an `int`: either `-1`, or `>= 1` with `ndim >= 2`. Otherwise `ValueError` is raised (`0` is rejected).
- With `requires_grad=None` (default) the result shares storage with `data` and stays attached to its autograd graph (`as_subclass`). With `requires_grad=True/False` a new leaf is created (like `torch.Tensor._make_subclass`), so `AudioTensor(x, 1, requires_grad=True)` works with `.backward()` and `torch.optim`.

## Mixed hop lengths

Out-of-place operations on AudioTensors with different `hop_length`s resample all operands to the greatest common divisor of the hops (time axis interpolated, see `interpolation_mode`), align `ndim` by appending trailing singleton dims, and truncate to the shortest `T`. The result gets the resampled hop.

```python
with interpolation_mode(
    "nearest"
):  # "linear" (default), "nearest", "cubic"; others raise ValueError
    out = a + b
```

## In-place rule

In-place operations (`add_`, `x += y`, `out=`, ...) broadcast the operands, never the destination. With several AudioTensors (`hop_length > 0`):

- If the destination already has the finest hop (its `hop_length` is the gcd of all hops), the other AudioTensor operands are resampled to it. Operands longer than the destination's `T` are truncated; shorter ones are not padded (torch raises its usual shape error). The destination is modified in place and returned (`r is x`), and gradients flow through the resampled operands.
- If an operand has a finer hop than the destination, `ValueError` is raised (the destination cannot be resampled in place; call `set_hop_length` on it first).
- `x[...] = y` (`__setitem__`) is strict: any hop mismatch raises `ValueError`.
- Only operands are aligned in `ndim` (trailing singleton dims); a destination with fewer dims than an operand is left to torch's shape error.
- With equal hops nothing is resampled or truncated; shape mismatches are left to torch.

## Time-axis contract

The hop describes dim 1. The only automatic invalidation is for results with `ndim < 2` (`a.sum()`, `a.sum(1)`, `a[0]`, `a.flatten()`, ...), which get `hop_length = -1`. Every other result keeps the input's hop, including ops that move or remove the time axis (`transpose` / `permute` / `.T` / `.mT`, `reshape`, integer indexing or reducing over dim 1, `stack`, `einsum`, ...). The caller is responsible for those, e.g. by working on `.as_tensor()` or re-wrapping with `AudioTensor(x, hop_length)`.

## Methods

- `set_hop_length(hop_length)` resamples to the new `hop_length` (must divide / be a multiple of the current one).
- `reduce_hop_length(factor=None)` upsamples time by `factor` (default: all the way to `hop_length = 1`).
- `increase_hop_length(factor)` downsamples time by striding (no filtering).
- `steps` is the size of `T`; `truncate(steps)` keeps the first `steps` frames.
- `as_tensor()` returns a plain `torch.Tensor` sharing storage and autograd history.
- `unfold`: `a.unfold(dimension, size, step)` is the standard `torch.Tensor.unfold`; when the unfolded dimension is the time axis (dim 1) the output hop is `hop_length * step`, otherwise the hop is unchanged. The shorthand `a.unfold(size, step=1)` unfolds the time axis. `torch.Tensor.unfold(a, 1, size, step)` behaves identically.
- `new_tensor` is the standard `torch.Tensor.new_tensor` (`a.new_tensor(data, dtype=..., requires_grad=...)`); the hop is inherited, or `-1` for 1D data.

## Aliasing

`set_hop_length`, `increase_hop_length`, `reduce_hop_length` and `truncate` return `self` when the request is a no-op, `truncate` otherwise returns a view sharing storage with the original, and `increase_hop_length` / `reduce_hop_length` return fresh storage. Clone before mutating in place if it matters. In-place operations never relabel the `hop_length` of an existing tensor, whichever operand it was.

## Copying and saving

`copy.deepcopy` (leaf tensors only, as in torch), `pickle` and `torch.save` / `torch.load` (default `weights_only=True`; `AudioTensor` is registered as a safe global on import) preserve `hop_length`.
