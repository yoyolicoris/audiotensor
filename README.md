# AudioTensor

A `torch.Tensor` subclass that remembers the `hop_length` (samples per frame) of frame-based audio features, so tensors at different frame rates can be combined directly.

```bash
pip install audiotensor
```

```python
import torch
from audiotensor import AudioTensor, interpolation_mode

f0 = AudioTensor(torch.randn(4, 100), hop_length=256)       # (B, T) at hop 256
loudness = AudioTensor(torch.randn(4, 400), hop_length=64)  # (B, T) at hop 64

x = f0 * loudness       # f0 is upsampled to hop 64 first
print(x.hop_length)     # 64

with interpolation_mode("cubic"):  # "linear" (default), "nearest", "cubic"
    y = f0 + loudness
```

- Wrap your tensor as `AudioTensor(tensor, hop_length)` and use it as a regular tensor.
- For `ndim > 1`, the shape is `(B, T, ...)`, where `B` is the batch size and `T` is the number of frames.
- Results with `ndim < 2` (e.g. `x.sum()`, `x[0]`) have no time axis and get `hop_length = -1`.
- Tensors with different `hop_length` are automatically resampled to the greatest common divisor of their `hop_length`s and truncated to the shortest one when you combine them.
- In-place operations (`x += y`, `x.add_(y)`, ...) resample the other operands to `x`, never `x` itself, so `x` must have the finest `hop_length`.
- `hop_length` always describes dim 1. Ops that move or remove that axis (`transpose`, `reshape`, ...) keep the `hop_length` unchanged, so use `as_tensor()` first if that is not what you want.
- `set_hop_length(hop_length)` resamples the tensor to the new `hop_length`.
- `reduce_hop_length()` upsamples the tensor to `hop_length = 1`.
- `steps` returns the size of `T` in the tensor.
- `truncate(steps)` truncates the tensor to the given `steps`.
- `unfold(size, step)` returns a view of the tensor with a sliding window of size `size` and step `step` along the `T` dimension.
- `as_tensor()` returns a plain `torch.Tensor`.
