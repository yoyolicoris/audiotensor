# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## State of the repo

Tiny library: `AudioTensor`, a `torch.Tensor` subclass carrying a `hop_length` attribute. Previously a git submodule of another project; now a proper package (setuptools, `src/` layout, version in [pyproject.toml](pyproject.toml)). Source lives in [src/audiotensor/](src/audiotensor/); the package is installed editable into the pixi env.

## Commands

All via pixi (tasks in [pixi.toml](pixi.toml)):
- `pixi install` - create/update env (default = py313 + test + dev)
- `pixi run test` - pytest with `--cov --cov-branch`; fails below 100% coverage (`fail_under` in [pyproject.toml](pyproject.toml)); writes `coverage.xml`. Single test: `pixi run pytest tests/test_audiotensor.py::test_hop_length_propagates`
- `pixi run -e py311 test` (also `py312`, `py313`) - suite on one interpreter; only the default env has the dev tools
- `pixi run lint` / `pixi run format` / `pixi run format-check` - ruff over `src tests scripts`
- `pixi run build` - sdist + wheel into `dist/`; `pixi run smoke` installs the newest wheel with `--no-deps --target` into a temp dir and runs an end-to-end check against it

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)): lint + format-check, tests on ubuntu/macos x py311-py313, build + smoke + `dist` artifact. All setup-pixi steps use `locked: true`, so re-run `pixi lock` after editing `pixi.toml`.

IDE diagnostics about missing `torch`/`numpy`/`philtorch` are false alarms: the IDE uses the system Python, not the pixi env.

Dependencies: `torch` (CPU build from conda-forge), `numpy`, and `philtorch` (PyPI; imported for `philtorch.lti.cubic_spline`, used by the `"cubic"` interpolation mode; it pins `torch==2.14.*`). Needs Python ≥ 3.11 (`requires-python`; conda-forge has no `pytorch` 2.14 for 3.10).

## Architecture

All logic is in [src/audiotensor/audiotensor.py](src/audiotensor/audiotensor.py). Public API: `AudioTensor`, `interpolation_mode` (context manager). User-facing docs are in [README.md](README.md).

**Data model.** For `ndim > 1` shape is `(B, T, ...)`: dim 0 batch, dim 1 time frames, `hop_length` = samples per frame. `hop_length = -1` marks a tensor with no time axis (any ndim); `check_hop_length`-decorated methods raise on it, and `__torch_function__` ignores such tensors when reconciling hops. `__new__` always sets `hop_length`; there is deliberately no class-level default (every torch-created result is stamped in `__torch_function__`). Only a raw `Tensor._make_subclass(AudioTensor, x)` bypassing `__new__` has no `hop_length`.

**Constructor.** `AudioTensor(data, hop_length=1, *, requires_grad=None)`; everything happens in `__new__` (no `__init__`; torch's `Tensor` does not define one, so the extra args are ignored). `torch.as_tensor(data)`, then `as_subclass` (keeps graph/storage) or `Tensor._make_subclass` when `requires_grad` is given. Invalid hops raise `ValueError`.

**`__torch_function__` is the core.** Every torch op on an `AudioTensor` goes through it, exactly once (internals run on plain tensors / under `torch._C.DisableTorchFunctionSubclass()` so they never re-enter):
1. Flatten args/kwargs with `tree_flatten`; collect `AudioTensor`s with `hop_length > 0`. Output hop = hop of the first one, else `-1`.
2. If more than one: out-of-place calls go to `_reconcile_resampled`, which resamples all to the gcd hop (`_upsample_time`), unsqueezes trailing dims to equal ndim and cuts all to the shortest `T` (`_narrow_time`). In-place calls (name ends in `_`, in-place dunders, `__setitem__`, or an `out=` kwarg; `_is_inplace`) go to `_reconcile_inplace`: the operands are broadcast to the destination (`args[0]`, or the `out` tensors), never the reverse: if the destination hop equals the gcd, the other operands are resampled to it and truncated to its `T` (not padded); if an operand has a finer hop, `ValueError`. `__setitem__` raises on any hop mismatch. Only the operands are ndim-aligned (trailing unsqueeze), never the destination: a lower-ndim destination is left to torch's own shape error. Equal hops: no resampling or truncation.
3. Run the real op through `super().__torch_function__`.
4. Post-process: stamp the output hop on every freshly created `AudioTensor` in the result (never on call inputs) (no `tree_map` when the result is a single Tensor). Only invalidation rule: `ndim < 2` -> `-1`. `Tensor.unfold` (`func is Tensor.unfold`) along dim 1 -> `hop * step` (inlined in `post_process`). Everything else keeps the hop, including ops that move/remove the time axis (transpose, reshape, ...): the hop describes dim 1 and the caller handles such ops (escape hatch: `.as_tensor()`). This is a deliberate decision; there is intentionally no name-based invalidation machinery.

Function names are only matched for in-place detection (`_is_inplace`).

**Resampling.** `reduce_hop_length` (upsample time axis by `factor`, hop shrinks) calls `_upsample_time`, which calls module-level `upsample(x, hop, mode)` (last-dim resampling, explicit mode); the mode comes from the `interpolation_mode_var` `ContextVar` (`"linear"` default, `"nearest"`, `"cubic"`). Switch modes with the `interpolation_mode(...)` context manager (unknown mode: `ValueError`). `increase_hop_length` (downsample) just strides `[:, ::factor]` - no filtering. Because `upsample` is applied to the last dim, `_upsample_time` transposes time (dim 1) to last for `ndim > 2` and back after. The public methods wrap plain-tensor results back with `_wrap` (`as_subclass` + set `hop_length`).

## Gotchas

- Tensors created inside torch (`as_subclass`, `_convert`) bypass `__new__`; the stamping in `__torch_function__` covers them. Never read `.ndim` / `.shape` / call tensor methods on an `AudioTensor` inside `__torch_function__` or helpers without `DisableTorchFunctionSubclass()`: it re-enters.
- `unfold` has a thin override: `unfold(size, step=1)` shorthand (time axis) vs the standard 3-argument form (`dimension` kwarg or exactly 3 args). The hop logic itself lives in `__torch_function__` and applies to `torch.Tensor.unfold(a, ...)` too.
- `new_tensor` is not overridden (standard torch signature).
- `copy.deepcopy` uses an own `__deepcopy__` (leaf-only): torch's default fails for subclasses because `new_empty` returns a plain Tensor under `DisableTorchFunctionSubclass` (reproducible with a bare `Tensor` subclass on torch 2.14). It deep-copies `self.detach()` as a plain tensor (torch handles storage/memo/conj/neg), re-wraps it with `_make_subclass`, then copies `.grad` and `__dict__`. `AudioTensor` is registered with `torch.serialization.add_safe_globals` at import so `torch.load` works with `weights_only=True`.
- In-place ops: the destination is never resampled. Mixed hops work only when the destination has the finest hop (operands are resampled/truncated to it); otherwise `ValueError`. The destination object itself is returned (`r is x`).
- Time-axis contract: `a.T`, `a.transpose(0, 1)`, `a.reshape(...)` etc. keep `hop_length` (pinned by `test_time_axis_contract_ops_keep_hop`). Do not add invalidation for them without a decision.
- Aliasing: no-op resampling/truncation returns `self`, `truncate` returns a view, `increase_hop_length`/`reduce_hop_length` return fresh storage (pinned by `test_aliasing_contract`, documented in README).
- `post_process` never touches the hop of an object that existed before the call (`inputs` set): in-place ops and `contiguous()`-like ops return the input untouched. Only `_wrap` (fresh tensors) and `__new__` assign `hop_length`.
- `upsample(x, hop_length, mode)` is pure; the `interpolation_mode_var` ContextVar is read only in `reduce_hop_length` and in `__torch_function__` (only when several AudioTensors must be resampled/aligned) and threaded down as `mode`.
- The `_reconcile_*` helpers are pure: they return `({index in flatten: replacement}, output_hop)`.
- The early `NotImplemented` guard in `__torch_function__` is needed: without it our in-place hop check raises `ValueError` before torch can defer to a foreign tensor subclass's `__torch_function__` (which should give `TypeError`); pinned by `test_foreign_subclass_defers_before_any_preprocessing`.
- Invalid resampling arguments (`set_hop_length`, `increase_hop_length`, `reduce_hop_length`) raise `ValueError`, not `assert`. Methods that rebuild a tensor go through `_rewrap`; the negative-hop check is `_require_time_axis`.
