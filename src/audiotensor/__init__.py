from importlib.metadata import PackageNotFoundError, version

from .audiotensor import AudioTensor, interpolation_mode

try:
    __version__ = version("audiotensor")
except PackageNotFoundError:  # pragma: no cover - source tree without installing
    __version__ = "0+unknown"

__all__ = ["AudioTensor", "interpolation_mode"]
