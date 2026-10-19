"""Install the freshly built wheel into a temp dir and check that it works.

``python -m build`` can succeed while producing an unusable wheel. This installs
the newest wheel in ``dist/`` with ``--no-deps --target`` (torch and philtorch
come from the pixi environment), then imports it in a subprocess with the temp
dir first on ``sys.path`` and runs a tiny end-to-end check. Run after ``build``.
"""

import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

CHECK = """
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))

import torch

import audiotensor
from audiotensor import AudioTensor, interpolation_mode

pkg = Path(audiotensor.__file__).resolve()
assert target in pkg.parents, f"audiotensor imported from {pkg}, not from {target}"

a = AudioTensor(torch.zeros(1, 5, 3), hop_length=2)
b = AudioTensor(torch.ones(1, 9, 3), hop_length=1)
c = a + b
assert c.hop_length == 1, c.hop_length
assert tuple(c.shape) == (1, 9, 3), tuple(c.shape)

a = AudioTensor(torch.zeros(2, 5), hop_length=2)
b = AudioTensor(torch.ones(2, 9), hop_length=1)
c = a + b
assert c.hop_length == 1, c.hop_length
assert tuple(c.shape) == (2, 9), tuple(c.shape)

with interpolation_mode("cubic"):
    d = AudioTensor(torch.arange(10.0).reshape(1, 10), hop_length=4).reduce_hop_length(2)
assert d.hop_length == 2, d.hop_length
assert d.shape[1] > 10 and torch.isfinite(d).all(), tuple(d.shape)
"""


def main() -> int:
    """Install the newest wheel in ``dist/`` and verify it."""
    wheels = sorted(DIST.glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        print(f"no wheel in {DIST}/; run `pixi run build` first", file=sys.stderr)
        return 1
    wheel = wheels[-1]
    # audiotensor-<version>-py3-none-any.whl
    wheel_version = wheel.name.split("-")[1]
    with open(ROOT / "pyproject.toml", "rb") as f:
        project_version = tomllib.load(f)["project"]["version"]
    if wheel_version != project_version:
        print(
            f"{wheel.name} has version {wheel_version} but pyproject.toml says {project_version}",
            file=sys.stderr,
        )
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        try:
            _install_and_check(wheel, tmp)
        except subprocess.CalledProcessError as e:
            print(
                f"smoke check failed: {e.cmd[:4]} exited with {e.returncode}",
                file=sys.stderr,
            )
            return 1
    print(
        f"{wheel.name} installs, imports from the wheel, and passes the end-to-end check"
    )
    return 0


def _install_and_check(wheel: Path, tmp: str) -> None:
    """Install ``wheel`` into ``tmp`` and run ``CHECK`` against it."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps"]
        + ["--target", tmp, str(wheel)],
        check=True,
    )
    # -I keeps the working directory off sys.path; the check itself puts the
    # target dir ahead of the editable install, so the wheel is what gets tested
    subprocess.run([sys.executable, "-I", "-c", CHECK, tmp], check=True, cwd=tmp)


if __name__ == "__main__":
    raise SystemExit(main())
