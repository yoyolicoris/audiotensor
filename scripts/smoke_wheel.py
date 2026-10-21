"""Install the freshly built wheel into a throwaway venv and check it works.

``python -m build`` can succeed while producing an unusable wheel, and the
version is wired through three hops (git tag, ``hatch-vcs``, then
``importlib.metadata`` at import time), none of which the build itself
verifies. This installs the newest wheel in ``dist/`` WITH its dependencies
into a clean venv (CPU torch only), imports it there, compares
``__version__`` with the wheel filename, and runs a tiny end-to-end check.
Run after ``build``.

Version policy:

- Tag build (``GITHUB_REF_TYPE=tag``, i.e. release.yml): the version must equal
  the pushed tag without its leading ``v``, so a mismatch never reaches PyPI.
- Otherwise the checkout should have tags. A checkout without them yields a
  ``0.1.devN`` placeholder. That is an error only for a shallow clone (tags
  were probably not fetched: set ``fetch-depth: 0``). A full clone that has
  never been tagged, as before the first release, passes with a note.
"""

import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

DIST = Path("dist")

CHECK = """
import audiotensor
import torch
from audiotensor import AudioTensor, interpolation_mode

print(audiotensor.__version__)

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


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], check=False, capture_output=True, text=True)


def main() -> int:
    """Install the newest wheel in ``dist/`` and verify it."""
    wheels = sorted(DIST.glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        print(f"no wheel in {DIST}/; run `pixi run build` first", file=sys.stderr)
        return 1
    wheel = wheels[-1]
    # audiotensor-<version>-py3-none-any.whl
    expected = wheel.name.split("-")[1]

    with tempfile.TemporaryDirectory() as tmp:
        venv.create(tmp, with_pip=True)
        win = sys.platform == "win32"
        py = (
            Path(tmp)
            / ("Scripts" if win else "bin")
            / ("python.exe" if win else "python")
        )
        # CPU torch only: the default Linux wheel pulls ~2.5 GB of CUDA libraries
        cpu_index = "https://download.pytorch.org/whl/cpu"
        subprocess.run(
            [py, "-m", "pip", "install", "--quiet", "--extra-index-url", cpu_index]
            + [str(wheel)],
            check=True,
        )
        # -I keeps the working directory off sys.path: run from the repo root, a
        # plain `python -c` imports the source tree and never touches the wheel
        try:
            out = subprocess.run(
                [py, "-I", "-c", CHECK],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except subprocess.CalledProcessError as e:
            print(f"smoke check failed:\n{e.stdout}{e.stderr}", file=sys.stderr)
            return 1
        got = out.splitlines()[0]

    if got != expected:
        print(
            f"{wheel.name} declares version {expected} "
            f"but audiotensor.__version__ is {got}",
            file=sys.stderr,
        )
        return 1
    # In a tag build (release.yml) the wheel must carry exactly the pushed tag;
    # anything else would publish a version nobody tagged.
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        tag = os.environ["GITHUB_REF_NAME"]
        if got != tag.removeprefix("v"):
            print(
                f"version resolved to {got} but the pushed tag is {tag}",
                file=sys.stderr,
            )
            return 1
    elif _git("describe", "--tags").returncode:
        if _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true":
            print(
                f"version resolved to {got}, a hatch-vcs placeholder: the checkout "
                "is shallow and has no tags (is fetch-depth: 0 set?)",
                file=sys.stderr,
            )
            return 1
        print(
            f"note: version {got} is a pre-release placeholder because no release "
            "has been tagged yet"
        )
    print(f"{wheel.name} installs, imports, works, and reports audiotensor {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
