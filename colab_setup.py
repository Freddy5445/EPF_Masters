"""
Get this project running on Google Colab.

    !git clone https://github.com/Freddy5445/EPF_Masters.git
    %cd EPF_Masters
    !python colab_setup.py

or, from a notebook cell, ``import colab_setup; colab_setup.bootstrap()``.

What it does: installs the handful of dependencies Colab lacks, installs the
vendored epftoolbox from the local source tree (never from PyPI -- see
SETUP.md), and reports what compute is actually available.

It deliberately does *not* install requirements.txt. That file is a full
transitive pin of the Windows Python 3.11 environment; applying it on Colab
downgrades numpy, pandas and tensorflow out from under the preinstalled CUDA
wiring and usually leaves the runtime unable to see the GPU at all.
"""

import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def in_colab():
    return "google.colab" in sys.modules or os.path.isdir("/content")


def _run(args, label, required=True):
    print(f"-- {label}")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        if not required:
            return result
        print(result.stdout[-2000:])
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"{label} failed (exit {result.returncode})")
    return result


def ensure_epftoolbox():
    """Make ``import epftoolbox`` work, without installing it.

    The vendored tree does not need to be on the path as an installed package:
    every module in lear_dk1/ and dnn_dk1/ inserts ``<root>/epftoolbox`` into
    sys.path before importing from it, which is why the test suite runs with no
    epftoolbox installed at all. Doing the same here means a fresh notebook cell
    can ``import epftoolbox`` directly too.

    Returns the file the import resolved to, so the caller can show which copy is
    in use -- the vendored source, not the older PyPI release.
    """
    root = os.path.join(THIS_DIR, "epftoolbox")
    if root not in sys.path:
        sys.path.insert(0, root)
    import epftoolbox
    return epftoolbox.__file__


def install(quiet=True):
    """Install the dependencies Colab does not already ship."""
    pip = [sys.executable, "-m", "pip", "install"] + (["-q"] if quiet else [])
    _run(pip + ["-r", os.path.join(THIS_DIR, "requirements-colab.txt")],
         "dependencies")

    # epftoolbox comes from the local tree, never PyPI: the published release is
    # older than the vendored source and the models differ.
    #
    # --ignore-requires-python is not optional. setup.py declares
    # python_requires='>=3.9, <=3.13', and under PEP 440 "<=3.13" excludes every
    # 3.13 patch release after 3.13.0 -- so on Colab's 3.13.15 pip refuses with
    #
    #   ERROR: Package 'epftoolbox' requires a different Python:
    #          3.13.15 not in '<=3.13,>=3.9'
    #
    # The bound is a packaging mistake, not a real incompatibility, and the
    # epftoolbox tree is not ours to edit.
    result = _run(pip + ["--no-deps", "--ignore-requires-python",
                         os.path.join(THIS_DIR, "epftoolbox")],
                  "epftoolbox", required=False)
    if result.returncode != 0:
        # Not fatal: the install is a convenience, not a requirement.
        print("   pip could not install it; using the source tree directly "
              "instead, which is how the test suite runs anyway.")


def describe_compute():
    """Print what TensorFlow can actually use, and say so plainly if it is CPU."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.python.platform import build_info

    gpus = tf.config.list_physical_devices("GPU")
    print(f"\nTensorFlow {tf.__version__}")
    print(f"  built for CUDA {build_info.build_info.get('cuda_version')}, "
          f"compute capabilities "
          f"{build_info.build_info.get('cuda_compute_capabilities')}")

    if not gpus:
        print("  GPU: none visible -- running on CPU.")
        print("  In Colab: Runtime > Change runtime type > GPU.")
        return False

    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        name = details.get("device_name", "unknown")
        cc = details.get("compute_capability")
        print(f"  GPU: {name}" + (f" (sm_{cc[0]}{cc[1]})" if cc else ""))

        built = build_info.build_info.get("cuda_compute_capabilities") or []
        if cc and not any(f"sm_{cc[0]}{cc[1]}" in str(b) for b in built):
            print(f"       note: this build ships no sm_{cc[0]}{cc[1]} binaries, so "
                  f"kernels must be JIT-compiled from PTX. Expect a slow first "
                  f"call, and watch for cuDNN errors.")
    return True


def mount_drive(path="/content/drive"):
    """Mount Google Drive, so datasets and results outlive the runtime.

    Colab runtimes are wiped when they disconnect. Anything not on Drive --
    a 14-hour hyperparameter search, in particular -- is gone with them.
    """
    if not in_colab():
        print("Not on Colab; skipping Drive mount.")
        return None
    if os.path.ismount(path):
        return path
    from google.colab import drive
    drive.mount(path)
    return path


def bootstrap(mount=True, quiet=True):
    """Install everything and report the compute. Returns True if a GPU is usable."""
    if not in_colab():
        print("Not running on Colab. Nothing to bootstrap.")
    else:
        install(quiet=quiet)

    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)

    print(f"-- epftoolbox: {ensure_epftoolbox()}")

    if mount:
        try:
            mount_drive()
        except Exception as exc:  # noqa: BLE001 - a failed mount must not stop setup
            print(f"Drive not mounted ({type(exc).__name__}: {exc}). "
                  f"Results will be lost when the runtime disconnects.")

    return describe_compute()


if __name__ == "__main__":
    bootstrap()
