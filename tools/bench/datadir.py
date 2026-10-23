"""Print the data folder (``data=``, ``SD_DATA_DIR`` or ``config/ground.toml``).

For recipes whose tools run outside the workspace's Python (the training in
``tools/train`` has its own, with PyTorch): ``just`` asks here first and passes
``data=<folder>`` on. Every other key is ignored.
"""

from __future__ import annotations

import sys

from bench.common import BenchError, data_dir, warn


def main(argv: list[str] | None = None) -> int:
    args = dict(a.split("=", 1) for a in (sys.argv[1:] if argv is None else argv) if "=" in a)
    try:
        print(data_dir(args))
    except BenchError as exc:
        warn(str(exc))
        return exc.code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
