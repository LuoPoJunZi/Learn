"""Inspect interpreter settings and omitted-vs-None semantics without changing files."""

from __future__ import annotations

import builtins
import locale
import sys


MISSING = object()


def describe_update(value: object = MISSING) -> str:
    if value is MISSING:
        return "keep current value"
    if value is None:
        return "clear current value"
    return f"set value: {value}"


def main() -> int:
    print("Python:", sys.version.split()[0])
    print("Executable:", sys.executable)
    print("UTF-8 mode:", sys.flags.utf8_mode)
    print("Default text encoding:", locale.getpreferredencoding(False))
    print("Filesystem encoding:", sys.getfilesystemencoding())
    for name in ("sentinel", "frozendict"):
        print(f"Builtin {name}:", hasattr(builtins, name))
    print(describe_update())
    print(describe_update(None))
    print(describe_update("Ada"))
    print("No locale settings, files, or interpreter installations were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
