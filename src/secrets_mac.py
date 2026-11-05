"""Read the OpenRouter key from the macOS Keychain at run time.

The key is never printed, logged or passed as a command-line argument. `-w` makes `security`
print it to stdout, which is captured here and returned, and is the only place it appears.
"""
import getpass
import os
import subprocess
import sys

SERVICE = "jev-benchmark-openrouter"


class KeyUnavailable(RuntimeError):
    """The key could not be read. The message never holds the key or the tool's output."""


def keychain_command(user: str) -> list[str]:
    return ["security", "find-generic-password", "-a", user, "-s", SERVICE, "-w"]


def _under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules or "_pytest" in sys.modules


def _read_with_runner(runner, user: str) -> str:
    """Run the command through `runner`. Tests pass a fake here; only read_key passes subprocess.run."""
    try:
        result = runner(keychain_command(user), capture_output=True, text=True, check=False)
    except OSError:
        raise KeyUnavailable("could not run the macOS `security` tool") from None
    if result.returncode != 0:
        raise KeyUnavailable(f"no key in the Keychain under service {SERVICE}; run scripts/set_key.command")
    key = (result.stdout or "").strip()
    if not key:
        raise KeyUnavailable(f"the Keychain entry under service {SERVICE} is empty")
    return key


def read_key() -> str:
    if _under_pytest():
        raise KeyUnavailable("read_key refuses to run under pytest; tests use a fake transport")
    return _read_with_runner(subprocess.run, getpass.getuser())
