"""Read the OpenRouter key from the macOS Keychain at run time.

The key is never printed, logged or passed as a command-line argument. `-w` makes `security`
print it to stdout, which is captured here and returned, and is the only place it appears.
"""
import getpass
import os
import subprocess
import sys

SERVICE = "jev-benchmark-openrouter"
# The full path, so a `security` earlier on PATH cannot be handed the job of reading the key.
SECURITY_PATH = "/usr/bin/security"
SECURITY_TIMEOUT_SECONDS = 60


class KeyUnavailable(RuntimeError):
    """The key could not be read. The message never holds the key or the tool's output."""


def keychain_command(user: str) -> list[str]:
    return [SECURITY_PATH, "find-generic-password", "-a", user, "-s", SERVICE, "-w"]


def _under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules or "_pytest" in sys.modules


def _read_with_runner(runner, user: str) -> str:
    """Run the command through `runner`. Tests pass a fake here; only read_key passes subprocess.run."""
    failure = None
    try:
        result = runner(keychain_command(user), capture_output=True, text=True, check=False, timeout=SECURITY_TIMEOUT_SECONDS)
    except OSError:
        failure = KeyUnavailable("could not run the macOS `security` tool")
    except subprocess.TimeoutExpired:
        failure = KeyUnavailable(f"`security` did not finish in {SECURITY_TIMEOUT_SECONDS} seconds")
    if failure is not None:
        # Raised out here: a TimeoutExpired can hold the tool's output, so it must not stay attached.
        raise failure
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
