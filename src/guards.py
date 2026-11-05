"""Compute guards for long local jobs (global constraints, Compute guards).

- stop when swap grows by more than 2 GB or free disk drops under 5 GB;
- `--max-hours`: a job stops itself on the limit and keeps what it has checkpointed;
- one model job at a time, by lock file;
- time a bootstrap at 100 replicates first and refuse to run the full one above 1 hour.

Readers (swap, disk, clock) are injectable so every guard can be tested with fake readings.
"""
import contextlib
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

MAX_SWAP_GROWTH_GB = 2.0
MIN_FREE_DISK_GB = 5.0
RUNS = Path(__file__).resolve().parents[1] / "runs"
LOCK_PATH = RUNS / "model_job.lock"


class JobStopped(Exception):
    """A job must stop now. `reason` is "max_hours" or "resources"."""

    reason = "stopped"


class ResourceLimitError(JobStopped):
    reason = "resources"


class MaxHoursExceeded(JobStopped):
    reason = "max_hours"


class JobLockedError(RuntimeError):
    """Another model job holds the lock."""


class BootstrapTooSlow(RuntimeError):
    """The projected full bootstrap is above the time limit."""


# --------------------------------------------------------------------------- readings


def parse_swap_used_gb(text: str) -> float:
    """Parse `sysctl -n vm.swapusage` ("total = 2048.00M  used = 1024.00M  free = ...") into GB used."""
    m = re.search(r"used\s*=\s*([\d.]+)\s*([KMGT])", text)
    if not m:
        raise ValueError(f"cannot parse swap usage from {text!r}")
    scale = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}[m.group(2)]
    return float(m.group(1)) * scale


def swap_used_gb() -> float:
    """Swap in use, in GB. macOS reads `sysctl vm.swapusage`; elsewhere /proc/meminfo, else 0."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=10, check=True)
        return parse_swap_used_gb(out.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        kb = lambda k: float(info[k].split()[0])  # noqa: E731
        return (kb("SwapTotal") - kb("SwapFree")) / 1024**2
    except (OSError, KeyError, ValueError):
        return 0.0


def free_disk_gb(path: str | os.PathLike = ".") -> float:
    return shutil.disk_usage(path).free / 1024**3


def check_resources(
    read_swap_gb: Callable[[], float], read_free_disk_gb: Callable[..., float], baseline: float, path: str = "."
) -> None:
    """Raise ResourceLimitError if swap grew by more than 2 GB over `baseline`, or free disk is under 5 GB."""
    grown = read_swap_gb() - baseline
    if grown > MAX_SWAP_GROWTH_GB:
        raise ResourceLimitError(f"swap grew by {grown:.2f} GB (limit {MAX_SWAP_GROWTH_GB:g} GB)")
    free = read_free_disk_gb(path)
    if free < MIN_FREE_DISK_GB:
        raise ResourceLimitError(f"free disk is {free:.2f} GB (minimum {MIN_FREE_DISK_GB:g} GB)")


def check_disk(read_free_disk_gb: Callable[..., float] = free_disk_gb, path: str = ".") -> None:
    """The disk half of check_resources, for use before a model download."""
    free = read_free_disk_gb(path)
    if free < MIN_FREE_DISK_GB:
        raise ResourceLimitError(f"free disk is {free:.2f} GB (minimum {MIN_FREE_DISK_GB:g} GB); not downloading")


# --------------------------------------------------------------------------- the per-job guard


class Guard:
    """Call it between units of work. It raises JobStopped on the time limit or a resource limit."""

    def __init__(
        self,
        max_hours: Optional[float] = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        read_swap_gb: Callable[[], float] = swap_used_gb,
        read_free_disk_gb: Callable[..., float] = free_disk_gb,
        path: str = ".",
    ):
        self.max_hours, self.clock, self.path = max_hours, clock, path
        self.read_swap_gb, self.read_free_disk_gb = read_swap_gb, read_free_disk_gb
        self.started = clock()
        self.baseline = read_swap_gb()

    @property
    def elapsed_hours(self) -> float:
        return (self.clock() - self.started) / 3600.0

    def __call__(self) -> None:
        if self.max_hours is not None and self.elapsed_hours >= self.max_hours:
            raise MaxHoursExceeded(f"reached --max-hours {self.max_hours:g}")
        check_resources(self.read_swap_gb, self.read_free_disk_gb, self.baseline, self.path)


@dataclass
class RunOutcome:
    completed: int
    stopped: Optional[str]  # None, "max_hours" or "resources"
    message: str = ""


def run_with_limit(
    items: Iterable,
    process: Callable,
    guard: Callable[[], None],
    checkpoint: Optional[Callable[[], None]] = None,
) -> RunOutcome:
    """Process items in order. The guard is checked before each one; on a stop or an error, `checkpoint`
    is called so what is done is kept. `process` should itself save after each unit (a resumable job).
    """
    completed = 0
    stopped, message = None, ""
    try:
        for item in items:
            guard()
            process(item)
            completed += 1
    except JobStopped as e:
        stopped, message = e.reason, str(e)
    finally:
        if checkpoint is not None:
            checkpoint()
    return RunOutcome(completed=completed, stopped=stopped, message=message)


# --------------------------------------------------------------------------- one model job at a time


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def model_job_lock(path: str | os.PathLike = LOCK_PATH):
    """Hold the single-model-job lock. A second job raises JobLockedError. A lock left by a dead process
    is reclaimed; one with an unreadable owner is refused rather than guessed at."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            owner = path.read_text().strip() if path.exists() else ""
            if attempt == 0 and owner.isdigit() and not _pid_alive(int(owner)):
                path.unlink(missing_ok=True)  # stale: its process is gone
                continue
            raise JobLockedError(f"another model job holds {path.name} (owner pid {owner or 'unknown'})") from None
    else:
        raise JobLockedError(f"could not take {path.name}")
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield path
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- bootstrap projection


@dataclass
class BootstrapProjection:
    n_replicates: int
    full_replicates: int
    seconds: float
    projected_hours: float


def project_bootstrap(
    fn: Callable[..., object],
    n_replicates: int = 100,
    full_replicates: int = 10_000,
    limit_hours: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
) -> BootstrapProjection:
    """Time `fn(n_replicates=...)`, project to `full_replicates`, and refuse above `limit_hours`.

    `fn` is any callable taking an `n_replicates` keyword; it is called once, at the small size.
    """
    t0 = clock()
    fn(n_replicates=n_replicates)
    seconds = clock() - t0
    hours = seconds * (full_replicates / n_replicates) / 3600.0
    proj = BootstrapProjection(n_replicates, full_replicates, seconds, hours)
    if hours > limit_hours:
        raise BootstrapTooSlow(
            f"{n_replicates} replicates took {seconds:.1f} s; {full_replicates} would take {hours:.2f} h, above the {limit_hours:g} h limit"
        )
    return proj
