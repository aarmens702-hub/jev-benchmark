"""Append-only spend ledger: the code's limit that protects the budget.

Each paid attempt writes a reservation before it is sent and a settlement after it returns. A
crash in between leaves the reservation counted, because the call may have been billed, so a
restart never undercounts. Spend is read back from the file, so it persists across runs and
across stages.
"""
import fcntl
import json
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

HARD_CAP_USD = 7.15
STAGE_CAPS_USD = {"stage0": 0.50}
BASES = ("reported", "estimate", "not_billed")
TORN_MARK = b"#torn"
# Float sums of small costs drift by far less than this; it only stops 0.05 reading as over 0.05.
_EPS = 1e-12


class RunStop(Exception):
    """Base for anything that must end a run cleanly: finished rows kept, the rest marked not run."""


class BudgetExceeded(RunStop):
    pass


class LedgerCorrupt(Exception):
    """The file cannot be trusted, so spend is unknown. Fail closed; a person must look."""


def split_torn_tail(raw: bytes) -> tuple[list[bytes], bytes]:
    """Complete lines, and any unfinished last line left by a crash mid-write."""
    head, newline, tail = raw.rpartition(b"\n")
    lines = head.split(b"\n") if newline else []
    return [line for line in lines if line and not line.endswith(TORN_MARK)], tail


def heal_torn_tail(handle) -> None:
    """Close an unfinished last line so later appends start clean, without rewriting anything."""
    size = os.fstat(handle.fileno()).st_size
    if size:
        handle.seek(size - 1)
        if handle.read(1) != b"\n":
            os.write(handle.fileno(), TORN_MARK + b"\n")


def _check_amount(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("a cost must be a finite number of US dollars, zero or more")
    return float(value)


class Ledger:
    def __init__(self, path, cap_usd=HARD_CAP_USD, stage_cap_usd=None, *, stage=None):
        if cap_usd > HARD_CAP_USD:
            raise ValueError(f"cap_usd cannot be above the hard cap of US${HARD_CAP_USD}")
        fixed = STAGE_CAPS_USD.get(stage)
        if fixed is not None:
            if stage_cap_usd is not None and stage_cap_usd > fixed:
                raise ValueError(f"{stage} cannot be above its limit of US${fixed}")
            stage_cap_usd = fixed if stage_cap_usd is None else stage_cap_usd
        self.path = Path(path)
        self.cap_usd = cap_usd
        self.stage_cap_usd = stage_cap_usd
        self.stage = stage
        self._lock = threading.Lock()
        self._offset = 0
        self._settled = {"reported": 0.0, "estimate": 0.0}
        self._settled_stage = 0.0
        self._open: dict[str, tuple[float, str | None]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._synced():
            pass

    @contextmanager
    def _synced(self):
        """Hold the file exclusively with an up-to-date view, so check-then-append is atomic across processes."""
        with self._lock, open(self.path, "a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(self._offset)
            lines, tail = split_torn_tail(handle.read())
            for line in lines:
                self._apply(line)
            self._offset = os.fstat(handle.fileno()).st_size - len(tail)
            if tail:
                heal_torn_tail(handle)
                self._offset = os.fstat(handle.fileno()).st_size
            self._handle = handle
            yield

    def _apply(self, line: bytes) -> None:
        try:
            entry = json.loads(line)
            kind, entry_id = entry["kind"], entry["id"]
            if kind == "reserve":
                self._open[entry_id] = (_check_amount(entry["amount"]), entry["stage"])
            elif kind == "settle":
                del self._open[entry_id]
                self._add(_check_amount(entry["cost"]), entry["basis"], entry["stage"])
            elif kind == "charge":
                self._add(_check_amount(entry["cost"]), entry["basis"], entry["stage"])
            else:
                raise ValueError(kind)
        except (ValueError, KeyError, TypeError):
            raise LedgerCorrupt(f"unreadable entry in {self.path}; spend is unknown, so nothing may be sent") from None

    def _add(self, cost: float, basis: str, stage) -> None:
        if basis != "not_billed":
            self._settled[basis] += cost
            if stage == self.stage:
                self._settled_stage += cost

    def _append(self, entry: dict) -> None:
        entry = {"ts": time.time(), "stage": self.stage, **entry}
        line = json.dumps(entry, separators=(",", ":")).encode() + b"\n"
        os.write(self._handle.fileno(), line)
        os.fsync(self._handle.fileno())
        self._offset += len(line)
        self._apply(line.rstrip(b"\n"))

    def _spent(self) -> float:
        return sum(self._settled.values()) + sum(amount for amount, _ in self._open.values())

    def _stage_spent(self) -> float:
        return self._settled_stage + sum(amount for amount, stage in self._open.values() if stage == self.stage)

    def _refuse_if_over(self, extra: float, what: str) -> None:
        total, staged = self._spent() + extra, self._stage_spent() + extra
        if total > self.cap_usd + _EPS:
            raise BudgetExceeded(f"{what}: total US${total:.6f} would pass the cap of US${self.cap_usd}")
        if self.stage_cap_usd is not None and staged > self.stage_cap_usd + _EPS:
            raise BudgetExceeded(f"{what}: {self.stage} total US${staged:.6f} would pass its limit of US${self.stage_cap_usd}")

    def spent(self) -> float:
        """Everything counted against the cap: settled, plus reservations not settled yet."""
        with self._synced():
            return self._spent()

    def stage_spent(self) -> float:
        with self._synced():
            return self._stage_spent()

    def confirmed(self) -> float:
        """Spend the API itself reported."""
        with self._synced():
            return self._settled["reported"]

    def uncertain(self) -> float:
        """Spend counted to be safe that the API did not confirm: estimates and open reservations."""
        with self._synced():
            return self._settled["estimate"] + sum(amount for amount, _ in self._open.values())

    def remaining(self) -> float:
        with self._synced():
            room = self.cap_usd - self._spent()
            if self.stage_cap_usd is not None:
                room = min(room, self.stage_cap_usd - self._stage_spent())
            return room

    def projected(self, n_rows: int, per_row: float) -> float:
        """Projected total spend if n_rows more rows run at per_row. Raises if that does not fit."""
        extra = _check_amount(n_rows * per_row)
        with self._synced():
            self._refuse_if_over(extra, f"projected {n_rows} rows at US${per_row:.6f}")
            return self._spent() + extra

    def reserve(self, estimate: float, meta: dict) -> str:
        """Claim room before sending. Raises, writing nothing, if the call might pass a cap."""
        estimate = _check_amount(estimate)
        with self._synced():
            self._refuse_if_over(estimate, "refused before sending")
            entry_id = uuid.uuid4().hex
            self._append({"kind": "reserve", "id": entry_id, "amount": estimate, "meta": meta})
            return entry_id

    def settle(self, entry_id: str, cost: float, basis: str, meta: dict) -> None:
        """Replace a reservation with what happened. Never raises on the cap: the money is already spent."""
        cost = _check_amount(cost)
        if basis not in BASES:
            raise ValueError(f"unknown basis {basis}")
        with self._synced():
            self._append({"kind": "settle", "id": entry_id, "cost": cost, "basis": basis, "meta": meta})

    def charge(self, cost: float, meta: dict) -> None:
        """Record a reported cost. It is written first, then raises if that passed a cap."""
        cost = _check_amount(cost)
        with self._synced():
            self._append({"kind": "charge", "id": uuid.uuid4().hex, "cost": cost, "basis": "reported", "meta": meta})
            self._refuse_if_over(0.0, "after charge")
