"""Build an inventory of all (sched, tr, exp, bs) runs in the raw dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class InventoryEntry:
    sched: int
    tr: int
    exp: int
    bs: int
    path: Path  # path to the bs directory

    @property
    def run_id(self) -> str:
        return f"s{self.sched}_tr{self.tr}_e{self.exp}_bs{self.bs}"

    @property
    def slices_dir(self) -> Path:
        return self.path / f"slices_bs{self.bs}"

    @property
    def bs_csv(self) -> Path:
        return self.path / f"bs{self.bs}.csv"


def scan_inventory(root: str | Path) -> list[InventoryEntry]:
    """Walk the raw dataset root and yield every (sched, tr, exp, bs) directory."""
    root = Path(root).resolve()
    entries: list[InventoryEntry] = []
    base = root / "rome_static_medium"
    if not base.exists():
        raise FileNotFoundError(f"{base} does not exist. Did you extract the tarball?")

    for sched_dir in sorted(base.glob("sched*")):
        sched = int(sched_dir.name.replace("sched", ""))
        for tr_dir in sorted(sched_dir.glob("tr*")):
            tr = int(tr_dir.name.replace("tr", ""))
            for exp_dir in sorted(tr_dir.glob("exp*")):
                exp = int(exp_dir.name.replace("exp", ""))
                for bs_dir in sorted(exp_dir.glob("bs*")):
                    bs = int(bs_dir.name.replace("bs", ""))
                    entries.append(
                        InventoryEntry(sched=sched, tr=tr, exp=exp, bs=bs, path=bs_dir)
                    )
    log.info(
        "Scanned inventory: %d runs "
        "(schedulers=%d, training_configs=%d, experiments=%d, base_stations=%d)",
        len(entries),
        len({e.sched for e in entries}),
        len({e.tr for e in entries}),
        len({e.exp for e in entries}),
        len({e.bs for e in entries}),
    )
    return entries
