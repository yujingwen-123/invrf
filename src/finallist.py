from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    phase: str
    evla: float
    evlo: float
    evdp: float
    dist_deg: float
    baz: float
    rayp: float
    magnitude: float
    gaussian: float


def parse_finallist(path: str) -> List[EventRecord]:
    records: List[EventRecord] = []
    fpath = Path(path).expanduser().resolve()
    with fpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                records.append(
                    EventRecord(
                        event_id=parts[0],
                        phase=parts[1],
                        evla=float(parts[2]),
                        evlo=float(parts[3]),
                        evdp=float(parts[4]),
                        dist_deg=float(parts[5]),
                        baz=float(parts[6]),
                        rayp=float(parts[7]),
                        magnitude=float(parts[8]),
                        gaussian=float(parts[9]),
                    )
                )
            except ValueError:
                continue
    return records
