from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DecisionTrace:
    timestamp: datetime
    symbol: str
    features: Dict[str, Any]
    signal: str
    risk_checks: List[str]
    execution: Dict[str, Any]
    state_change: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def append_trace(path: str, trace: DecisionTrace) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace.to_dict(), default=str) + "\n")
