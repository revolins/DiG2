from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class UnifiedMetricContext:
    """Per-forward-pass shared context for mixed metric evaluation.

    Lightweight holder for the prefiltered generated records, shared by the synthetic
    and molecular evaluators operating over the same generated batch.
    """

    record_by_id: Dict[int, Dict[str, Any]] = field(default_factory=dict)
