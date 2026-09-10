"""Target registration — the shared record of what was found and where.

This is the project's *secondary contribution* (CLAUDE.md Phase 5.6). Detections
stream in from every agent's detector; the ``TargetRegister`` fuses them into a
deduplicated set of estimated target locations on the same shared map the drones
are exploring.

Deduplication matters because (a) one target is seen over many consecutive frames
as an agent flies past, and (b) two agents may see the same target. Without
fusion the register would hold hundreds of entries for a handful of real targets.

Fusion rule: a new detection within ``dedup_radius_m`` of an existing entry (same
class) *updates* that entry — a confidence-weighted running mean of position, max
confidence, union of contributing agents — otherwise it starts a new entry. This
is a lightweight online clustering; it needs no knowledge of the true targets
(that would be cheating — the register is what the system actually believes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.perception.detector import Detection


@dataclass
class RegisteredTarget:
    """One fused belief about a target: an estimated world position and the
    provenance needed for the §5 detection metrics."""

    id: int
    x: float
    y: float
    cls: str
    confidence: float          # best (max) confidence seen
    first_step: int            # step of the first contributing detection
    last_step: int             # step of the most recent contributing detection
    n_obs: int                 # number of detections fused into this entry
    agent_ids: set = field(default_factory=set)
    _weight: float = 0.0       # sum of confidences (for the weighted-mean update)


class TargetRegister:
    """Shared, deduplicated store of registered targets (all agents write here)."""

    def __init__(self, dedup_radius_m: float = 1.0):
        self.dedup_radius = float(dedup_radius_m)
        self._entries: list[RegisteredTarget] = []
        self._next_id = 0

    # ------------------------------------------------------------------ #
    def add(self, det: Detection) -> RegisteredTarget:
        """Fuse one detection into the register; return the affected entry."""
        match = self._nearest_same_class(det)
        if match is None:
            entry = RegisteredTarget(
                id=self._next_id, x=det.x, y=det.y, cls=det.cls,
                confidence=det.confidence, first_step=det.step, last_step=det.step,
                n_obs=1, agent_ids={det.agent_id}, _weight=max(det.confidence, 1e-6),
            )
            self._next_id += 1
            self._entries.append(entry)
            return entry

        # Confidence-weighted running mean of position (down-weights low-conf hits).
        w = max(det.confidence, 1e-6)
        new_w = match._weight + w
        match.x = (match.x * match._weight + det.x * w) / new_w
        match.y = (match.y * match._weight + det.y * w) / new_w
        match._weight = new_w
        match.confidence = max(match.confidence, det.confidence)
        match.first_step = min(match.first_step, det.step)
        match.last_step = max(match.last_step, det.step)
        match.n_obs += 1
        match.agent_ids.add(det.agent_id)
        return match

    def _nearest_same_class(self, det: Detection) -> RegisteredTarget | None:
        """Closest existing entry of the same class within ``dedup_radius``, else None."""
        best, best_d = None, self.dedup_radius
        for e in self._entries:
            if e.cls != det.cls:
                continue
            d = math.hypot(e.x - det.x, e.y - det.y)
            if d <= best_d:
                best, best_d = e, d
        return best

    # ------------------------------------------------------------------ #
    def entries(self) -> list[RegisteredTarget]:
        """All registered targets (live view; treat as read-only)."""
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)
