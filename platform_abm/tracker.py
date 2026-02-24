"""Relocation tracking infrastructure for movement analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from platform_abm.agents.community import Community
    from platform_abm.agents.platform import Platform


@dataclass
class RelocationEvent:
    """A single community relocation between platforms."""

    community_id: int
    community_type: str
    from_platform_id: int
    to_platform_id: int
    from_institution: str
    to_institution: str


@dataclass
class GovernanceSnapshot:
    """Snapshot of a platform's governance state at a given step."""

    platform_id: int
    institution: str
    coalition_votes: list[int | None] = field(default_factory=list)
    winning_coalition_index: int | None = None
    community_order: list[int] = field(default_factory=list)
    group_membership: dict[int, int | str] = field(default_factory=dict)


@dataclass
class StepRecord:
    """All tracked data for a single simulation step."""

    step: int
    relocations: list[RelocationEvent] = field(default_factory=list)
    governance: list[GovernanceSnapshot] = field(default_factory=list)


class RelocationTracker:
    """Records relocation events and governance snapshots per step."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._log: dict[int, StepRecord] = {}

    def record_step(
        self,
        step: int,
        relocations: list[tuple[Any, Any, Any]],
    ) -> None:
        """Record relocation events for a step.

        Args:
            step: The simulation step number.
            relocations: List of (community, old_platform, new_platform) tuples.
        """
        if not self.enabled:
            return

        events = [
            RelocationEvent(
                community_id=comm.id,
                community_type=comm.type,
                from_platform_id=old_plat.id,
                to_platform_id=new_plat.id,
                from_institution=old_plat.institution,
                to_institution=new_plat.institution,
            )
            for comm, old_plat, new_plat in relocations
        ]

        if step in self._log:
            self._log[step].relocations = events
        else:
            self._log[step] = StepRecord(step=step, relocations=events)

    def record_governance_state(
        self,
        step: int,
        platforms: Any,
    ) -> None:
        """Snapshot governance state for all platforms at this step.

        Captures coalition votes/winner for coalition platforms and
        community→group mapping for algorithmic platforms.
        """
        if not self.enabled:
            return

        snapshots: list[GovernanceSnapshot] = []
        for plat in platforms:
            snapshot = GovernanceSnapshot(
                platform_id=plat.id,
                institution=plat.institution,
            )

            if plat.institution == "coalition":
                snapshot.coalition_votes = list(
                    plat.coalition_votes if plat.coalition_votes else []
                )
                snapshot.winning_coalition_index = getattr(
                    plat, "winning_coalition_index", None
                )
                snapshot.community_order = [c.id for c in plat.communities]

            elif plat.institution == "algorithmic":
                snapshot.group_membership = {
                    c.id: c.group for c in plat.communities
                }

            snapshots.append(snapshot)

        if step in self._log:
            self._log[step].governance = snapshots
        else:
            self._log[step] = StepRecord(step=step, governance=snapshots)

    def get_log(self) -> dict[int, StepRecord]:
        """Return the full step-keyed log."""
        return self._log

    def get_all_relocations(self) -> list[RelocationEvent]:
        """Return a flat list of all relocation events across all steps."""
        events: list[RelocationEvent] = []
        for step in sorted(self._log):
            events.extend(self._log[step].relocations)
        return events
