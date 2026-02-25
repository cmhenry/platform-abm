"""MiniTiebout model - the core simulation."""

from __future__ import annotations

import agentpy as ap

from platform_abm.agents.community import Community
from platform_abm.agents.platform import Platform
from platform_abm.config import (
    INSTITUTION_TYPE_COUNT,
    CommunityType,
    InstitutionType,
    Strategy,
)
from platform_abm.institutions.algorithmic import AlgorithmicInstitution
from platform_abm.metrics import (
    compute_average_moves,
    compute_average_utility,
    compute_extremist_metrics,
    compute_mixed_institution_metrics,
)
from platform_abm.utils import (
    generate_binary_preferences,
    generate_ones_preferences,
    generate_zero_preferences,
)


class MiniTiebout(ap.Model):
    """Tiebout model: communities choose between platforms based on policy preferences."""

    ### SETUP ###

    def setup(self) -> None:
        """Initialize the agents and network of the model."""
        if not hasattr(self, "tracker"):
            self.tracker = None
        self._last_n_relocations = 0
        self.communities = ap.AgentList(self, self.p.n_comms, Community)

        if self.p.extremists == "yes":
            extremists, _mainstream = self._mix_agents_by_percentage(
                self.communities, self.p.percent_extremists
            )
            self._setup_community_types(extremists)

        self.platforms = ap.AgentList(self, self.p.n_plats, Platform)

        if self.p.institution == InstitutionType.MIXED.value:
            sub_platforms = self._mix_agents_by_split(self.platforms, INSTITUTION_TYPE_COUNT)
            self._setup_platform_institutions(sub_platforms)
        else:
            institution_name = self.p.institution
            for plat in self.platforms:
                plat.set_institution(institution_name)

        self._setup_platform_policies()

        if self.p.extremists == "yes":
            extremists, _mainstream = self._mix_agents_by_percentage(
                self.platforms, self.p.percent_extremists
            )
            self._setup_platform_types(extremists)

        # Assign communities to platforms
        if getattr(self.p, 'initial_distribution', 'random') == 'equal':
            platform_list = list(self.platforms)
            for i, community in enumerate(self.communities):
                platform = platform_list[i % len(platform_list)]
                community.join_platform(platform)
                platform.add_community(community)
        else:
            for community in self.communities:
                platform = self.random.choice(self.platforms)
                community.join_platform(platform)
                platform.add_community(community)

        # Setup algorithmic platforms
        algo_inst = AlgorithmicInstitution()
        for platform in self.platforms:
            if platform.institution == InstitutionType.ALGORITHMIC.value:
                platform.grouped_communities = []
                algo_inst.group_communities(platform)
                algo_inst.rate_policies(platform)
                algo_inst.set_group_policies(platform)

        self.agents = self.communities + self.platforms

    def _mix_agents_by_split(self, agentlist: ap.AgentList, split: int) -> list[list[int]]:
        """Split agents into sublists of roughly equal size."""
        remaining = list(agentlist.id)
        self.random.shuffle(remaining)
        total = len(remaining)
        sublist_size = total // split
        remainder = total % split

        sublists: list[list[int]] = []
        idx = 0
        for i in range(split):
            extra = 1 if i < remainder else 0
            chunk = sublist_size + extra
            sublists.append(remaining[idx : idx + chunk])
            idx += chunk
        return sublists

    def _mix_agents_by_percentage(
        self, agentlist: ap.AgentList, percentage: int
    ) -> tuple[list[int], list[int]]:
        """Select a percentage of agents randomly."""
        if percentage <= 0 or percentage > 100:
            raise ValueError("percentage must be between 0 and 100")
        all_ids = list(agentlist.id)
        num_items_to_select = int(len(agentlist) * (percentage / 100))
        selected = self.random.sample(all_ids, num_items_to_select)
        unselected = [item for item in all_ids if item not in selected]
        return selected, unselected

    def _setup_platform_institutions(self, sub_platforms: list[list[int]]) -> None:
        """Assign institution types to platform sublists."""
        instlist = [
            InstitutionType.ALGORITHMIC.value,
            InstitutionType.DIRECT.value,
            InstitutionType.COALITION.value,
        ]
        for sublist_idx, sublist in enumerate(sub_platforms):
            for item_id in sublist:
                plat_sel = self.platforms.select(self.platforms.id == item_id)
                inst_name = instlist[sublist_idx]
                for plat in plat_sel:
                    plat.set_institution(inst_name)

    def _setup_platform_types(self, sub_platforms: list[int]) -> None:
        """Set extremist platform policies to all-zeros."""
        for plat_id in sub_platforms:
            plat_sel = self.platforms.select(self.platforms.id == plat_id)
            for plat in plat_sel:
                if plat.institution != InstitutionType.ALGORITHMIC.value:
                    plat.policies = generate_zero_preferences(self.p.p_space)

    def _setup_platform_policies(self) -> None:
        """Set initial platform policies."""
        for platform in self.platforms:
            if platform.institution != InstitutionType.ALGORITHMIC.value:
                platform.policies = generate_binary_preferences(self.random, self.p.p_space)
            else:
                platform.policies = platform.institution_strategy.cold_start_policies(platform)

    def _setup_community_types(self, extremists: list[int]) -> None:
        """Set extremist community types and preferences (randomly all-zeros or all-ones)."""
        for comm_id in extremists:
            comm_sel = self.communities.select(self.communities.id == comm_id)
            comm_sel.type = CommunityType.EXTREMIST.value
            if self.random.random() < 0.5:
                comm_sel.preferences = generate_zero_preferences(self.p.p_space)
            else:
                comm_sel.preferences = generate_ones_preferences(self.p.p_space)

    ### UPDATE ###

    def update(self) -> None:
        """Record variables after setup and each step."""
        if self.p.stop_condition == "satisficed":
            if self._check_satisficed():
                self.end()

    def _check_satisficed(self) -> bool:
        """Check if all communities are satisfied (no movers)."""
        for community in self.communities:
            if community.strategy == Strategy.MOVE.value:
                return False
        return True

    ### STEP ###

    def step(self) -> None:
        """Define the model's events per simulation step.

        Order: elections → utility → relocation.
        Elections run first so governance state is fresh before utility computation.
        """
        self._step_elections()
        if self.tracker is not None:
            self.tracker.record_governance_state(self.t, self.platforms)
        self._step_update_utility()
        self._step_relocation()

        if self.tracker is not None:
            self.tracker.record_step_metrics(
                self.t, self.communities, self._last_n_relocations
            )

    def _step_elections(self) -> None:
        """Hold elections on all platforms."""
        for platform in self.platforms:
            platform.election()

    def _step_update_utility(self) -> None:
        """Update all community agent utilities."""
        for community in self.communities:
            community.update_utility()
            community.set_strategy()

    def _step_relocation(self) -> None:
        """Batch-relocate communities that want to move.

        Two-phase approach: collect all moves first, then execute them.
        This ensures processing order doesn't affect outcomes.
        """
        moves = []
        for community in self.communities:
            if community.strategy == Strategy.MOVE.value:
                community.find_new_platform()
                new_platform = self.random.choice(community.candidates)
                moves.append((community, community.platform, new_platform))

        for community, old_platform, new_platform in moves:
            old_platform.rm_community(community)
            community.join_platform(new_platform)
            community.last_move_step = self.t
            new_platform.add_community(community)

        self._last_n_relocations = len(moves)

        if self.tracker is not None:
            self.tracker.record_step(self.t, moves)

    ### END ###

    def end(self) -> None:
        """Report final metrics."""
        self.report(
            "average_moves",
            compute_average_moves(self.communities, self.p.n_comms),
        )
        self.report(
            "average_utility",
            compute_average_utility(self.communities, self.p.n_comms),
        )

        if self.p.institution == InstitutionType.MIXED.value:
            mixed_metrics = compute_mixed_institution_metrics(
                self.communities, self.platforms, self.p.n_comms
            )
            for key, value in mixed_metrics.items():
                self.report(key, value)

        if self.p.extremists == "yes":
            extremist_metrics = compute_extremist_metrics(self.communities)
            for key, value in extremist_metrics.items():
                self.report(key, value)
