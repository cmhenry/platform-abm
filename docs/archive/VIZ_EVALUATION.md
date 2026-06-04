# Visualization Evaluation

**Last updated**: Post-fix reassessment (B3 metric corrected, B6 epoch baselined).

**Status**: All 27 exp2 configs complete. ANOVA interaction significant at all three ρ_e levels (F=31.7, 106.9, 92.2; all p < 10⁻²⁵). Key headline number: mainstream utility on direct at {N_p=27, ρ_e=0.15, α=10} = **3.29**.

---

## B1: Interaction Heatmap — Normalized Mainstream Utility
**Impact: HIGH — Publication-ready with optional polish**

All 27 cells populated. The viridis gradient from yellow (high utility, top-left) to purple (low utility, bottom-right) tells the diversification × parasitism story at a glance. Shared color scale across panels correctly shows the entire ρ_e=0.15 panel shifting darker. The complete factorial at ρ_e=0.15 now shows the full range: 0.642 (N_p=27, α=2) to 0.511 (N_p=3, α=10).

**Optional polish**: Δ annotations between N_p=3 and N_p=27 rows would make the diversification premium self-evident without requiring the reader to mentally subtract (e.g., "Δ=0.098" at α=10/ρ_e=0.15). Not blocking.

**Verdict**: Ready for paper. Strong headline figure.

---

## B2: Escalation Slope Heatmap
**Impact: HIGH — Publication-ready with minor fixes**

Complete with the new np27/rho015/alpha10 cell (slope=1.39). The gradient from near-white (N_p=27 row) to deep red (N_p=3, α=10 corner) is visually striking. Three "ns" cells (all α=2, N_p=27) correctly labeled. The one grey NaN cell (N_p=3, ρ_e=0.05, α=10) should be relabeled "n/a" rather than left ambiguous.

**Remaining issues**:
1. Grey NaN cell needs explicit "n/a" label with caption note.
2. The diverging RdBu scale wastes half its range on blues that never appear. A sequential white→red scale would give better visual resolution in the 0–3 range where 20 of 27 cells sit. Not blocking but would improve legibility.

**Verdict**: Usable now. The strongest single dynamics figure.

---

## B3: Governance Utility Divergence
**Impact: HIGHEST — Fixed and excellent** ✅

The metric fix (avg_utility_gov_* → avg_utility_mainstream_*) transformed this from misleading to the paper's most impactful figure. The corrected version shows exactly the "scissors" pattern the narrative needs:

- At N_p=3: all three governance types decline together (clustered between 4.6 and 6.1), with direct slightly below but not dramatically so.
- At N_p=9: direct separates from the pack, dropping to 3.9 at α=10 while coalition and algorithmic hold near 5.7–6.0.
- At N_p=27: the full scissors — direct plunges from 7.0 to **3.3**, crashing below the random baseline (dashed at 5.0), while coalition (6.15) and algorithmic (6.20) barely move.

The three-panel N_p progression is itself an argument: diversification amplifies governance differences. At low N_p, governance type barely matters. At high N_p, it determines whether your communities thrive or suffer worse-than-random outcomes.

**Remaining polish opportunities** (none blocking):
- A light shaded region below y=5.0 labeled "Below random assignment" would emphasize the damage zone.
- The y-axis could read "Mainstream community utility" for precision.
- Error bars are present but tiny at this scale — correct behavior given n=200.

**Verdict**: Paper-ready. Lead figure for Section 3.2.

---

## B4: Burst Heatmap Grid
**Impact: MEDIUM — Needs readability fix** ⚠️

The two-panel layout (burst size + burst rate at ρ_e=0.15) effectively shows both dimensions. The N_p=3 row carries the most dramatic numbers (burst rate ~1.0, median burst 58–66), but those numbers are **virtually unreadable** — black text on near-black cells. The N_p=27 row is clear (light cells, legible text). The N_p=9 row is borderline.

**Required fix**: Conditional text color — white text when the cell fill is dark (above scale midpoint), black text otherwise. This is a small R tweak to the geom_text() calls in generate_phase_b.R.

**Additional note**: The median_burst_size=0 for np27/rho015/alpha10 is a real data point (the burst detector found no discrete bursts at that config), which is itself interesting. The burst_rate for that cell is 0.645, meaning there is movement but it doesn't cluster into discrete burst events. The cell currently shows "0" which reads as "nothing happened" — might benefit from annotation.

**Verdict**: Fix the text color, then ready for paper.

---

## B5: Enclave Trajectory
**Impact: HIGH — Excellent, would be stronger as two-panel comparison**

The best-executed figure in the set. Grey individual-platform traces against the bold green mean line. Rapid convergence to ~1.0 by step 25. The 0.9 threshold is well-placed. Late-game dips (steps 70–100) show that even stable enclaves suffer occasional disruptions — these likely correspond to raid events and bridge the enclave story (Result 5) to the dynamics story (Result 7).

**Recommended addition**: A companion left panel showing N_p=3, ρ_e=0.15, α=10 — the configuration where enclaves are weakest (mean homogeneity ~0.80). The contrast between a successful enclave and a failing one, side by side, would make the diversification dependence of the coalition firewall immediately visible.

**Verdict**: Good as-is. Becomes great with the failure-case panel.

---

## B6: Superposed Epoch
**Impact: HIGH — Fixed and revealing** ✅

The baseline-normalization fix transformed this from a dead figure into a genuinely informative one. The 3×2 layout (3 configs × {count, utility}) shows three distinct displacement regimes:

**Bottom row (N_p=27, α=5 — diversified, moderate parasitism):**
- Left: Clean, small signal. Coalition dips ~1–2 at t=0, algorithmic rises symmetrically. Direct barely moves.
- Right: Utility steps up ~+0.02 at t=+1 and sustains through t=+8.
- Reading: The system absorbs raids gracefully. A small rebalancing, a modest welfare improvement.

**Middle row (N_p=9, α=10 — moderate diversity, strong parasitism):**
- Left: Coalition drops ~3.8 at t=0, algorithmic rises ~3.7. The signal is clean and clearly significant (SE bands separate from zero).
- Right: Utility jumps +0.045 at t=+1, oscillates between +0.035 and +0.050. Sustained improvement.
- Reading: Raids push communities from coalition to algorithmic. Mainstream welfare improves because the concentrated extremist presence is temporarily broken.

**Top row (N_p=3, α=10 — concentrated, strong parasitism):**
- Left: Wild oscillation — ±5 communities per step, no stable pre/post pattern. All three governance types swing erratically.
- Right: Utility swings ±0.12 per step. A +0.12 spike at t=+1, then -0.03 at t=+2, then +0.08 at t=+3. No recovery to baseline — continuous turbulence.
- Reading: This is the continuous-raiding regime. "Raid events" are not discrete disruptions but part of ongoing chaos. There is no stable state between events.

**Critical narrative revision**: The epoch tells a different directional story than RESULTS_NARRATIVE_MAP predicted. The map expected "extremists depart direct → arrive algorithmic → mainstream flees → some flow to coalition." What actually happens is that the displacement at t=0 is primarily **coalition losing communities to algorithmic** — not direct losing to algorithmic. Direct barely moves in system-level counts. This is because direct platforms are already small and extremist-dominated; the action happens in the rebalancing between the two larger governance pools. The utility jump occurs because the raid temporarily disperses concentrated extremists, improving mainstream welfare system-wide. See updated Result 8 framing in RESULTS_NARRATIVE_MAP.md.

**Remaining polish**: The figure is somewhat dense at 3×2. Consider whether the N_p=3 row (continuous raiding) is better described in text than shown, since its visual message is "chaos" which is harder to read than the clean signals in the other two rows. On balance, keeping all three rows is probably right — the contrast between clean signal and chaos IS the story.

**Verdict**: Paper-ready. The three-regime comparison across rows is powerful.

---

## B7: Extremist Concentration Bar Chart
**Impact: LOW-MEDIUM — Functional but too busy for its finding**

The 9-panel layout correctly shows ~50% of extremists on direct across all configs. But the visual complexity is disproportionate to the simplicity of the finding. The α variation across columns is minimal (bars barely change), making the 9-panel grid look like it's documenting non-variation.

**Planned redesign**: Two alternatives will be produced:
1. **Simplified 3-panel version** — collapse across α, one panel per ρ_e, N_p on x-axis.
2. **Overrepresentation ratio format** — show the ratio of extremist share to population share by governance type, making the 3–5× overrepresentation on direct visually immediate.

**Verdict**: Replace with one or both redesigned versions.

---

## Summary: Updated Figure Disposition

| Figure | Status | Action needed |
|--------|--------|---------------|
| B1: Interaction heatmap | ✅ Publication-ready | Optional Δ annotations |
| B2: Escalation heatmap | ✅ Publication-ready | Minor: relabel NaN cell, consider sequential scale |
| B3: Governance divergence | ✅ **Fixed — paper's strongest figure** | Optional below-baseline shading |
| B4: Burst heatmap | ⚠️ Readability fix needed | **White text on dark cells** (R tweak) |
| B5: Enclave trajectory | ✅ Good, upgrade planned | **Add failure-case companion panel** (Claude Code) |
| B6: Superposed epoch | ✅ **Fixed — three-regime comparison** | None required |
| B7: Extremist concentration | ⚠️ Too busy | **Simplify + overrepresentation ratio** (Claude Code) |

## Figures not yet produced (from VISUALIZATION_CONCEPTS)

| Concept | Status | Priority |
|---------|--------|----------|
| Platform biography (Viz 4.2) | Not started | **HIGH** — single most impactful addition |
| Alluvial/Sankey (Viz 4.5) | Not started | Medium — consider for revision |
| Network flow (Viz 4.6) | Not started | Low — save for talks |

---

## Narrative Additions Suggested by the Visualizations

### 1. The 3.29 headline (from corrected B3)
Mainstream communities on direct platforms at {N_p=27, ρ_e=0.15, α=10} have utility of **3.29** — 1.7 points below random assignment and barely half the utility of their counterparts on coalition (6.15) or algorithmic (6.20) platforms. Governance failure so complete that affected communities would be better off without sorting.

### 2. The parasitism asymmetry
The B3 bug accidentally revealed: `avg_utility_gov_direct` (all residents) *rises* with α because extremists thrive, while `avg_utility_mainstream_direct` collapses. The same platform simultaneously produces increasing utility for parasites and decreasing utility for hosts. Consider presenting this asymmetry as a table or annotation.

### 3. The displacement paradox (from corrected B6)
Raids produce a *system-wide welfare improvement* — mainstream utility jumps +0.02 to +0.045 after a raid event and sustains for 8+ steps. This is paradoxical: the event that triggers the displacement (an extremist burst) temporarily improves outcomes by dispersing concentrated parasites. The policy implication is subtle: the raiding cycle is costly in volatility but may be self-correcting in aggregate welfare.

### 4. Three displacement regimes (from B6)
The epoch reveals qualitatively distinct dynamics at each N_p level: graceful absorption (N_p=27), oscillatory recovery (N_p=9), continuous turbulence (N_p=3). These correspond to the intermittent vs. continuous raiding distinction identified in B4.

### 5. The enclave disruption-and-recovery cycle (from B5)
Late-game dips in the enclave trajectory suggest coalitions occasionally suffer transient disruptions from which they recover within 2–3 steps. If linkable to burst events, this bridges the static enclave finding (R5) and the dynamic raiding finding (R7).

### 6. The continuous-raiding regime (from B4 + B6)
At N_p=3, burst rate ≈ 1.0 and median burst size 58–66. The system is in continuous raiding — no stable periods between events. This is qualitatively different from intermittent bursting at N_p≥9 and deserves explicit naming in the narrative.
