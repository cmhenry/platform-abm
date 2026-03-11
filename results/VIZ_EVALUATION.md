# Visualization Evaluation

**Status**: All 27 exp2 configs are now complete (including the formerly missing exp2_np27_rho015_alpha10). Key new numbers: mainstream utility on direct at {N_p=27, ρ_e=0.15, α=10} = **3.29** (deeply below the 5.0 random baseline); ρ_e=0.15 ANOVA interaction F=92.2, p<10⁻⁷¹.

Seven figures have been generated. Below is a figure-by-figure evaluation with impact ratings and recommended changes.

---

## B1: Interaction Heatmap — Normalized Mainstream Utility
**Impact: HIGH — Keep with minor revisions**

**Strengths**: Clean, readable layout. The viridis scale works well for the monotonic gradient. The three-panel structure makes the ρ_e progression immediately legible. Cell annotations are appropriately sized. The visual gradient from yellow (high utility, top-left of each panel) to purple (low utility, bottom-right) tells the story at a glance: more platforms and less parasitism = higher welfare.

**Issues**:
1. The color scale is *shared* across all three panels, which is correct and important — it lets the reader see that the entire panel shifts darker (lower utility) as ρ_e increases. Good design choice.
2. The new cell (N_p=27, ρ_e=0.15, α=10) is now populated at 0.609, completing the factorial. **The figure needs to be regenerated** to include this cell — the current PNG appears to already have it, so this may already be done. Verify.

**Recommended changes**:
- Add a thin border or bold font to the diagonal cells ({3,2}, {9,5}, {27,10}) to visually highlight the diversification-premium comparison.
- Consider adding small Δ annotations between the bottom row (N_p=3) and top row (N_p=27) to show the diversification premium directly on the figure. E.g., a small arrow or difference label: "Δ=0.098" between 0.511 and 0.609 at α=10/ρ_e=0.15.
- Panel title could use the actual ρ_e values with more space between panels for readability.

**Verdict**: This is a strong headline figure. With the Δ annotations it becomes a self-contained argument.

---

## B2: Escalation Slope Heatmap
**Impact: HIGH — Keep with revisions**

**Strengths**: The diverging RdBu scale is the right choice. The "ns" labels on the three non-significant cells (all at α=2, N_p=27) are informative. The dramatic gradient from near-white (top row) to deep red (bottom-left) vividly shows that low N_p + high α = runaway escalation.

**Issues**:
1. The N_p=3, ρ_e=0.05, α=10 cell is **grey** because the escalation slope was NaN (too few burst events to compute). This reads as "missing data" which is ambiguous. It should be explicitly labeled — something like "—" or "n/a" with a note that the burst rate was too low (0.163) for reliable slope estimation.
2. The color scale is symmetric (±15) but all meaningful values are positive. The blue half of the scale is wasted. Consider using a **sequential** red scale (white → deep red) since there are no negative escalation slopes in the data. This would give better visual resolution in the 0–3 range where most cells sit.
3. The new cell (N_p=27, ρ_e=0.15, α=10 = slope 1.39) should now be included. Verify.

**Recommended changes**:
- Switch to a sequential white-to-red scale.
- Replace the grey NaN cell with explicit "n/a" text and a neutral fill (light grey with explanation in caption).
- Add significance stars to significant cells (*** for p<0.001, ** for p<0.01) to reinforce the statistical story.

**Narrative note**: The escalation heatmap is the paper's strongest dynamics figure. Consider making it Figure 3 or 4 (early in the dynamics section) since it summarizes the entire escalation story in one panel.

---

## B3: Governance Utility Divergence
**Impact: CRITICAL BUG — Must be regenerated**

**The figure is plotting the wrong metric.** Line 239 of generate_phase_b.R reads `avg_utility_gov_direct`, which is the average utility of *all* communities on direct platforms (mainstream + extremist combined). Because extremists have enormous utility (α × parasitism bonus), this inflates the direct-platform average, making it appear that direct platforms have the *highest* utility — the opposite of the intended story.

The actual data:
| Config (N_p=27, ρ_e=0.15) | α=2 | α=5 | α=10 |
|---|---|---|---|
| avg_utility_gov_direct (CURRENT, wrong) | 7.62 | 8.13 | **10.20** |
| avg_utility_mainstream_direct (CORRECT) | 7.01 | 5.65 | **3.29** |
| avg_utility_mainstream_coalition | 6.35 | 6.26 | 6.15 |
| avg_utility_mainstream_algorithmic | 6.41 | 6.34 | 6.20 |

The corrected figure should show direct utility **collapsing** from 7.01 to 3.29 — crashing below the 5.0 random baseline — while coalition and algorithmic utility barely moves (6.35→6.15 and 6.41→6.20). This is the "scissors" pattern: the governance types diverge as parasitism intensifies.

**Required fix**: Change the measure name in the R script from `avg_utility_gov_*` to `avg_utility_mainstream_*`:
```r
measure <- paste0("avg_utility_mainstream_", gov)
```

**When corrected, this becomes the paper's most impactful single figure.** The visual of the orange direct line plunging below the dashed "random baseline" at α=10 while coalition and algorithmic hold steady is a devastating illustration of how governance structure determines who bears the cost of extremism.

**Additional recommended changes once the metric is fixed**:
- Add a shaded region below y=5.0 (light red or grey) labeled "Below random assignment" to emphasize the damage zone.
- Consider adding the N_p=9 data at α=10 as well (mainstream direct utility = 3.91 there), perhaps as a secondary dashed line or an inset comparison.
- The y-axis label should be "Mainstream community utility" (not just "Mainstream utility") to be precise about who is being measured.

---

## B4: Burst Heatmap Grid
**Impact: MEDIUM — Needs visual fixes**

**Strengths**: Two-panel layout effectively shows both dimensions of burst activity (size and frequency). The ρ_e=0.15 slice is the right choice for the paper body (lower ρ_e values can go in supplementary).

**Issues**:
1. **Readability crisis on the N_p=3 row.** The magma color scale maps the highest values (burst size 66, burst rate 1.00) to near-black. The cell text is also black, making the numbers on the bottom row virtually unreadable. This is the row that carries the most dramatic numbers.
2. The N_p=27, α=10 cell now has the new data (previously missing). The burst heatmap needs regeneration too. Looking at the current figure: the N_p=27 burst data does seem present — but the *median burst size* of 0 for that cell is suspicious. Let me check: the burst_aggregate.json for np27_rho015_alpha10 shows median_burst_size=0, which may indicate the burst detection didn't find typical bursts (possible if the dynamics are continuous raiding rather than discrete bursts).
3. The two panels use different color scales (one for burst size, one for burst rate) which is correct, but the scales are both named "magma reversed," making dark = high for both. This creates a visual where both panels look similar, reducing the information density.

**Recommended changes**:
- **White text on dark cells.** Add conditional text coloring: if fill value > midpoint of scale, use white text; otherwise black.
- Consider using a different color palette for the burst rate panel (e.g., YlOrRd) to visually distinguish it from the burst size panel.
- Add a caption or subtitle noting that the N_p=3 row represents systems where burst events are so pervasive and large that they constitute near-continuous raiding.
- For the np27_rho015_alpha10 cell with median_burst_size=0: verify the burst detection. If correct (genuinely no discrete bursts because extremists are spread too thin across 27 platforms), this is actually an interesting finding — label it explicitly.

---

## B5: Enclave Trajectory
**Impact: HIGH — Keep with additions**

**Strengths**: This is the best-executed figure in the set. The grey individual-platform traces against the bold green mean line create a clear visual narrative: rapid convergence to high homogeneity, sustained stability, occasional disruptions. The 0.9 threshold line provides meaningful context. The config choice (N_p=27, ρ_e=0.15, α=5) is well-chosen — harsh enough to test the mechanism but not so extreme that we lose the signal.

**Issues**:
1. The late-game dips (around steps 70–100) are interesting — these could be extremist raid disruptions. The figure shows them but doesn't help the reader interpret them.
2. No comparison config. The figure shows that enclaves work at N_p=27/α=5, but doesn't show what happens when they fail (N_p=3/α=10).

**Recommended changes**:
- **Add a companion panel** showing the same plot for N_p=3, ρ_e=0.15, α=10 — the configuration where enclaves are weakest (homogeneity ~0.80, cycle_rate low). A two-panel figure (left: success case, right: failure case) would be far more powerful than either panel alone.
- Optionally annotate 1–2 of the late-game dips with arrows pointing to "raid disruption?" to seed the reader's interpretation and connect this figure to the burst dynamics discussion.
- The y-axis label "Coalition platform homogeneity" is correct but long. Consider "Type homogeneity" with explanation in the caption.

---

## B6: Superposed Epoch
**Impact: LOW AS CURRENTLY GENERATED — Needs fundamental rework**

**This is the most important figure in the analysis plan but the current version shows almost no signal.** The community counts and utility lines are essentially flat across the ±8 step window. The "raid event" at t=0 produces no visible perturbation.

**Diagnosis**: The problem is almost certainly methodological. The displacement_aggregate.json was likely computed from the *stepwise.csv* data, which contains *cross-iteration averages* at each simulation step. When you align burst events across 200 iterations (each with raids at different absolute steps), the signal washes out because the stepwise.csv already represents the mean trajectory — not per-iteration trajectories.

**The correct approach** requires:
1. For each iteration, identify burst events from per_iter_raiding.json or per_iter_burst_analysis.json
2. For each burst event, extract the per-step community counts and utility from that *specific iteration's* raw trajectory (not the cross-iteration average)
3. Align those per-event windows to t=0 and average across events

The data to do this exists — per_iter_raiding.json and per_iter_burst_analysis.json contain per-iteration burst step information, and raw.csv contains per-iteration summary data (though we may need per-step-per-iteration data, which might need to be regenerated from the simulation).

**Alternatively**: If per-step-per-iteration data isn't available, a simpler version could use the *flow.npz* matrices directly. These are per-step platform-to-platform flow matrices (already averaged across iterations). For each step, sum flows FROM direct platforms TO {algorithmic, coalition} platforms, and flows FROM {algorithmic, coalition} TO direct. Then align these to burst events detected in the stepwise.csv and plot the epoch.

**Recommended path forward**:
- If per-step-per-iteration data exists: rebuild the epoch from iteration-level trajectories.
- If not: use flow.npz to build a directional flow epoch (which would be even more informative — it shows WHERE communities go, not just how counts change).
- Consider reducing the window to ±3 or ±4 steps if the burst events are short-lived.
- If the epoch signal remains genuinely flat even with correct methodology, that itself is a finding worth discussing ("displacement events are absorbed within a single simulation step").

**Do not include this figure in the paper in its current form.** It adds no information and could undermine confidence in the burst analysis.

---

## B7: Extremist Concentration Bar Chart
**Impact: MEDIUM — Usable but consider redesign**

**Strengths**: Complete 9-panel factorial. The "equal share" reference line is helpful. The pattern is clear: orange (direct) bars dominate at ~50% across all configs while holding only 10–20% of total population.

**Issues**:
1. **Too busy.** Nine panels × 3 bars × 3 clusters = the reader's eye bounces everywhere. The main finding is simple (extremists overconcentrate on direct), and this layout obscures it.
2. **The denominator problem.** Showing "fraction of extremists" at 50% looks moderate. The real impact comes from comparing that to the fraction of *total population* on direct (~10–20%). The overrepresentation ratio (50% of extremists ÷ 10% of population = 5× overrepresentation) is the more compelling number.
3. The α variation across columns is minimal (the bars barely change), which makes the 9-panel layout seem like it's showing non-variation.

**Recommended redesign options** (pick one):
- **Option A: Overrepresentation ratio plot.** A single heatmap or grouped bar chart showing the ratio of extremist share to population share for each governance type. Direct would show 3–5× overrepresentation; coalition and algorithmic would be near or below 1×.
- **Option B: Simplified 3-panel version.** Collapse across α (since it barely matters) and show one panel per ρ_e, with just N_p on the x-axis. This reduces 9 panels to 3.
- **Option C: Table instead of figure.** Given the stability of the ~50% finding, this might be better communicated as a single sentence in the text with a supporting table in the appendix.

---

## Summary: Figure Disposition

| Figure | Status | Priority |
|--------|--------|----------|
| B1: Interaction heatmap | ✅ Keep — minor revisions (add Δ annotations) | Low |
| B2: Escalation heatmap | ✅ Keep — switch to sequential scale, fix NaN cell | Medium |
| B3: Governance divergence | 🔴 **CRITICAL FIX** — wrong metric, must regenerate | **Highest** |
| B4: Burst heatmap | ⚠️ Fix readability (white text on dark cells) | Medium |
| B5: Enclave trajectory | ✅ Keep — add failure-case companion panel | Medium |
| B6: Superposed epoch | 🔴 **Rework from scratch** — no visible signal | High |
| B7: Extremist concentration | ⚠️ Redesign as overrepresentation ratio or simplify | Low |

---

## Narrative Additions Suggested by the Visualizations

### 1. The 3.29 headline (from corrected B3)
The corrected governance divergence figure reveals the paper's most dramatic number: mainstream communities on direct platforms at {N_p=27, ρ_e=0.15, α=10} have utility of **3.29** — a full 1.7 points below random assignment and barely half the utility of their counterparts on coalition (6.15) or algorithmic (6.20) platforms. This isn't just underperformance; it's a governance failure so complete that the affected communities would be better off if sorting didn't exist at all. This deserves a dedicated paragraph in the results and a callback in the discussion.

### 2. The asymmetry between extremist and mainstream utility on direct platforms
The bug in B3 accidentally revealed an important finding: `avg_utility_gov_direct` (all residents) *rises* with α because extremists thrive on direct platforms as α increases. Meanwhile `avg_utility_mainstream_direct` collapses. This asymmetry — the same platform producing increasing utility for parasites and decreasing utility for hosts — is the essence of the parasitism dynamic. Consider adding a small inset or annotation to the corrected B3 showing extremist utility rising alongside the mainstream collapse. Or present this as a separate table in the text.

### 3. The enclave disruption-and-recovery cycle (from B5)
The late-game dips visible in the enclave trajectory (steps 70–100) suggest that even stable enclaves occasionally suffer disruptions. If these can be linked to burst events in the raiding data (from per_iter_raiding.json), this creates a rich micro-narrative: "Coalition enclaves form quickly, resist most raids, but occasionally suffer transient disruptions from which they recover within 2–3 steps." This bridges the static enclave finding (Result 5) and the dynamic raiding finding (Result 7) into a unified mechanism story.

### 4. The continuous-raiding regime at N_p=3 (from B4)
The near-black cells at N_p=3 in the burst heatmap (burst rate ≈ 1.0, median burst size 58–66) indicate that with only 3 platforms, the system is in **continuous raiding** — there are no stable periods between raids. This is qualitatively different from the intermittent burst pattern at N_p=9 and N_p=27. The narrative should name this: "At low platform diversity, the system transitions from intermittent raiding to continuous raiding, where extremist displacement becomes the norm rather than the exception."

### 5. Flow-based displacement direction (data available, not yet visualized)
The flow.npz files contain complete platform-to-platform transition matrices per step. These could reveal the *direction* of displacement after raids: do mainstream communities primarily flee from direct to algorithmic? To coalition? Is the direction different at different α levels? This would answer the question left open by the flat superposed epoch: even if the aggregate counts don't move much (because the system quickly rebalances), the underlying flow rates could be enormous. A Sankey diagram or flow-arrow figure showing mean directional flows at t=0 vs t=-5 would be a powerful addition.
