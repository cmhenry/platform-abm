# Visualization Concepts: Raiding Dynamics and Mainstream Displacement

## Overview

The raiding cycle is the paper's most novel finding and the one that most benefits
from visualization. The challenge is showing a *system-level temporal process* —
communities moving between governance types in response to extremist behavior —
in a way that's legible in a static journal figure. Below are five visualization
concepts ranked by impact-to-effort ratio.

---

## 1. Superposed Epoch Plot (RECOMMENDED — primary figure)

**What it shows:** The "average raid" — what happens to the system in the steps
surrounding a typical burst departure from a direct platform.

**Construction:** Align all raid events to t=0 (the burst step). For each event,
extract governance-type community counts and utilities in a window of ±8 steps.
Average across all events. The displacement_diagnostic.py module produces this
data in the `superposed_epoch` output.

**Layout:** Two vertically stacked panels sharing the x-axis (relative time).

- **Top panel:** Community counts by governance type. Three lines: direct (red),
  algorithmic (blue), coalition (green). At t=0, direct drops sharply (the raid
  departure). Algorithmic rises (raid arrival). Coalition stays flat (enclave
  stability). The key visual: the *shape* of the algorithmic line after t=0 —
  does it stay elevated (extremists stick) or return to baseline (re-sorting)?

- **Bottom panel:** Mainstream utility by governance type. Same three lines.
  At t=0, we expect algorithmic utility to dip (cold-start disruption from
  arriving extremists) and then partially recover. Direct utility may *rise*
  temporarily (the parasitic extremists just left). Coalition flat.

**Shading:** Light confidence bands (±1 SE across events) around each line.
Vertical dashed line at t=0 labeled "Raid departure." Gray region for t<0
(pre-raid) vs white for t>0 (post-raid).

**Why this works:** It distills dozens of raid events into a single canonical
trajectory. The reader immediately sees: raids hurt algorithmic platforms
temporarily, relieve direct platforms temporarily, and don't touch coalition
platforms. The asymmetry between the three governance types is visible at a
glance. This is the quantitative version of the narrative in the raiding
dynamics subsection.

**Journal-friendly:** Two-panel figure, black-and-white compatible with
line patterns, standard matplotlib output. Works at single-column width.

---

## 2. Alluvial / Sankey Diagram (RECOMMENDED — secondary figure)

**What it shows:** The cumulative flow of communities between governance types
over the full simulation, separated by community type (mainstream vs extremist).

**Construction:** Divide the simulation into 4-5 time bands (e.g., steps 1-20,
21-40, 41-60, 61-80, 81-100). At each band boundary, show the distribution of
communities across governance types. Flows between bands show net movement.
Color-code flows by community type: mainstream in blue, extremist in red.

**Layout:** Horizontal axis is time (left to right). Three vertical stacks at
each time band: direct, coalition, algorithmic (top to bottom). Band width
proportional to community count. Flow ribbons connect bands, with width
proportional to the number of communities in each flow.

**Key visual features:**
- Thick red ribbons cycling between direct and algorithmic (the raiding loop)
- Thin blue ribbons flowing steadily toward algorithmic (mainstream sorting)
- Coalition stacks staying constant width with stable composition
- The direct stack thinning over time as mainstream communities evacuate
- Red ribbons getting wider in later time bands if raids escalate

**Challenge:** Standard Sankey diagrams show flows at a single point in time.
Showing repeated cycling requires either multiple Sankey panels (one per time
band) or a modified alluvial diagram with multiple vertical slices. The alluvial
format (like those produced by the R `ggalluvial` package or Python `plotly`)
handles this naturally.

**Implementation note:** The current tracking data gives us net counts per
governance type per step, not directional flows. The Sankey requires directional
flows, which we can approximate: if direct loses 40 extremists and algorithmic
gains 35 in the same step, the inferred flow is direct→algorithmic. This is
approximate but sufficient for visualization. For precise flows, the simulation
would need to log (source, destination, type) per relocation.

---

## 3. Heatmap Grid: Burst Amplitude × Interval Across the Factorial (RECOMMENDED)

**What it shows:** How raiding intensity varies across the 27 Experiment 2
configurations.

**Construction:** Two 3×3 heatmaps (one for burst amplitude, one for inter-burst
interval), each with N_p on the y-axis and α on the x-axis, at a fixed ρ_e
(or three panels for all three ρ_e levels).

**Layout:** Either a 2×3 grid (2 metrics × 3 ρ_e levels) or a 2×1 grid at
ρ_e=0.15 (the most extreme case). Cell color intensity proportional to the
metric value. Annotate cells with the actual number.

**Key visual features:**
- Burst amplitude should increase moving right (higher α) and possibly down
  (lower N_p, more concentrated targets)
- Inter-burst interval should decrease moving right (more frequent raids at
  higher parasitism)
- The bottom-right cell (N_p=3, α=10) should be the hottest — most intense
  and most frequent raids

**Why this works:** Compact summary of 27 configurations in a single figure.
Immediately shows the factorial structure and which factors drive raiding
intensity. Clean enough for a journal, informative enough for a talk.

**Data source:** burst_master.csv from the full pipeline rerun (pending).

---

## 4. Platform Biography (single iteration, illustrative)

**What it shows:** The full life story of one direct platform over 100 steps,
showing the accumulation-raid-recovery cycle in detail.

**Construction:** Pick the most active direct platform from a representative
iteration (e.g., platform 918 from the {27, 0.15, 10} run with 10 bursts).
Plot three series on the same time axis:

- **Stacked area:** Community count on this platform, split by type (mainstream
  below, extremist above). Shows the composition shifting as extremists accumulate
  and mainstream evacuates.
- **Overlay line:** Platform utility for mainstream communities on this platform.
  Shows the utility collapsing as extremist concentration rises.
- **Vertical markers:** Burst departure events (from burst_analysis), marked as
  red vertical lines with height proportional to burst size.

**Key visual features:**
- Sawtooth pattern in the extremist count: gradual accumulation punctuated by
  sharp drops (the raids)
- Inverse sawtooth in mainstream count: gradual decline as mainstream flees,
  brief recovery after extremists depart
- Utility line tracking the mainstream count — utility drops when extremists
  arrive, recovers when they leave
- The visual rhythm of the cycle visible across the full 100 steps

**Why this works:** Tells a concrete story. The reader can follow one platform
through the cycle and understand the mechanism viscerally. Complements the
superposed epoch (which averages across events) with a specific narrative.

**Limitation:** Single iteration, single platform. Risk of cherry-picking. Mitigate
by selecting the platform with the median burst count (not the most extreme) and
noting that the pattern is representative.

---

## 5. Network Flow Diagram (aspirational — high impact, high effort)

**What it shows:** The full system at a single snapshot or animated across time,
with platforms as nodes and community flows as directed edges.

**Construction:** Nodes are platforms, positioned by governance type (three
clusters or three columns). Node size proportional to community count. Edge
width proportional to the number of communities moving between platforms in
a given step (or averaged over a time window). Edge color by community type.

**Variants:**
- **Static snapshot:** Show the system at one step during a raid, with thick
  red arrows from direct→algorithmic showing extremist outflow.
- **Small multiples:** 4-6 panels showing the system at different phases of
  a raiding cycle (pre-raid, raid departure, post-arrival, re-sorting, next
  cycle, equilibrium).
- **Animation:** Full simulation as a GIF or video. Beautiful for talks,
  useless for print journals.

**Why this's powerful:** Network diagrams are the natural visual language for
inter-platform dynamics. The reader literally sees communities "voting with
their feet" across the platform ecosystem. The clustering by governance type
makes the structural argument visible — coalition platforms are stable islands,
direct platforms are volatile, algorithmic platforms are the contested middle.

**Why it's hard:** Requires directional flow data (source → destination per
community per step), which the current tracking doesn't produce. The
approximation from net changes works for 2-3 governance types but gets noisy
with 27 individual platforms. Would need either simplified (3-node: one per
governance type) or a simulation code change to log per-community movement
history.

**Recommendation:** Use the 3-node simplified version (one node per governance
type) for the journal figure. Save the 27-node animated version for talks.

---

## Recommended Figure Set for the Paper

Given journal space constraints (likely 6-8 figures total), the raiding dynamics
should get 2-3 figures:

1. **Figure N: Superposed Epoch Plot** — the primary evidence for mainstream
   displacement. Two panels: community counts and mainstream utility around
   the average raid event.

2. **Figure N+1: Platform Biography OR Alluvial Diagram** — the illustrative
   mechanism figure. Platform biography is easier to produce from existing data
   and tells a clearer single-platform story. Alluvial is more comprehensive
   but requires flow approximation.

3. **Figure N+2: Burst Heatmap** — the factorial summary. Only if the burst
   analysis pipeline produces the full cross-config data. Otherwise, report
   the numbers in a table.

The remaining figures should cover:
- Experiment 1: one table (mixed-system governance comparison by N_p)
- Experiment 2 cross-sectional: the N_p × α interaction table (R3), the
  governance-specific mainstream utility table (R5)
- Enclave dynamics: one panel showing coalition homogeneity over time
  (representative platform from enclaves.json)

---

## Implementation Priority

1. Superposed epoch — data already produced by displacement_diagnostic.py.
   Standard matplotlib, an afternoon of work.
2. Platform biography — data available from stepwise + burst_analysis for the
   single iteration we've examined. Standard matplotlib.
3. Burst heatmap — requires full pipeline rerun. Trivial to produce once
   burst_master.csv exists.
4. Alluvial — requires either flow approximation code or simulation change.
   Medium effort.
5. Network flow — requires simulation change for full version, or heavy
   approximation for simplified version. Save for revision or talks.
