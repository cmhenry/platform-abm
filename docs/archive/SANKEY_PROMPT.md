# Claude Code Prompt: Governance Flow Alluvial Diagram

## Task

Create a new R script `results/viz/generate_alluvial.R` that produces an alluvial (Sankey-style) diagram showing the flow of communities between governance types over the course of a simulation.

## Critical Data Note: Do NOT Use flow.npz

The `flow.npz` files in each config's `dynamics/` folder contain platform-to-platform flow matrices averaged across iterations. **These are unusable for governance-level analysis** because platform governance assignment is randomized per iteration — the model shuffles platform IDs and splits them into thirds (algorithmic/direct/coalition). Platform index 3 might be "direct" in iteration 0 but "coalition" in iteration 47. Averaging raw matrices across iterations produces meaningless numbers.

Instead, use **`step_metrics.json`**, which records per-governance community counts at every step of every iteration.

## Data Source

**File**: `results/exp2/{config}/step_metrics.json`

Structure: JSON object keyed by iteration number (string "0" through "199"). Each iteration contains a list of 100 step records. Each step record has:

```json
{
  "step": 51,
  "avg_utility": 6.49,
  "n_relocations": 572,
  "per_governance_utilities": {"algorithmic": 6.28, "coalition": 5.41, "direct": 8.68},
  "per_governance_community_count": {"algorithmic": 598, "coalition": 165, "direct": 137},
  "per_type_utility": {"mainstream": 5.63, "extremist": 11.35},
  "per_type_relocations": {"mainstream": 572, "extremist": 0}
}
```

The key field is `per_governance_community_count` — the total communities on each governance type at each step.

## Configs to Show

Produce three alluvial diagrams (or a single figure with three panels) for:

1. **N_p=27, ρ_e=0.15, α=5** (`exp2_np27_rho015_alpha5`) — diversified, moderate parasitism
2. **N_p=9, ρ_e=0.15, α=10** (`exp2_np9_rho015_alpha10`) — moderate diversity, strong parasitism
3. **N_p=3, ρ_e=0.15, α=10** (`exp2_np3_rho015_alpha10`) — concentrated, strong parasitism

These are the same three configs used in the superposed epoch (B6), allowing direct comparison.

## Construction

### Step 1: Extract Governance Counts by Time Band

Divide the 100 simulation steps into **5 time bands**: steps 1–20, 21–40, 41–60, 61–80, 81–100.

For each config:
1. Read step_metrics.json
2. For each iteration (200 total), for each time band, extract the governance community counts at the **last step** of the band (steps 20, 40, 60, 80, 100). This gives the "state" at each time boundary.
3. Also extract counts at step 1 (the initial state) for the left-most column.
4. Average across all 200 iterations to get the mean community count per governance type at each of the 6 time boundaries (step 1, 20, 40, 60, 80, 100).

This yields a table like:
```
time_band | governance   | mean_count
step_1    | algorithmic  | 546
step_1    | coalition    | 178
step_1    | direct       | 176
step_20   | algorithmic  | 610
step_20   | coalition    | 170
step_20   | direct       | 120
...
```

### Step 2: Compute Net Flows Between Consecutive Time Bands

For each pair of consecutive time boundaries, compute the change per governance type:
```
Δ_algo = count_algo(t+1) - count_algo(t)
Δ_coal = count_coal(t+1) - count_coal(t)
Δ_dir  = count_dir(t+1)  - count_dir(t)
```

These must sum to ~0 (total communities is conserved at 900).

### Step 3: Produce Alluvial Diagram

Use `ggalluvial` (install if needed: `install.packages("ggalluvial")`).

The alluvial shows:
- **x-axis**: 6 time slices (columns/strata), labeled "Step 1", "Step 20", "Step 40", "Step 60", "Step 80", "Step 100"
- **y-axis (implicit)**: Community count (height of each stratum)
- **Strata (vertical blocks)**: Three blocks per time slice, one per governance type
- **Flows (ribbons)**: Connect governance blocks across time slices

Since we're tracking aggregate governance counts (not individual community trajectories), the alluvial shows the *stable portion* of each governance type as ribbons that stay within the same governance, and *net transfers* as ribbons that cross from one governance to another.

To construct the flow ribbons from net changes:
- If algorithmic gains +38 and direct loses -48 and coalition gains +10 between step 1 and step 20:
  - A ribbon of width 38 flows from direct → algorithmic
  - A ribbon of width 10 flows from direct → coalition
  - The remaining communities in each governance stay put (shown as same-governance ribbons)

When a governance type both gains from one source and loses to another in the same band, split accordingly. The constraint is that net flows must be consistent (total gains = total losses = 0).

### Color Palette

Use the standard governance colors:
- Direct: `#D55E00` (orange-red)
- Coalition: `#009E73` (green)
- Algorithmic: `#0072B2` (blue)

Ribbons should be colored by their **source** governance type (where communities came from), with alpha=0.5 for transparency.

## Figure Layout

**Three panels** arranged vertically (one per config), sharing the x-axis:

- **Top**: N_p=3, α=10 (continuous raiding — expect large, chaotic flows)
- **Middle**: N_p=9, α=10 (oscillatory — expect moderate, structured flows)
- **Bottom**: N_p=27, α=5 (graceful absorption — expect small, steady flows)

Use `patchwork` for layout: `(p1 / p2 / p3)`.

Panel subtitles (expression notation):
- `N[p] == 3, alpha == 10`
- `N[p] == 9, alpha == 10`
- `N[p] == 27, alpha == 5`

Overall title: `Community flows between governance types`
Overall subtitle: `ρ_e = 0.15, averaged across 200 iterations`

## Expected Visual Patterns

**N_p=27, α=5 (bottom — stable system)**:
- Three approximately stable blocks across all 6 time slices
- Minimal flow ribbons crossing between governance types
- Algorithmic block may grow very slightly; direct block may shrink very slightly
- Most ribbons are same-governance (communities staying put)

**N_p=9, α=10 (middle — active sorting)**:
- Clear net flow from direct → algorithmic visible as orange ribbons crossing to the blue block
- Coalition block relatively stable (flat green block)
- Direct block progressively shrinking; algorithmic block growing
- The flows should be moderate in size — visible but not dominant

**N_p=3, α=10 (top — continuous raiding)**:
- Large flow ribbons between direct and algorithmic (the raiding cycle)
- Possibly bidirectional flows in different time bands (communities bouncing between governance types)
- Coalition block may show more instability than in the other configs
- The visual should look "turbulent" compared to the calm bottom panel

## Output

Save as `fig_governance_alluvial.{pdf,png}`, width=10, height=12.

## Style

Use the same `theme_pub` as the other figures (theme_minimal, base_size=12, white background, grey90 grid). Same `save_fig` helper. The script should be self-contained with its own library imports and theme setup.

## Notes

- Total communities = 900 in all configs. The strata heights should be proportional to community count, so all time slices have the same total height.
- If `ggalluvial` proves awkward for this type of net-flow alluvial (it's designed for individual-level longitudinal data), consider using `ggforce::geom_parallel_sets()` or a manual ribbon approach with `geom_ribbon()` or `geom_polygon()`. The visual goal is: stacked bars at each time slice connected by flow ribbons. Use whatever ggplot2 extension achieves this most cleanly.
- An alternative approach: if the alluvial package is difficult, produce a **stacked area chart** with governance-colored bands showing the count evolution over all 100 steps (averaged across iterations), with faint vertical lines at the time band boundaries. This is simpler and may be more legible. In that case, overlay the net relocation count (`n_relocations` from step_metrics) as a secondary y-axis or a small panel below, to show system activity.
- The `per_type_relocations` field (mainstream vs extremist relocation counts) could be used to annotate each time band with the fraction of relocations that are extremist, giving a rough proxy for "who is moving" even though we can't decompose governance-level flows by type.
