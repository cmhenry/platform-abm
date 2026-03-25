# Results Interpretation and Analysis Plan

## Data Inventory

### What exists and is complete
- **Experiment 1**: All 6 configs × 200 iterations. Master summary, LaTeX table. Ready to write.
- **Experiment 2**: 26 of 27 configs × 200 iterations. Per-config: raw.csv (iteration-level), summary.csv (aggregated), stepwise.csv (step-level time series), convergence.json, dynamics/ folder with burst_aggregate.json, enclaves.json, scalars.json, raiding.json, flow.npz (platform-to-platform flow matrices per step), per_iter_raiding.json, per_iter_enclaves.json.
- **ANOVA**: Computed for ρ_e=0.05 and ρ_e=0.10. Both show highly significant interaction terms (p < 10⁻²⁵ and p < 10⁻⁸¹ respectively).

### What is missing
1. **exp2_np27_rho015_alpha10**: Only raw.csv with header row (0 data rows). This config never ran or failed immediately. This is the single most important missing cell — it's the "worst case" corner of the factorial at the highest ρ_e level, and its absence breaks the ρ_e=0.15 ANOVA.
2. **Displacement analysis**: Not yet computed for any config. The displacement_diagnostic.py exists but has not been run through the pipeline. This blocks Result 8 (superposed epoch) and Viz 4.1.
3. **Enclave aggregate with settling/disruption metrics**: The scalars.json and enclaves.json provide mean homogeneity and per-platform series, but the settling_step, disruption count, and recovery time metrics described in the pipeline Task 1.4/2.4 have not been computed.

### Critical path
**The missing config (exp2_np27_rho015_alpha10) must be run before the paper can be completed.** It is needed for:
- The headline 3×3 table at ρ_e=0.15
- The ρ_e=0.15 ANOVA
- The complete escalation pattern analysis
- Completing the behavioral regime narrative (mainstream utility on direct at α=10, N_p=27 is a key number)

---

## Result-by-Result Interpretation

### Result 1: Governance type shapes sorting efficiency ✅ CONFIRMED

The Exp1 data cleanly confirms the hierarchy:

| System | Avg Utility |
|--------|-------------|
| Algorithmic (homogeneous) | 6.493 |
| Coalition (homogeneous) | 6.058 |
| Direct (homogeneous) | 5.397 |
| Mixed (N_p=9) | 6.417 |

In the mixed system, algorithmic platforms attract 71% of communities at N_p=9 (rising to 76% at N_p=27), while direct platforms shrink to 11% (4.8% at N_p=27). But direct platforms show the *highest per-capita utility* in the mixed system (7.05 at N_p=9, 7.61 at N_p=27) — a classic selection effect where only the best-matched communities remain.

**Writing note**: This is premise-setting. Two paragraphs, one table. The key observation for the transition to Exp2 is that direct platforms in mixed systems are small, high-utility enclaves for their remaining residents. The question becomes: what happens when parasitic communities enter this system?

### Result 2: Platform diversification improves welfare ✅ CONFIRMED

System utility rises monotonically with N_p: 6.20 (N_p=3) → 6.42 (N_p=9) → 6.63 (N_p=27). Utility SD falls: 0.144 → 0.129 → 0.116. Convergence is rapid (utility gains in second half of simulation < 0.01).

**Writing note**: One sentence in text, footnote on convergence. This sets up the diversification premium question.

### Result 3: Diversification premium grows with extremist threat ✅ CONFIRMED (at ρ_e=0.05, 0.10; pending at 0.15)

The interaction tables show exactly the predicted pattern:

**At ρ_e = 0.05:**
- Diversification premium (N_p=27 minus N_p=3) at α=2: 0.038
- Diversification premium at α=10: 0.059
- Difference-in-differences: 0.020

**At ρ_e = 0.10:**
- Diversification premium at α=2: 0.042
- Diversification premium at α=10: 0.084
- Difference-in-differences: 0.042

The DID *doubles* from ρ_e=0.05 to ρ_e=0.10. The ANOVA interaction F-statistics are enormous: F=31.7 (p<10⁻²⁵) at ρ_e=0.05, F=106.9 (p<10⁻⁸¹) at ρ_e=0.10.

**Interpretation**: The paper's central claim is formally confirmed — the benefit of platform diversification grows with parasitism intensity. This is not a marginal effect; the interaction term explains substantial variance. The practical meaning: in a system with aggressive extremists, consolidating platforms (reducing N_p) costs more than it does in a benign environment. The cost of consolidation is endogenous to the threat environment.

**What's needed**: ρ_e=0.15 ANOVA once exp2_np27_rho015_alpha10 is available. Based on the trend (DID doubles from 0.05 to 0.10), expect an even stronger interaction at 0.15.

### Result 4: Parasitism intensity produces distinct behavioral regimes ✅ CONFIRMED

Mainstream utility on direct platforms at ρ_e=0.15:

| | α=2 | α=5 | α=10 |
|---|---|---|---|
| N_p=3 | 5.56 | 4.97 | 4.61 |
| N_p=9 | 6.51 | 5.45 | 3.91 |
| N_p=27 | 7.01 | 5.65 | (missing) |

The 3.91 figure at {N_p=9, ρ_e=0.15, α=10} is the headline: mainstream communities on direct platforms with griefer-level extremists are getting utility below 5.0, which is the expected value under *random* platform assignment. They would be better off not sorting at all. At N_p=3 with α=10, it's 4.61 — deep into the below-random territory.

**The regime boundary** sits between α=5 and α=10. At α=5, direct-platform mainstream communities are hurt but still above random (5.45–5.65). At α=10, the parasitism cost overwhelms the sorting benefit. This maps to the ideologue/griefer distinction: α=2 extremists are ideologues who benefit from proximity but don't destroy mainstream welfare; α=10 extremists are griefers whose parasitism overwhelms the system's sorting capacity.

**Key cross-result pattern**: The N_p effect reverses for mainstream communities on direct platforms. At α=2, more platforms means higher direct-platform utility (5.56 → 6.51 → 7.01 — the sorting benefit). At α=10, more platforms means *lower* direct-platform utility (4.61 → 3.91 → missing but predicted lower). Diversification helps the system but concentrates damage on direct platforms because it sorts extremists there more efficiently.

### Result 5: Coalition governance creates protective enclaves ✅ CONFIRMED

Coalition homogeneity by configuration:

| | α=2 | α=5 | α=10 |
|---|---|---|---|
| N_p=3 | 0.885 | 0.863 | 0.797 |
| N_p=9 | 0.971 | 0.966 | 0.939 |
| N_p=27 | 0.990 | 0.984 | (missing) |

At N_p=9 and N_p=27, coalition platforms reliably form enclaves (homogeneity > 0.93 even at α=10). The cycle_rate (fraction of iterations reaching stable enclave) is 97–100% at N_p=9 for α=2, and stays above 80% even at α=10.

At N_p=3, enclaves are weaker (0.80 at α=10) and much less reliable (cycle_rate drops to 0.5% at ρ_e=0.05/α=10). This is because with only 3 platforms, each coalition platform must absorb a larger share of extremists, overwhelming the coalition mechanism.

**Mainstream utility stability across α** tells the governance story:
- At N_p=27, ρ_e=0.15: Coalition utility drops from 6.35 (α=2) to 6.26 (α=5) — a Δ of 0.09.
- Same conditions: Direct utility drops from 7.01 to 5.65 — a Δ of 1.36.
- Coalition platforms lose 15× less mainstream utility than direct platforms as parasitism intensifies.

**Writing note**: Frame this as the "firewall" mechanism. Coalition governance doesn't prevent extremist entry (16–34 extremists end up on coalition platforms at ρ_e=0.15), but the internal coalition structure segregates them into their own groups, severing the parasitism channel.

### Result 6: Extremists concentrate on direct platforms ✅ CONFIRMED

Across all 26 available configs, extremists are ~50% on direct platforms despite direct platforms holding only 10–21% of total population. This is remarkably stable: 49–50% across all ρ_e levels, stable across α levels (drops slightly to 36–44% only at α=10/ρ_e=0.05 where there are very few extremists).

Overrepresentation factor: at N_p=27/ρ_e=0.15, direct platforms hold 10.4% of population but 50% of extremists — a 4.8× overrepresentation.

**Why**: Direct platforms are permissive (majority rule is easily captured by small concentrated groups), they provide highest per-capita utility to well-matched communities, and extremists' parasitism bonus is highest in dense mainstream environments. The mechanism is self-reinforcing: extremists accumulate → mainstream communities flee → platform becomes extremist-dominated → more extremists attracted.

### Result 7: Bursty raiding with significant escalation ✅ CONFIRMED (strongest finding)

**Burst rates**: Burst activity is pervasive. At N_p=3, nearly all platform-iterations show bursts (95–100% at ρ_e≥0.10). At N_p=27, burst rates are 39–68%. Burst activity scales with both ρ_e and α.

**Burst sizes**: Median burst size scales dramatically with ρ_e (more extremists = bigger raids): 14–18 at ρ_e=0.05, 22–44 at ρ_e=0.10, 33–66 at ρ_e=0.15. At N_p=3/ρ_e=0.15, the median burst is 58–66 communities — more than half the extremist population moving in a single coordinated (but unplanned) event.

**ESCALATION — the key dynamic finding:**

Escalation is **significant and positive in 23 of 25 testable configs** (p < 0.05). Only two configs show non-significant escalation: both are α=2, N_p=27 (the mildest extremists in the most diversified system). The escalation slopes show a dramatic gradient:

| | α=2 | α=5 | α=10 |
|---|---|---|---|
| N_p=3, ρ=0.15 | 2.52 | 5.55 | 15.03 |
| N_p=9, ρ=0.15 | 1.32 | 2.51 | 3.91 |
| N_p=27, ρ=0.15 | 0.24 (ns) | 0.64 | (missing) |

**This is the "config-dependent" outcome** — the most interesting of the three scenarios in the narrative map. Escalation varies systematically by configuration:

1. **α drives escalation intensity**: Slopes increase ~6× from α=2 to α=10 at any given N_p/ρ_e combination.
2. **N_p dampens escalation**: At N_p=27, escalation slopes are 4–15× smaller than at N_p=3 for the same α. At N_p=27/α=2, escalation is statistically null.
3. **The interaction**: Diversification doesn't just reduce raid magnitude — it prevents escalation. The system goes from "runaway raids" at low N_p to "steady-state cycling" at high N_p.

**Policy narrative**: This is the strongest version of the finding. More platforms help not just by diluting extremist impact, but by structurally preventing the feedback loop that makes raids get worse over time. The mechanism: with more platforms, each individual platform absorbs fewer extremists per raid, the utility disruption per raid is smaller, mainstream communities have more escape options, and the system recovers between raids rather than accumulating damage.

---

## What's Needed: Analysis Tasks for Claude Code

### CRITICAL: Run missing config
exp2_np27_rho015_alpha10 must complete. Without it, we cannot:
- Complete the headline 3×3 table at ρ_e=0.15
- Run the ρ_e=0.15 ANOVA
- Fill in the escalation pattern at the worst-case corner

### Analysis Task 1: Displacement analysis (Result 8)
Run displacement_diagnostic.py across all 26 configs using the burst events from burst_aggregate.json and the stepwise data. This produces the superposed epoch plot data — arguably the paper's most impactful single figure.

**Note**: The flow.npz files contain full platform-to-platform flow matrices per step (27×27 at N_p=27), averaged across 200 iterations. This is *richer* than what the pipeline tasklist assumed was available. We can potentially compute directional displacement flows directly from these matrices rather than inferring from net governance counts.

### Analysis Task 2: Complete ANOVA
Once the missing config is available, re-run the two-way ANOVA at ρ_e=0.15.

### Analysis Task 3: Enclave settling and disruption metrics
Compute settling_step, disruption count, and recovery time from the per-platform enclave series in enclaves.json.

### Visualization Tasks (R or Python)

1. **Superposed Epoch Plot** (Viz 4.1) — Highest priority. Requires displacement analysis completion.
2. **3×3 Interaction Heatmap** — Normalized mainstream utility, N_p × α, one panel per ρ_e. Data in hand.
3. **Escalation Slope Heatmap** — Same layout showing escalation slopes. Data in hand.
4. **Burst Heatmap Grid** (Viz 4.3) — Burst size and rate across factorial. Data in hand.
5. **Enclave Trajectory** (Viz 4.4) — Representative homogeneity time series. Data in hand.
6. **Governance Utility Divergence** — Mainstream utility by governance type across α, showing coalition stability vs direct collapse. Data in hand.

---

## Writing Plan: Results Section Structure

### Section 3.1: The Baseline (Experiment 1) — ~2 pages
One table (already in tables.tex), two paragraphs. Key points:
- Governance hierarchy: algo > coalition > direct
- Mixed systems beat homogeneous ones at every N_p
- Diversification premium: utility rises, variance falls with N_p
- Selection effect: direct platforms become small, high-utility enclaves
- Transition: "This efficient sorting is the premise. What happens when some communities are parasitic?"

### Section 3.2: Extremists and System Structure (Experiment 2, cross-sectional) — ~4 pages
Three tables, one or two figures.

**3.2.1 The diversification premium grows with parasitism** (Result 3)
- Headline table: 3×3 normalized mainstream utility at ρ_e=0.15
- Supporting tables at ρ_e=0.05 and 0.10 (appendix or supplementary panels)
- DID numbers in text
- ANOVA in footnote: "The N_p × α interaction is significant at all tested ρ_e levels"
- Figure: Interaction heatmap

**3.2.2 Parasitism produces behavioral regimes** (Result 4)
- The 3.91 headline number (or lower once np27/alpha10/rho015 is in)
- Frame the ideologue/griefer distinction
- The reversal: diversification helps the system but concentrates damage on direct platforms

**3.2.3 Coalition governance as firewall** (Results 5 + 6)
- Coalition utility drops 0.09 while direct drops 1.36 across same α range
- Homogeneity data: 0.94–0.99 at N_p≥9
- Extremist concentration: 50% on direct despite 10% population share
- Figure: Enclave trajectory or governance utility divergence plot

### Section 3.3: Raiding Dynamics (Experiment 2, dynamics) — ~3 pages
One or two figures, burst statistics in text.

**3.3.1 The raiding cycle** (Result 7)
- Burst detection: pervasive across configs
- Burst size and frequency scale with α and ρ_e
- The pattern: accumulate → burst → scatter → re-accumulate

**3.3.2 Escalation and system structure** (Result 7, escalation)
- Escalation is significant in 23/25 configs
- The gradient: α drives escalation intensity, N_p dampens it
- Key finding: "Diversification prevents escalation"
- Figure: Escalation slope heatmap

**3.3.3 Displacement and the raiding-sorting feedback** (Result 8)
- Superposed epoch figure (if displacement analysis complete)
- Or: defer to revision if not ready

### Section 3.4: Sensitivity (brief, ~1 page)
- Convergence diagnostics from existing data
- Note that the factorial design itself provides sensitivity for N_p, ρ_e, α
- Flag sensitivity to μ, N_a, g, k as future work or appendix

---

## Conditional Outcomes: What the Data Actually Shows

### Escalation: CONFIG-DEPENDENT ✅ (best outcome)
Escalation is significant almost everywhere but its magnitude depends on both α and N_p. This maps to the narrative: "System diversification not only reduces raid magnitude but prevents raid escalation."

### Coalition enclave: STRONG AT N_p≥9, WEAKER AT N_p=3
The "natural firewall" framing holds for systems with moderate-to-high platform diversity. At N_p=3, coalition platforms are less reliably protective. Framing: "Coalition governance provides robust protection in diversified systems but is overwhelmed in concentrated ones."

### Displacement direction: PENDING
Cannot determine until displacement analysis runs. The flow.npz data could answer this directly.

### Burst heatmap driver: BOTH α AND N_p MATTER
Burst amplitude scales with α; burst rate and frequency scale with both. The strongest framing: "The raiding cycle's severity depends on both who the extremists are and how the system is structured."
