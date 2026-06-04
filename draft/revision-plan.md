# Structural Revision Plan: Empirics-First Reorientation
**Branch:** `revision/empirics-first`
**Date:** 2026-04-19

---

## Core Directive

Reorient the paper so that empirics and descriptives lead. Move model machinery to appendices. Explain the setup and mechanisms in plain language and basic mathematical notation. Sell the paper through theory and results, not the simulation build.

The paper currently reads as: *Here is my model. Here is how I built it. Here are the results.*

The revised paper should read as: *Here is the puzzle. Here is the pattern I find. Here is why the platform architecture produces it. The formal details are in the appendix.*

---

## Current Structure (990 lines)

```
§1 Introduction                           (~35 lines)
§2 Model                                  (~145 lines)
  §2.1 Agents and Environment
  §2.2 Utility Functions
  §2.3 Neighborhoods and Platform Governance
  §2.4 Platform Governance Mechanisms
    §2.4.1 Direct Voting
    §2.4.2 Coalition Formation
    §2.4.3 Algorithmic Recommendation
  §2.5 The Tiebout Cycle
  §2.6 Mapping to Real Platforms
§3 Experimental Design                    (~130 lines)
  §3.1 Outcome Measures
  §3.2 Experiment 1
  §3.3 Experiment 2
  §3.4 Sensitivity Analysis
§4 Results                                (~580 lines)
  §4.1 Experiment 1
    R1: Governance type shapes sorting efficiency
    R2: Platform diversification improves welfare
  §4.2 Experiment 2
    R3: Diversification premium grows with threat
    R4: Behavioral regimes (ideologue vs griefer)
    R4b: Mixed populations (exp2b)
    R5: Coalition governance / enclave formation
    R6: Extremists concentrate on direct platforms
  §4.3 Raiding Dynamics
    R7: Emergent raiding cycles
  §4.4 Summary
§5 Discussion                             (~85 lines)
  §5.1 Platform Design and Consolidation
  §5.2 Governance Design and Coalition Paradox
  §5.3 Extremist Strategy as System Property
  §5.4 Limitations
  §5.5 Conclusion
```

---

## Target Structure

```
§1 Introduction                           [REVISED — lead with findings]
§2 The Sorting Puzzle                     [NEW — empirics + descriptives]
§3 Framework                              [CONDENSED — language + light math]
§4 Results                                [REORDERED — lead with finding, not model]
  §4.1 Sorting Works (R1, R2 brief)
  §4.2 Diversification Protects (R3 — leads)
  §4.3 Governance Type Determines Vulnerability (R4/R5/R6 cluster)
  §4.4 Raiding Is Structural (R7 — leads)
  §4.5 Mixed Populations Confirm the Pattern (R4b — robustness register)
§5 Discussion                             [LIGHTLY REVISED]
§6 Conclusion                             [SEPARATED from Discussion]

Appendix A: Formal Model Specification    [= current §2 in full]
Appendix B: Experimental Design           [= current §3]
Appendix C: Sensitivity Analysis          [= current §3.4 + pending OAT results placeholder]
```

---

## Section-by-Section Instructions

### §1 Introduction [REVISED]

**Keep:** The empirical puzzle paragraph (extremist cycling documented). The Tiebout analogy paragraph. The mapping to real platform behaviors.

**Add at the start after the puzzle:** Three headline findings, stated plainly:
> "I find three things. First, platform diversification protects mainstream communities, and the premium grows with extremist threat severity. Second, coalition-based governance provides structural immunity through endogenous enclave formation — paradoxically, the governance type that underperforms in benign conditions becomes the strongest firewall under threat. Third, the concentration–raid–retreat cycle is an emergent structural property: no extremist community plans it, but the architecture of the platform ecosystem makes it the equilibrium outcome."

**Cut or footnote:** The lengthy "my argument is threefold" and "to demonstrate these dynamics, I develop" paragraphs — these preview the model at length rather than the findings. Condense to 2 sentences: "I adapt the Tiebout sorting model to this setting and conduct three simulation experiments..."

**Add at end:** Brief roadmap: §2 describes the empirical phenomenon, §3 develops the framework, §4 presents findings, appendices contain formal specifications.

---

### §2 The Sorting Puzzle [NEW SECTION]

This section does what no section currently does: grounds the reader in what the simulation observes before introducing the formal machinery.

**Content to write (~300-400 words):**

1. **What we observe empirically:** Extremist communities do not distribute evenly. They concentrate on direct-voting platforms (approximately 50% of all extremist communities at all levels of system size and threat intensity, despite those platforms holding only 5–23% of total population). They overrepresent at 3–10× their population share, stably.

2. **What we observe dynamically:** Movement is not continuous. It is bursty. 94–100% of extremist outflow occurs in burst events. Median burst size: 31 communities. Median inter-burst interval: ~9 steps. The pattern resembles cycles, not drift.

3. **The governance signature:** Mainstream welfare on coalition platforms declines by 0.22 utility units as parasitism intensity rises from low to high; on direct platforms, the decline is 3.60 units. This is not a difference of degree — it is a difference in kind.

4. **The puzzle:** No individual community plans the raiding cycle. The model contains no coordination mechanism. How does this pattern emerge?

**Sources:** Pull these numbers directly from the Results section. This is preview, not repetition — the §4 results will explain them. Use 1-2 figures here (biography_panel.pdf and/or burst_marginals.pdf as illustration of the phenomenon).

---

### §3 Framework [CONDENSED — target ~500 words + 4 equations]

This section replaces the current 145-line §2. The full formal specification moves to **Appendix A**.

**Structure:**

**3.1 Platforms as Quasi-Governments (1 paragraph, no equations)**
- Tiebout in plain language: communities "vote with their feet"
- Platforms offer governance bundles (moderation rules, discourse architecture, community autonomy)
- Communities move when they can do better elsewhere
- Three types of governance regime (brief description: majority rule / movement formation / algorithmic grouping)
- Citation to Tiebout (1956); footnote pointing to Appendix A for formal specification

**3.2 Two Kinds of Extremist Community (1 paragraph + 2 equations)**
- Mainstream communities want matching governance; they suffer when extremists are present
- Extremist communities extract utility from mainstream neighbors ("utility vampires")
- Ideologues: low α — seek policy alignment, parasitism secondary
- Griefers: high α — seek accessible targets, policy alignment secondary
- Show only: $u_{c_m} = u_{\text{base}} - \alpha \cdot \frac{N_e}{N_e + N_m}$ and $u_{c_e} = u_{\text{base}} + \alpha \cdot \frac{N_m}{N_m + N_e}$
- One sentence: "α parameterizes the ideologue–griefer spectrum; we vary it across {2, 5, 10}."

**3.3 How Governance Type Creates Exposure (1 paragraph)**
- Direct platforms: all communities are mutual neighbors → no structural insulation
- Coalition platforms: winning coalitions define neighborhoods → potential segmentation
- Algorithmic platforms: SVD groupings define neighborhoods → partial insulation, cold-start vulnerability
- The critical claim: governance type determines whether extremists have structural access to mainstream communities
- Footnote to Appendix A §A.3 for full mechanism specification

**3.4 The Sorting Process (1 paragraph + 1 equation)**
- Each step: governance updates bundle → communities compute utility → communities search alternatives (bundles only, not social composition) → simultaneous relocation
- Information restriction: communities observe governance outputs, not population composition → strategic patterns are emergent, not planned
- $p^* = \arg\max_{q \neq p} u_{\text{base}}(c, q)$ — the search rule
- Simulation runs until no community can improve by moving

**End of §3:** "The formal model, governance algorithms, and experimental design are specified in Appendices A and B."

---

### §4 Results [REORDERED — lead with findings, not methods]

Each result subsection should begin with **a one-sentence statement of the finding**, then the supporting evidence, then the mechanism. Currently most subsections bury the finding.

**§4.1 The Sorting Mechanism Works: Baseline Results (R1, R2)**
Brief. 2–3 paragraphs. These results establish the baseline (Tiebout translates to platforms) and set up the more important findings. Do not expand.

**§4.2 Platform Diversification Protects — And More So Under Threat (R3)**
Lead with: "The diversification premium — the welfare gain from more platforms — nearly doubles as parasitism intensity increases." Table 3 (the N_p × α grid) comes first, before any methodological setup.

**§4.3 Governance Type Determines Structural Vulnerability (R4, R5, R6 cluster)**
Reorganize these three results as a single analytical cluster:
- R6 (concentration finding: 50% on direct, 3–10× overrepresentation) leads — it is the most striking descriptive
- R4 (behavioral regime shift: ideologue vs griefer) explains why
- R5 (coalition enclave mechanism) explains the protection channel
Currently R4 leads with the utility comparison; starting with R6's stark concentration figure is more arresting.

**§4.4 Raiding Cycles Are Structural, Not Strategic (R7)**
Lead with the burst statistics (94–100% outflow in bursts, median size 31, inter-burst ~9 steps). Then the mechanism paragraph. Then the displacement paradox (raids actually improve displaced communities' utility). This section currently buries the punchline; move it to the first sentence.

**§4.5 Mixed Populations Confirm the Pattern (R4b, robustness register)**
Keep as-is but retitle and add an explicit framing sentence: "These results hold when the extremist population is a realistic mixture of ideologues and griefers rather than a pure type."

---

### §5 Discussion [LIGHTLY REVISED]

Current content is strong. Minor changes:
- Adopt exploratory register throughout (see previous revision plan's framing table)
- Add one sentence in §5.3 on the policy implication of the structural interpretation: deplatforming a single venue does not break the cycle if the ecosystem conditions persist
- Limitations section: add explicit note on sensitivity analysis (pending results from OAT on HPC)

### §6 Conclusion [SEPARATED]

Pull the Conclusion subsection (§5.5) out as a standalone §6. No content change needed — separation signals it deserves its own weight.

---

### Appendix A: Formal Model Specification

Move here from current §2:
- §A.1 Agents and Environment (current §2.1)
- §A.2 Utility Functions, full specification (current §2.2, including all four equations)
- §A.3 Neighborhoods and Platform Governance (current §2.3)
- §A.4 Platform Governance Mechanisms — full algorithmic detail (current §2.4)
- §A.5 The Tiebout Cycle, formal specification (current §2.5)
- §A.6 Mapping to Real Platforms (current §2.6)

### Appendix B: Experimental Design

Move here from current §3:
- §B.1 Outcome Measures
- §B.2 Experiment 1 design table
- §B.3 Experiment 2 factorial design table
- §B.4 Sensitivity Analysis specification

### Appendix C: Sensitivity Analysis Results
Placeholder for OAT tornado table and α×N_p interaction table (results pending from HPC run).

---

## Key Principles for the Rewrite

1. **Lead with the finding, not the method.** Every result subsection opens with the substantive claim.
2. **Plain language first, equations second.** Each equation is preceded by a plain-language statement of what it says.
3. **Footnotes for qualifications.** The main text should flow without interruption. Assumptions, caveats, and limitations go in footnotes, except for the Limitations subsection.
4. **The model is in the appendix.** §3 should feel like a framework sketch, not a model specification. If a reader could understand the findings without the equations, the section is right.
5. **No new content.** Do not add claims not supported by the existing results. Restructure what exists.
6. **Preserve all citations, all LaTeX labels, all figure environments.** The bibliography is intact; do not drop citations.
7. **Framing register:** Use exploratory language throughout. "I find that..." not "I demonstrate that..." "This is consistent with..." not "This proves..."

---

## What NOT to Do

- Do not condense the results themselves. The evidence paragraphs can stay at their current length; what changes is the ordering and the lead sentence of each result.
- Do not delete the formal model content — move it to the appendix in full.
- Do not add new figures. Reference existing figure files by their current labels.
- Do not change the bibliography or citation keys.
- Do not rewrite the Discussion — minor edits only.
