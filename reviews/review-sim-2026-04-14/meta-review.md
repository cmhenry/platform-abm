# Meta-Review: Synthesis and Decision
*Simulated review — generated 2026-04-14*
*Synthesizing R1 (identification), R2 (theorist), R3 (generalist)*

---

## Points of Consensus

**Strength — preserved across all three reviews.** The coalition paradox (coalition governance mediocre without extremists, strongest firewall under threat) and the displacement paradox (raids benefit displaced communities) are recognized by all three reviewers as genuine, non-obvious results that could not arise from a static model. These should not be softened or hedged in revision.

**Weakness — universal agreement.** The absence of reported sensitivity analyses is the single highest-signal item in this review package. All three reviewers flag it; R1 and R2 call it major and addressable, R3 states the paper is not submittable without it. The paper's limitations section self-reports the gap. This is not a dispute about framing — it is a missing component of the experimental design that conditions all seven results.

**Weakness — two of three reviewers.** R1 and R2 both identify the simultaneity of relocation (Step 5 of the Tiebout cycle) as a confound in the emergence claim. R1 frames it as a mechanism-bundling problem: the simultaneity assumption may generate mass-departure bursts independently of the payoff structure, making it impossible to attribute the raiding cycle purely to decentralized sorting incentives. R2 approaches it through the information-restriction lens, questioning whether the blind-search assumption is defensible for strategically oriented extremist communities. These are not the same objection, but they converge on the same target: the paper's central causal claim — that the raiding cycle is emergent rather than coordinated — is not adequately isolated from the design assumptions that could themselves produce the result.

---

## Points of Disagreement

**Scope of the empirical claim.** R2 raises the bait-and-switch scope problem most explicitly: the paper's introduction frames results as explaining an observed phenomenon, while the limitations section retreats to exploratory modeling. R1 and R3 do not raise this as sharply. This is a genuine evaluative difference with no clear resolution from the reviews alone, but it is not a reviewer-preference issue — it reflects real tension in the manuscript between explanatory and exploratory registers. The author must choose one framing and apply it consistently. Weakening the explanatory claim (structural incentives are a sufficient condition for the cycling pattern to arise) is the lower-cost revision; strengthening it by engaging the specific empirical cases more directly is the higher-cost but potentially stronger move. Either is defensible; the current middle ground is not.

**The coalition mechanism.** R2 identifies the enclave segregation mechanism as critically underspecified: the paper reports end-state homogeneity (0.92–0.98) but does not trace how extremist-dominated coalitions are prevented from absorbing mainstream communities during settling. R1 and R3 do not raise this. This is not a reviewer preference — R2 has identified a genuine mechanistic gap in the paper's strongest result. Treat this as a required revision target regardless of R1 and R3's silence.

**Presentation.** R3 flags the inconsistent result count (three in abstract, seven in results section, three again in summary) and the buried contribution as a major presentation problem. R1 and R2 do not raise this. R3's misread of the paper's central claim is itself informative: if a generalist reviewer cannot identify the paper's core claims by entering at different points, the paper will have difficulty at journals with mixed-expertise editorial boards. Fix it.

---

## Blind Spots

**Reproducibility.** No reviewer addressed it. The model is implemented in AgentPy; neither the code availability statement nor the data-sharing protocol appears in the reviews. At CCR and JCSS, computational reproducibility is a near-mandatory standard. The author should confirm whether code and simulation outputs will be archived and add a reproducibility statement before submission.

**Single-author scope.** Seven results from a 3×3×3 factorial with 200 iterations per configuration, planned sensitivity analyses not yet run, and a limitations section that essentially promises a future paper — the revision target is ambitious. The sensitivity analyses are the binding constraint on R&R timeline. The author should be realistic about this before committing to a venue deadline.

---

## Revision Roadmap

### Required

1. **Run and report the sensitivity analyses.** This is not optional. Without robustness results, the paper's policy claims rest on a single parameterization, and no reviewer believes that is sufficient for the target venues. Priority should go to the staggered-relocation variant (which addresses the simultaneous-departure confound directly) and to parameter-space coverage around the diversification premium and enclave-homogeneity statistics.

2. **Trace the coalition enclave mechanism, not just the end state.** Show why extremist-dominated coalitions do not absorb mainstream communities during settling. If the model produces this outcome deterministically from the coalition formation rules, demonstrate it. If it does not always produce it, report the failure-mode rate.

3. **Resolve the scope of the empirical claim.** Pick one register — explanatory or exploratory — and apply it uniformly from abstract through conclusion. The current mismatch is not a minor framing issue; it is the foundation of R2's most fundamental objection and will surface at all three candidate venues.

### Strengthening

4. **Disaggregate the ρ_e operationalization.** R1's point about the conflation of ideologue and griefer counts at fixed α is well-taken. A supplementary set of runs varying ideologue-griefer composition at fixed ρ_e and α would sharpen the behavioral-regime claim in Result 4.

5. **Resolve the abstract/results/summary inconsistency.** Align the claimed results across all three locations. Low-cost fix, disproportionate impact on first-impression reads.

6. **Add a reproducibility statement and confirm code availability.**

### Discretionary

7. R2's concern about the information restriction's applicability to strategically sophisticated extremist groups is worth engaging briefly in the limitations or methods. A full sensitivity analysis varying information availability is not required — a principled defense of the modeling choice relative to the empirical cases cited is sufficient.

8. R1's concern about normalized utility comparability across community types is valid but minor. A footnote acknowledging the limitation — that extremist normalized utility includes a transferred component absent from mainstream utility — is adequate.

9. R3's suggestion that the introduction structure should track venue selection is reasonable if *Political Communication* is still under consideration. If the target is CCR or JCSS, the current structure is appropriate and this suggestion can be declined.

---

## Mock Decision

**Major Revisions**

The paper makes a genuine theoretical contribution and reports a set of results that are internally coherent and individually interesting. However, as currently written it cannot succeed at any of the three candidate venues: the sensitivity analyses are absent, the coalition enclave mechanism is underspecified at its most critical point, and the scope of the empirical claim is internally contradictory. These are not cosmetic problems. At CCR or JCSS — competitive but not top-5 venues — a paper at this stage would typically receive major revisions with a clear path to acceptance if the required items are addressed. The same package would face higher risk at *Political Communication*, where the explanatory framing is likely to be held to a higher evidentiary standard and the formal modeling contribution may be less familiar to reviewers. The sensitivity analyses are the long pole in the revision tent; everything else can be done in parallel.
