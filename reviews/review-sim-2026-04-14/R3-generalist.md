# Reviewer 3: Generalist
*Simulated review — generated 2026-04-14*
*Persona: framing, clarity, and presentation focus*

---

**Summary**

This paper adapts the Tiebout sorting model to explain why extremist communities cycle across the platform ecosystem rather than disappear after deplatforming. Using an ABM with three platform governance types (direct voting, coalition formation, algorithmic recommendation), the author finds that platform-system architecture — specifically the mix of governance types and degree of jurisdictional diversity — shapes whether mainstream communities are structurally protected from extremist parasitism. The headline claim is that coalition-based governance paradoxically produces the strongest protection because it generates enclaves that sever the parasitism channel, and that the cycling pattern is an emergent structural property rather than coordinated strategy. I read this as primarily a computational political science paper with platform governance implications — I may be underselling the formal modeling contribution, but the abstract and introduction read as promising a substantive finding, not a methodological one, so I'm taking the paper at its word.

---

**Strengths**

The mapping from the Tiebout framework to the platform setting is intuitive and explained well enough that I could follow it without expertise in public choice theory. The three governance types correspond to recognizable real platforms, which makes the results feel grounded. The "coalition paradox" — the governance type that appears mediocre without extremists provides the strongest protection with them — is a genuinely interesting finding and is stated clearly. The discussion section correctly connects findings back to the empirical puzzle in the introduction.

---

**Weaknesses**

**Major — presentation problem, significant**

*The contribution is buried and stated inconsistently.* The abstract lists three results; the results section reports seven; the summary collapses them into three again. By the time I reached the discussion, I was not sure which finding the paper was actually claiming as its central contribution. The abstract foregrounds resilience; the discussion opens with it but immediately qualifies it into near-irrelevance. A reader skimming to find the paper's point — which is how most readers approach journal submissions — will land on different answers depending on where they enter. This needs to be fixed before submission regardless of venue.

**Major — not submittable without it**

*The sensitivity analysis is missing.* The paper promises it in the experimental design section and explicitly notes in the limitations that "those results are not yet reported." This is not a minor gap. The model contains parameters chosen for tractability and the paper's policy conclusions rest on findings that have not been robustness-checked. Reviewers who know ABM work will flag this immediately. (Repositioning not applicable — this is a required revision.)

**Minor — one sentence fix**

*The null comparator is unexplained in the results.* Throughout the results, findings are compared against the "random baseline of 5.0." Where this baseline comes from is not explained in the results section — it appears to derive from N_a = 10 and random allocation, but a reader who has not read the model section carefully will not know this. A single sentence at the start of the results section would fix it.

**Minor — repositioning, venue-dependent**

*Introduction structure should track venue selection.* The paper is positioned as political communication / platform governance research, but the introduction devotes significant space to the Tiebout literature and ABM methodology before arriving at the political puzzle. For CCR or JCSS, this ordering works. For *Political Communication*, reviewers will expect the political puzzle to lead and the method to follow. If the venue has not been decided, the introduction's structure should be part of that decision.

---

**Questions for the Authors**

1. Communities in the model move to maximize utility — that is a form of strategy. What specifically distinguishes the model's "emergent" claim from decentralized strategic behavior? This matters for the policy conclusion about disrupting coordination channels.

2. The coalition governance finding is described as paradoxical, but is it actually surprising given that coalitions create internal segregation mechanisms? What would have made this a non-result?

3. The sensitivity analysis table exists in the paper but the results are not reported. When will they be available, and how confident are the authors that the core findings hold at different N_a and governance parameters?

---

**Recommendation: Revise and Resubmit**

The theoretical contribution is clear and the core findings are interesting, but the paper should not be submitted without the promised sensitivity analysis, and the results section needs structural work so that the paper's central claim is the same wherever a reader enters.
