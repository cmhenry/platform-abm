# Peer Review Memo

**Manuscript:** "Community Sorting Across Platforms"
**Author:** Colin Henry
**Reviewer:** Senior Scholar, Computational Social Science / Political Communication
**Date:** 2026-04-19

---

## Overall Assessment

This is a well-constructed theoretical paper with a genuine contribution: it shows that cross-platform extremist cycling can be explained as an emergent structural property of decentralized sorting, without invoking coordination. The Tiebout adaptation is intellectually defensible, the three-experiment architecture is sensible, and the Discussion does real analytical work connecting findings to policy. The empirics-first restructuring is, on balance, a success — §2 succeeds in making the results feel urgent before the model arrives. That said, several claims outrun their evidence, the Limitations section requires expansion given the still-absent sensitivity results, and a small set of register lapses undermine the paper's careful methodological positioning. These are fixable, but they need to be fixed before submission.

---

## Argument Structure

The empirics-first gambit in §2 (The Sorting Puzzle) largely works. Presenting the three headline results — 50% concentration, bursty movement, and the coalition welfare differential — before the model creates genuine anticipation. The reader wants to know how these patterns arise, and that motivation pulls them through the formal framework in §3. This is preferable to the conventional structure where simulation results feel anticlimactic after a long methods section.

Two problems, however, limit how well this structure lands.

First, the three-headline-findings announcement in the Introduction (lines 26–28) uses language that is almost identical to the §2 sub-section headers and the §5 summary. By the time the reader reaches the results, the punchlines have been delivered three times. The Introduction's findings paragraph should be pithier and less fully specified — enough to orient the reader, not enough to make §2 feel redundant. Right now the Introduction, §2, and §5 summary contain essentially the same sentences at different levels of detail. The repetition is not a structural virtue; it is copy that needs to be thinned.

Second, the condensed framework in §3 is adequate for a sophisticated reader but assumes more patience with formal notation than many readers in political communication will have. The four equations are introduced efficiently, but the connection between the search rule (Eq. 4) and the emergence of the raiding cycle is stated rather than shown: the text says communities evaluate alternatives "using only base utility" and leaves the implication to the reader. A single bridging sentence explaining why blind search plus the utility asymmetry between mainstream and extremist communities is sufficient to generate the bursty dynamics would make §3 earn more of its weight and reduce dependence on the Appendix.

---

## Claims vs. Evidence

**The "emergent property" claim for the raiding cycle** is the paper's most interesting theoretical move, and it is substantially supported. The mechanism description at lines 218–221 is specific and internally consistent: accumulation, policy drift, mainstream departure, parasitism drop, mass dispersal, repeat. The information restriction in the search rule (Eq. 4) is the key architectural choice that makes emergence credible rather than stipulated — communities cannot pre-select target-rich platforms, so the cycle cannot be the product of forward-looking coordination. This claim is proportionate and well-grounded.

**"Structural immunity" for coalition governance** (§3.3, §4.3, and the Discussion) is a different matter. The language is too strong relative to the evidence presented. The paper's own §4.4 (Result 4b) reports that at $f_g \geq 0.50$, at least 10% of iterations do not achieve enclave equilibrium within $t_{\max} = 100$, and that settling time rises from 21.8 to 39.7 steps across the griefer fraction gradient. The enclave mechanism is characterized explicitly as "slowed but not defeated" — which is precisely what "structural immunity" is not. Immunity implies resistance to breach; what the model actually shows is durable but not unconditional protection with a non-trivial settling lag. The paper should replace "structural immunity" throughout with language like "robust structural protection" or "durable structural buffering," and the claim in the Introduction ("coalition-based governance provides structural immunity through endogenous enclave formation," line 27) must be qualified to reflect the $f_g$ results. This is not a minor stylistic point: the overstated claim is the one most likely to attract adversarial scrutiny from a specialist reviewer.

**The "regardless of system parameters" stability claim** for the 50% concentration figure (line 44: "regardless of system size or threat intensity"; line 175: "stable across all levels of $\alpha$ and $N_p$") is too strong in its current form for a paper that acknowledges pending sensitivity analysis. The factorial design spans three levels each of $N_p$, $\rho_e$, and $\alpha$ — a reasonable range, but far from the full parameter space. The claim that concentration is stable "regardless of system parameters" implicitly generalizes beyond the tested domain. Because the sensitivity analysis results are not yet reported (Appendix §C is a placeholder, lines 614–617), the paper cannot currently support a universality claim. The appropriate phrasing is "across all tested factor levels" or "robust across the $3 \times 3 \times 3$ factorial," with a note that robustness to variation in $N_a$, coalition parameters, and $t_{\max}$ remains to be confirmed.

---

## Framing and Register

The paper is predominantly well-calibrated for an exploratory modeling paper. The Discussion correctly frames findings as mechanism identification rather than point prediction, and the limitations paragraph at lines 317–322 is honest about scope. However, several specific formulations need attention.

The phrase "structural immunity" (Introduction, line 27; §4.3 heading; §5 summary, line 280; Discussion, lines 297–298) is used six times and is consistently too strong, as noted above. This is the most pervasive register problem in the manuscript.

The sentence "the architecture of the platform system, not only the moderation decisions of individual platforms, determines whether mainstream communities are structurally protected or exposed" (Abstract, line 18; echoed at lines 315–316 and 328–329) slides from "the model suggests" to an unqualified causal determination claim. In a calibrated-simulation paper, "determines" requires an empirical foothold the model cannot provide; "shapes" or "conditions" would be accurate.

The Discussion (lines 285–286) characterizes the full range of normalized mainstream utility across the factorial as "a margin that would be difficult to distinguish from noise in most empirical settings." This sentence does double duty as a resilience claim and an implicit dismissal of the effect sizes. The comparison to empirical noise thresholds is rhetorically awkward for a simulation paper — the simulation produces clean data, so the comparison is not meaningful — and should be cut or rewritten.

The Limitations section (§5.4) acknowledges that sensitivity analysis results are pending but says only that they "will be reported in a subsequent update." This is inadequate. Given that the abstract and results sections make stability claims that the sensitivity analysis is specifically designed to test — particularly the $\alpha \times N_p$ interaction and the $N_a$ robustness — the paper needs either to hold those claims more tentatively throughout, or to explicitly flag in the Limitations that the stability claims are preliminary pending Appendix C. Right now the body of the paper reads as if the sensitivity analysis is a confirmatory formality rather than a live epistemic question.

---

## Key Weaknesses

**1. "Structural immunity" language overstates the coalition protection finding (lines 27, 197, 201, 209, 280, 297–298).** The paper's own Result 4b demonstrates that the enclave mechanism fails to converge in a non-trivial fraction of runs at high griefer fractions, and that settling time nearly doubles across the $f_g$ gradient. "Structural immunity" implies categorical protection; the results support "robust but conditional protection with a significant settling lag." This needs to be corrected in the Introduction, Results section header, and Discussion before submission.

**2. The sensitivity analysis placeholder undermines the universality of the concentration claim (Appendix §C, lines 614–617; Introduction line 44; Results lines 175–179).** The 50% concentration result is described as stable "regardless of system parameters" before the paper has tested robustness to $N_a$, coalition parameters, $g$ and $m$, or extended $t_{\max}$. Until Appendix C is populated, these claims must be bounded to the tested factorial. This is not a theoretical objection — the finding is plausible and potentially robust — but the language currently claims more than the evidence can bear.

**3. The Introduction-§2-§5 summary triple repetition blunts the empirics-first structure (lines 26–28, 43–49, 280–281).** The three headline findings are stated at essentially the same level of specificity in all three locations. This eliminates the anticipatory value of §2 and makes the summary in §5 feel superfluous. The Introduction should gesture at the findings without specifying the numbers; §2 should be where the numbers land with full impact.

**4. The mechanism bridging §3 and §4 is underdeveloped.** The framework section correctly identifies that search is blind to social composition (Eq. 4) and notes that this means emergent patterns arise without coordination. But it does not explain, even informally, why blind search plus the asymmetric utility functions is sufficient to produce the concentration-raid-retreat cycle. A reader unfamiliar with agent-based tipping dynamics will arrive at §4.4 without understanding why the result is non-obvious. One or two sentences in §3.4 making this connection explicit — something like: "because extremists cannot identify target-rich platforms before arriving, their concentration on direct platforms must be explained by the payoff structure they encounter after arrival, not before" — would make the model's theoretical work visible.

**5. The Discussion's treatment of the displacement paradox (lines 242–243, 313–314) is positioned as a resilience affirmation but could be a limitation.** The finding that displaced mainstream communities typically land on better-matched platforms is presented as evidence that "the Tiebout mechanism self-corrects." But this self-correction depends on there being enough platforms and community diversity to absorb displacement — a condition that the paper elsewhere acknowledges is not guaranteed (at $N_p = 3$, 6.1% of displacements produce negative utility changes). The Discussion should acknowledge that positive mean displacement utility is a feature of the high-$N_p$ regime, not a general property of the sorting mechanism, and frame this as a boundary condition on the resilience claim rather than confirmation of it.

---

## Strengths to Preserve

The behavioral regime shift analysis in §4.3 (Result 4) is the paper's strongest analytical set-piece. The distinction between ideologue and griefer behavior under the $\alpha$ parameterization is cleanly operationalized, the regime shift is tracked through specific utility ratios across $\alpha$ levels, and the counterintuitive finding — that mainstream utility on direct platforms is *higher* in consolidated systems at $\alpha = 10$ due to turbulence — is the kind of result that demonstrates genuine model insight rather than confirmation of priors. This should not be touched.

The Tiebout adaptation is defended convincingly and with appropriate scope. The paper does not overclaim the analogy; it is clear that platforms are quasi-governments, not municipalities, and that the model is a theoretical device, not a predictive instrument. This calibration is exactly right for the venue and should be maintained.

The information-restriction design choice — communities observe governance outputs but not social composition (§3.4, §A.5) — is the move that makes the emergence claim credible, and the paper explains it well in both the framework and appendix. This should remain intact and, if anything, be made more prominent in the Discussion when addressing the policy implications for deplatforming.

The experimental structure of §4.4 (mixed populations) as a validation of the $\alpha$ proxy is methodologically sound and the "validation" framing is honest about what it does and does not demonstrate. The finding that the ideologue-griefer transition is smooth and continuous rather than categorical is a genuine contribution to how we should model extremist community behavior. Keep it.

---

## Priority Fixes Before Submission

**Priority 1:** Replace all instances of "structural immunity" with qualified language ("robust structural protection," "durable but conditional buffering") and add a sentence in §5.4 Limitations explicitly flagging that the protection claim is qualified by the settling-time and convergence-failure results in §4.4.

**Priority 2:** Populate Appendix C (Sensitivity Analysis Results) or, if the HPC results are not yet available, downgrade all "regardless of system parameters" and "stable across all factor levels" claims in the main text to explicitly bound the assertion to the tested factorial design, with a forward reference to the pending appendix.

**Priority 3:** Cut the findings paragraph in the Introduction back to two or three sentences that orient without specifying; let §2 carry the empirical weight. The current triple repetition (Introduction, §2, §5 summary) costs the paper the anticipatory energy that the empirics-first structure was designed to create.

---

## Recommendation

**Major Revision.** The paper has a sound theoretical contribution and a workable structure, but two issues prevent submission in the current form: the overstated immunity language needs systematic correction, and the sensitivity analysis must either be reported or the claims bounded accordingly throughout the paper. These are substantive changes to the argument, not editorial polishing. After revision addressing the three priority items above, the paper should be in shape for resubmission.
