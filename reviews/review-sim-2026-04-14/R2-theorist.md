# Reviewer 2: Theorist
*Simulated review — generated 2026-04-14*
*Persona: theoretical contribution and novelty focus*

---

**Summary**

This paper adapts the Tiebout model of citizen sorting to explain emergent extremist cycling across multi-platform ecosystems. Platforms are treated as quasi-governments offering public good bundles, and communities — including parasitic extremist types — vote with their feet across governance regimes (direct voting, coalition formation, algorithmic recommendation). Using a 3×3×3 factorial ABM experiment, the paper argues that the cycling pattern is a structural property of the ecosystem, not coordinated strategy, and that platform-system architecture — rather than individual moderation decisions — determines whether mainstream communities are protected. The coalition governance finding — that a mechanism appearing mediocre without extremists becomes the strongest firewall under threat — is the paper's most interesting theoretical claim.

---

**Strengths**

The Tiebout adaptation is genuinely original and theoretically motivated, not merely a borrowed metaphor. The reconceptualization of platforms as public good providers and communities as mobile agents captures two features missing from the existing deplatforming literature: communities respond to a governance bundle rather than a single policy, and their decisions are shaped by system composition rather than by any one platform. This is a real theoretical move, not redescription.

The typological distinction between ideologue and griefer extremists — governed by a single continuous parameter — is elegant. It avoids the binary forced choice between "preference-seeker" and "harassment-seeker" while generating qualitatively distinct behavioral regimes as the parameter varies. The regime-shift result (Result 4) is the paper's most specific and surprising empirical finding: the same governance mechanism that functions as a boutique space at low parasitism collapses to below-random baseline at high parasitism. That is a falsifiable, non-obvious prediction.

The displacement paradox (Result 7) is theoretically useful precisely because it cuts against the expected narrative. Raids are costly to individual platforms but net-positive for displaced communities in high-diversity systems. This self-correcting property of the sorting mechanism is the paper's cleanest contribution to understanding resilience.

The η² reporting is appropriate — the ANOVA is used as a variance decomposition rather than a significance test, which is the right application for a simulation with controlled parameter variation.

---

**Weaknesses**

**Major — addressable**

*The raiding cycle claim relies on an information restriction that is not adequately defended for the cases where it matters most.*

The paper asserts that the raiding cycle is an "emergent property of decentralized sorting, not coordinated strategy," and treats this as a central theoretical contribution. The mechanism depends on communities observing only the current public good bundle of alternative platforms without observing community composition. This assumption does a lot of work. The cited evidence (Fiesler et al.; Chandrasekharan et al.) applies to fandom communities facing technical barriers to platform migration, not to well-organized extremist groups who actively surveil target platforms. The paper needs either a sensitivity analysis varying information availability or an explicit argument for why the information restriction is the right model for strategically oriented extremist communities specifically.

**Major — addressable**

*The coalition enclave mechanism is underspecified at its critical point.*

The paper states that "extremist and mainstream communities naturally sort into separate coalitions." Why naturally? In the model, coalitions form through a mutation-based search process responding to tenant community preferences. There is no reason built into the mechanism that prevents an extremist-dominated coalition from absorbing mainstream communities, particularly during the settling period (median 23 steps at high parasitism). What determines whether sorting produces segregated enclaves versus mixed coalitions that extremists dominate? The paper characterizes only the end state (homogeneity 0.92–0.98), not the mechanism producing segregation. This is the key mechanism of the paper's strongest result and needs to be traced, not assumed.

**Major — fundamental limitation**

*The paper claims to explain an empirically documented pattern but the model has no empirical calibration to those cases.*

The introduction and conclusion frame results as explaining the observed concentration-raid-retreat cycling. The limitations section disclaims predictive intent and invokes the exploratory modeling tradition. This is the bait-and-switch scope problem: the scope of the claim and the scope of the evidence are misaligned. The paper should either (a) make the explanatory claim weaker — structural incentives are a sufficient condition for the cycling pattern to arise from decentralized behavior — or (b) strengthen the case for mechanism correspondence by engaging with the specific empirical cases more carefully. The current mismatch will produce a standard objection at any of the three candidate venues.

**Minor — addressable**

*The coalition mechanism mapping is uneven.* The cited example — the trans deadnaming reversal campaign — is a within-platform governance change driven by user advocacy, not a community migration decision. The Tiebout logic requires communities moving between platforms. The mapping section needs a clearer statement of what the coalition mechanism is actually modeling.

**Minor — note**

*Content registry unavailable for this review.* I cannot confirm whether the specific mechanism — public-good sorting under parasitic agents with a behavioral regime shift — has been proposed elsewhere under a different name. There is likely work in computational social science on Schelling-type sorting with adversarial agents, and in population dynamics on host-parasite systems where diversity serves as insurance against parasite load, that may make structurally similar moves. The authors should check and engage or note that they have reviewed and found no close analogues.

---

**Questions for the Authors**

1. What happens to the raiding cycle if communities have partial information about extremist density on alternative platforms — for example, a noisy signal of extremist proportion at a candidate platform before deciding to move? Does the burst pattern survive, or does it depend on information asymmetry to generate concentration?

2. In the coalition formation mechanism, is there any condition under which a griefer-dominated coalition captures a platform and absorbs mainstream communities into its neighborhood rather than segregating them? If not, why not — and if so, how frequently does this occur across simulation runs?

3. The introduction frames the paper as explaining an empirically documented puzzle; the limitations section concedes the model is not calibrated to real platforms. What is the precise scope of the explanatory claim: sufficient condition for the cycling pattern, or plausible contributing mechanism alongside coordination?

4. The displacement paradox (positive mean displacement utility, zero negative events at high diversity) is striking. Is this an artifact of communities evaluating destinations using base utility only, or does it hold when realized parasitism at the destination platform is included?

---

**Recommendation: Revise and Resubmit**

The theoretical contribution is real and the coalition paradox result is the strongest simulation-based finding I have seen in this genre in some time. But the bait-and-switch between the empirical puzzle framing and the exploratory-model disclaimer must be resolved, and the coalition enclave mechanism needs to be traced rather than asserted. The information restriction assumption requires either a sensitivity check or a principled defense that distinguishes strategically sophisticated extremist communities from the fandom-migration evidence cited to justify it. These are addressable revisions, not fundamental flaws.
