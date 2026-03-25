
## The Framing–Results Connection

The paper has a strong theoretical scaffold, but I think there's a tension between the framing and what the results are best positioned to deliver. Let me unpack that.

**What the introduction promises.** The paper opens with an empirical puzzle—extremist *cycling* across platforms—and offers a system-level model of inter-platform dynamics. The three-part argument promises claims about (1) sorting efficiency, (2) extremist strategy shaped by system structure, and (3) the paradoxical protective effect of coalition governance. That's a lot to carry, and the audiences you're writing for will weight these differently.

**What the results actually show.** Looking at your seven planned results, you have two distinct papers layered on top of each other:

- **Paper A** is a Tiebout-for-platforms paper (Results 1–2). It demonstrates that the foot-voting mechanism translates to online platforms and that diversification improves welfare. This is clean and defensible but, on its own, is a theoretical extension with a well-known punchline: more options → better sorting. Political scientists who know Tiebout will nod; pol comm and platform studies people will say "so what?"

- **Paper B** is the extremist dynamics paper (Results 3–7). This is where the novel contributions live. The interaction between system structure and parasitism intensity (R3), the ideologue/griefer behavioral regimes (R4), the coalition enclave mechanism (R5), extremist concentration on direct platforms (R6), and the bursty raiding cycle (R7)—these are the findings that will catch attention across all three audiences.

The current manuscript structure treats Paper A as setup and Paper B as the main event, which is correct. But the framing in the introduction could be tightened to match. Right now the abstract and intro emphasize "do communities efficiently sort?" as a standalone question of theoretical interest. For your audiences, the more compelling framing is: *the Tiebout mechanism works for platforms—until parasitic actors break it in structurally predictable ways, and the way it breaks tells us something important about platform system design.*

## Plain-Text Takeaways: What Should They Be?

Here's how I'd articulate the findings your audiences will actually remember and cite, ranked by expected impact:

**1. "The platform ecosystem is a system, not a collection of independent sites."** This is your meta-contribution. Pol comm people study individual platforms; platform studies people study governance within platforms. Your model forces the reader to think about cross-platform dynamics as an object of theory. The raiding cycle is the killer illustration: it's a system-level emergent pattern that no single platform's moderation policy produces or can prevent alone.

**2. "Extremists don't just get deplatformed—they exploit the *interfaces* between platform types."** Result 6 (concentration on direct platforms) plus Result 7 (raiding via the direct→algorithmic channel) is the most novel and empirically resonant finding. The plain-text version: extremists use small, permissive platforms as staging grounds and algorithmically curated platforms as hunting grounds, exploiting the cold-start problem to temporarily access mainstream audiences before the recommendation system re-sorts them. This maps directly onto observed behavior (Kiwi Farms → Twitter, 4chan → YouTube, etc.) and gives journalists and policymakers a structural explanation for something they've been describing anecdotally.

**3. "More platforms help everyone—but they help mainstream communities *most* when the extremist threat is worst."** Result 3 is your most policy-relevant finding: the diversification premium grows with parasitism intensity. The takeaway isn't just "monopoly bad" (which everyone already believes). It's that the *cost* of platform consolidation is endogenous to the threat environment. When extremists are ideologues, consolidation hurts a little. When they're griefers, consolidation hurts a lot. That interaction is the finding, not the main effect.

**4. "Coalition governance works not because it's democratic, but because it produces enclaves."** Result 5 is counterintuitive and interesting. The mechanism—movements harden into type-homogeneous blocs that sever the parasitism channel—reframes coalition platforms from "messy and unstable" (their Experiment 1 reputation) to "natural firewall." The plain-text version: platforms that let communities organize and form political movements end up protecting mainstream users by segregating extremists into their own spaces. This is a theoretically grounded case for community self-governance.

**5. "The ideologue/griefer distinction is behavioral, not typological."** Result 4 matters because it reframes how we think about extremist communities. They aren't born as one type or the other; the payoff structure makes them behave as one or the other. A community that looks like an ideologue at α=2 looks like a griefer at α=10. The implication: platform design choices that increase the payoff to parasitism (e.g., recommendation systems that expose mainstream users to hostile content) can convert ideologues into griefers.

## What the Best Collection of Results Looks Like

Given all of this, here's what I'd recommend for the ideal results portfolio:

**Keep all seven results, but restructure the narrative arc.** Right now the results are organized sequentially (Experiment 1 → Experiment 2 → Dynamics). I'd recommend organizing by *argument* instead:

1. **The baseline works** (R1 + R2, compressed). Governance type matters; diversification helps. Establish this in roughly two pages with a single summary table and a brief note on the mixed-system ranking inversion. This isn't the contribution—it's the premise.

2. **Extremists break the baseline in structurally predictable ways** (R3 + R4). The diversification premium interacts with parasitism. The α parameter produces qualitatively distinct behavioral regimes. These are your two cross-sectional headline findings. Present them as a pair: system structure and extremist behavior are *jointly determined*, not independent.

3. **The breaks are governance-specific** (R5 + R6). Coalition platforms form protective enclaves. Extremists concentrate on direct platforms. These explain *where* the system-level effects land. They also set up the dynamics story by establishing the geography of the sorting equilibrium.

4. **The dynamics reveal an emergent raiding cycle** (R7, expanded). This is the most novel result and deserves its own subsection (which you already have). But I'd push harder on three things the current comment skeleton doesn't fully develop:
   - **Whether raids escalate or are stationary.** Your comments note this is pending. If raids escalate, that's a runaway dynamic with alarming policy implications. If they're stationary (bursty but non-escalating), that's a steady-state cost of the system architecture. Either is interesting, but they imply very different policy responses.
   - **The cold-start mechanism as a specific vulnerability.** The fact that the raiding cycle *depends on* the algorithmic platform's cold-start problem is a concrete, actionable finding for platform designers. It says: if you can improve how your recommendation system handles sudden influxes of new users with correlated preferences, you can disrupt the raiding cycle.
   - **Cycle length and amplitude across the factorial.** You have 27 configurations. The burst statistics (median size, inter-burst interval) across the α × N_p grid would make a compelling heatmap and would show exactly where in parameter space the raiding cycle is most dangerous.

**Add one result you don't currently have planned: mainstream exit dynamics.** You track relocations for all community types, so you should be able to show what happens to *mainstream* communities when a raid arrives. Do they leave? How fast? Where do they go? This would close the loop on the raiding cycle by showing the full sequence: extremist concentration → raid → mainstream displacement → mainstream re-sorting → extremist return. If mainstream communities consistently flee to coalition platforms after being raided on algorithmic ones, that's a finding about endogenous demand for community self-governance that ties R5 and R7 together beautifully.

**For the sensitivity analysis:** report it concisely. The most important result from sensitivity will be whether the α × N_p interaction is robust—that's the load-bearing policy claim. The OAT results can go in an appendix as a table showing which parameters flip which qualitative conclusions (ideally none).

## One Flag on Audience Framing

The paper currently positions itself primarily as a formal-modeling contribution with policy implications. For political scientists, that's fine—they're comfortable with ABMs as theory-building tools, and the Tiebout framing gives it a disciplinary home. But for pol comm and platform studies audiences, I'd recommend a small but important addition somewhere in the discussion: an explicit mapping between model constructs and observable platform features. Something like: "direct voting maps to Reddit-style community upvoting; coalition formation maps to community-organized campaigns like those that changed Twitter's deadnaming policy; algorithmic recommendation maps to TikTok/YouTube-style content curation." You already gesture at these in the commented-out sections, but the revised paper should make them prominent enough that a reader who has never seen a Tiebout model can immediately see why the institutional distinctions matter.