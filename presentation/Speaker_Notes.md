# Speaker notes - Mentor Presentation (15 minutes)

Time budget per slide is in the footer of each slide and in the table below; appendix slides are held back for the Q&A.

| # | Section | Slide | Minutes | Cumulative |
|---|---|---|---|---|
| 1 | Title | Health-Investment Policy Advisor | 0.5 | 0.5 |
| 2 | 1 · Industry context and problem definition | Where should a ministry put the next health dollar? | 1.5 | 2 |
| 3 | 2 · Integrated AI system overview | One question in, one audited brief out | 1.5 | 3.5 |
| 4 | 2 · Integrated AI system overview | Four layers, one control loop | 1 | 4.5 |
| 5 | 3 · Integration of prior capstone projects | Four prior projects, each forced a design decision | 2 | 6.5 |
| 6 | 3 · Integration of prior capstone projects | How the integration improved the solution | 1 | 7.5 |
| 7 | 4 · Key technical decisions and tradeoffs | Decisions, alternatives, and what was given up | 2 | 9.5 |
| 8 | 5 · Ethical considerations and responsible AI | Each risk maps to a mechanism in the code | 2 | 11.5 |
| 9 | 6 · Evaluation, limitations and risks | Evaluation: 11/11 scenarios, 13/13 tests, three fixes | 1.5 | 13 |
| 10 | 6 · Evaluation, limitations and risks | Known limitations and production risks | 1 | 14 |
| 11 | 7 · Professional relevance and next steps | What this demonstrates, and what I would do next | 1 | 15 |

## Slide 1: Health-Investment Policy Advisor (0.5 min)

Good [morning/afternoon]. I will present the Health-Investment Policy Advisor, the integrated system I built for Project 7. In one sentence: it helps a ministry of health decide which national investment levers are associated with longer lives, and it is engineered so that every number it reports is traceable and a human signs off before anything is saved. I will follow the seven required sections: context, system, integration, decisions, ethics, evaluation, and professional relevance. About fifteen minutes, then questions.

## Slide 2: Where should a ministry put the next health dollar? (1.5 min)

Ground the listener before any technical content. Decision, stakeholders, stakes. Then the two-part 'why AI': the question is a pattern-recognition problem over World Bank data, and the real bottleneck is synthesis, not data. Close on the risk, because the risk is what shaped the design: the failure mode I most fear is a fluent, wrong number ending up in a cabinet paper.

## Slide 3: One question in, one audited brief out (1.5 min)

Walk the workflow left to right. (1) Scope check - refuse individual medical or political requests before any tool runs. (2) Retrieve the country profile, peer comparison and analyst notes. (3) Simulate the requested interventions with a country-grouped gradient-boosting model. (4) Ask a regularised neural network for a second opinion and uncertainty flags. (5) Draft a brief from a typed evidence bundle. (6) Audit the draft: every number must trace to a tool output. (7) Human approval before saving. (8) Return structured JSON: answer, sources, confidence, caveats, refused.

## Slide 4: Four layers, one control loop (1 min)

Colours map layers to prior projects. Evidence layer produces numbers; neural layer qualifies them; generative layer communicates them; agentic control layer decides when each may be called and whether the result may be saved. Interaction is through typed tools: the agent never touches the data directly, it only calls tools with validated arguments.

## Slide 5: Four prior projects, each forced a design decision (2 min)

This is the synthesis slide - give it two minutes. For each row: what came over, and what decision it changed. The blog post's list of systematically mis-predicted countries is the most valuable carry-over: it tells the agent when to trust itself less. The deep-learning project taught me how to run a fair controlled experiment; here the regularised network wins against its baseline but not against boosting, so it is deployed as a second opinion. The generative project taught me to audit generated text with n-gram and format checks rather than by eye; that became the grounding audit. The agentic project supplied the whole control layer almost unchanged. Why combine: evidence produces numbers, neural qualifies them, generative communicates them, agentic governs them. Remove any one and a specific safeguard disappears.

## Slide 6: How the integration improved the solution (1 min)

Make the improvement concrete with the counterfactual: what would be lost without each layer. The through-line is that integration bought governance, not accuracy - the point estimate is the same as the blog post; what is new is knowing when not to trust it and making sure nothing ungrounded is saved.

## Slide 7: Decisions, alternatives, and what was given up (2 min)

For each row say the alternative and what was given up. Two stories worth telling: first, when the audit rejected a correct template draft because a derived gap (observed minus predicted) was not in the bundle, I added the derived quantity to the bundle rather than loosening the audit. Second, MC-dropout spread alone was uncorrelated with error (r about -0.01), so I added cross-model disagreement, which is what actually tracks error. I am not claiming these choices are perfect; the through-line is that I traded fluency and flexibility for traceability because the audience is a public institution.

## Slide 8: Each risk maps to a mechanism in the code (2 min)

Be specific, not generic: every row names a mechanism in the code. Mention the planted Cambodia injection that is flagged and has no effect, and that on Nigeria the low confidence (0.30) is the equity safeguard doing its job. Frameworks: NIST AI RMF Govern/Manage for the trace and approval gate; Mitchell et al. model cards for sub-population reporting; OWASP LLM Top 10 for injection. Add the honest caveat: the scope and injection guards are regular expressions - tripwires, not walls; the structural defence is that the planner does not read tool output as instructions.

## Slide 9: Evaluation: 11/11 scenarios, 13/13 tests, three fixes (1.5 min)

Lead with 'did it meet its goals': yes for traceability and governance (every scenario passes, every brief grounded), partially for accuracy (2.4-3.0 years MAE is useful for ranking options, not for fine differences). Then the three things that did not work first time - these are the strongest evidence of reflective judgment. Emphasise that evaluation changed the system, it did not just measure it.

## Slide 10: Known limitations and production risks (1 min)

Own these plainly; they are also in the paper and the notebook's boundaries section. If asked which is most serious for production: the associational framing, because it is the one a busy reader is most likely to forget.

## Slide 11: What this demonstrates, and what I would do next (1 min)

Connect to employer expectations: the whole loop of an applied AI engineer, plus judgment about what not to build - no causal claims, no autonomy, no individual advice. Close: 'For a development-bank country team this compresses a week of spreadsheet work into a reproducible, auditable brief without removing the analyst from the decision.' Then: 'Thank you - I welcome your questions.'

## Slide 12: Model card: GradientBoostingRegressor (appendix)

Backup slide - not presented. Pull up if asked about the model, features, or validation.

## Slide 13: Seven typed tools; one side effect (appendix)

Backup slide. Budgets: max 10 steps, 4 tool calls per step, 180 s runtime, 30 context messages. Allow-list enforced; unknown tool or invalid args return an error object, never a crash.

## Slide 14: How confidence and grounding are computed (appendix)

Backup slide. Be upfront that the penalties are heuristic; conformal intervals are the planned replacement.

## Slide 15: 11 scenarios, 11 pass (appendix)

Backup slide. Each scenario has explicit expectations (status, tool-call count, confidence bounds, caveat text, grounding, injection flagged). Faults are injected via tool overrides: simulate_intervention raising, and a writer that fabricates numbers.
