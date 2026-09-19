# Speaker notes - Mentor Presentation (15 minutes)

## Slide 1: Health-Investment Policy Advisor

(1 min) Introduce the project: an AI advisor that helps ministries of health and development banks decide which health-investment levers are associated with longer lives, while staying honest about uncertainty. Four prior capstone projects are integrated; I will show what each contributes and how they interact.

## Slide 2: Industry problem: where should a ministry put the next dollar?

(1.5 min) Frame the decision and the stakes. Stress that the bottleneck is synthesis under time pressure, and that the failure mode I most fear is an over-confident or fabricated number reaching a cabinet paper.

## Slide 3: Integrated solution: one question in, one audited brief out

(1.5 min) Walk the seven steps: scope check, profile/peers/notes, scenario simulation, neural second opinion, draft + grounding audit, human approval, structured JSON answer. Every number in the answer traces to a tool call.

## Slide 4: Architecture: four prior projects, four layers

(1.5 min) Colours map layers to prior projects. Evidence layer (Data Science), second opinion (Deep Learning), brief generation (Generative AI), control layer (Agentic Workflows). The control layer decides when each may be used.

## Slide 5: Project 1 - Data Science Blog Post -> evidence layer

(1.5 min) The blog post's methodological lessons are now hard constraints in code. The list of mis-predicted countries is the most valuable carry-over: it tells the agent when to trust itself less.

## Slide 6: Project 2 - Deep Learning Systems -> second opinion & uncertainty

(1.5 min) Be candid: I kept the neural model because the experiment was informative, not because it won. The disagreement signal is the useful product; the dropout spread on its own would have been decorative.

## Slide 7: Project 3 - Generative AI -> brief writer + grounding audit

(1.5 min) Explain why the audit is strict and why I fixed a false rejection by enriching the bundle rather than loosening the check. Mention that a silent fallback is not governance.

## Slide 8: Project 4 - Agentic Workflows -> control layer

(1 min) Almost unchanged from the Research Triage Assistant; what is new is that the tools wrap the three other layers and the approval gate is coupled to the machine audit.

## Slide 9: Key tradeoffs

(1.5 min) For each tradeoff say what was given up and what was bought. The through-line: I traded fluency and flexibility for traceability, because the audience is a public institution.

## Slide 10: Ethics, governance and responsible use

(1.5 min) Tie each risk to a concrete mechanism in the code and to a framework citation. Mention the planted lobbyist injection in the Cambodia notes that is flagged and ignored.

## Slide 11: Evaluation: 11 scenarios, 13 tests

(1.5 min) Emphasise that the evaluation changed the system. Low confidence on Nigeria is the system working, not failing: Nigeria is one of the countries the blog post identified as systematically over-predicted.

## Slide 12: Limitations

(1 min) Own the limits plainly. These are also printed in the notebook's boundaries section.

## Slide 13: Professional relevance and next steps

(1 min) Close by connecting to employer expectations and invite questions. Total ~15 minutes.
