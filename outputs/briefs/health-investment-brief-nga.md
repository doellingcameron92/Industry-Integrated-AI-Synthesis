# Health-investment options for Nigeria (NGA), 2022 data
*Prepared for: ministry of health. Evidence: World Bank development indicators; gradient-boosting life-expectancy model (grouped-CV MAE 2.4 years).*

## Where the country stands
Nigeria (Sub-Saharan Africa, Lower middle income) recorded a life expectancy of 54.1 years. Given its indicator profile the model expects 64.4 years; the gap of -10.3 years reflects factors the indicators do not capture (conflict, epidemics, data quality).
Peer countries in the same region and income group (19 countries) have a median life expectancy of 64.2 years.
The strongest wealth-adjusted over-performer is Tanzania (66.9 years, +3.4 versus the model).

## Scenario comparison
- **A: Infrastructure & services package** (basic_sanitation_pct 46.57 -> 90.0, basic_water_pct 79.46 -> 95.0): predicted 67.4 years, a change of +3.0 years versus the modelled baseline.
- **B: Double health spending per capita** (health_exp_per_capita 89.63 -> 179.2): predicted 64.4 years, a change of +0.0 years versus the modelled baseline.
  - Caveat: Predicted change (+0.0y) is small relative to model MAE (2.4y).

## Reading the evidence
The largest modelled gain comes from **A: Infrastructure & services package** (+3.0 years). Differences smaller than the model error of 2.4 years should not be treated as meaningful. The model is associational: it describes countries that already have these indicator levels and cannot prove that a programme would cause the change.

## Confidence
A regularised neural network gives an independent estimate of 61.8 years (Monte-Carlo dropout spread 3.93 years) against 64.4 years from the boosting model; overall confidence is **low**.
Evidence-quality flags: country was a large-residual case in grouped cross-validation; hold-out error 6.8y exceeds twice the model MAE.

## Recommended next steps
Validate the indicator values with the national statistics office, cost each option, and pilot before scaling. This brief was drafted by an AI assistant and must be reviewed by a qualified analyst before use.
