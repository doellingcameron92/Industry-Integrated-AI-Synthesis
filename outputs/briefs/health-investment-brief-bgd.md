# Health-investment options for Bangladesh (BGD), 2022 data
*Prepared for: ministry of health. Evidence: World Bank development indicators; gradient-boosting life-expectancy model (grouped-CV MAE 2.4 years).*

## Where the country stands
Bangladesh (South Asia, Lower middle income) recorded a life expectancy of 74.3 years. Given its indicator profile the model expects 71.1 years; the gap of 3.2 years reflects factors the indicators do not capture (conflict, epidemics, data quality).
Peer countries in the same region and income group (3 countries) have a median life expectancy of 71.7 years.
The strongest wealth-adjusted over-performer is India (71.7 years, +0.7 versus the model).

## Scenario comparison
- **A: Water, sanitation & electricity** (basic_sanitation_pct 63.65 -> 90.0): predicted 72.7 years, a change of +1.6 years versus the modelled baseline.
- **B: Double health spending per capita** (health_exp_per_capita 56.62 -> 113.2): predicted 71.1 years, a change of -0.0 years versus the modelled baseline.
  - Caveat: Predicted change (-0.0y) is small relative to model MAE (2.4y).

## Reading the evidence
The largest modelled gain comes from **A: Water, sanitation & electricity** (+1.6 years). Differences smaller than the model error of 2.4 years should not be treated as meaningful. The model is associational: it describes countries that already have these indicator levels and cannot prove that a programme would cause the change.

## Confidence
A regularised neural network gives an independent estimate of 72.3 years (Monte-Carlo dropout spread 4.59 years) against 71.1 years from the boosting model; overall confidence is **low**.

## Recommended next steps
Validate the indicator values with the national statistics office, cost each option, and pilot before scaling. This brief was drafted by an AI assistant and must be reviewed by a qualified analyst before use.
