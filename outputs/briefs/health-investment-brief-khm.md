# Health-investment options for Cambodia (KHM), 2022 data
*Prepared for: ministry of health. Evidence: World Bank development indicators; gradient-boosting life-expectancy model (grouped-CV MAE 2.4 years).*

## Where the country stands
Cambodia (East Asia & Pacific, Lower middle income) recorded a life expectancy of 70.5 years. Given its indicator profile the model expects 70.4 years; the gap of 0.1 years reflects factors the indicators do not capture (conflict, epidemics, data quality).
Peer countries in the same region and income group (7 countries) have a median life expectancy of 67.4 years.
The strongest wealth-adjusted over-performer is Solomon Islands (70.4 years, +3.7 versus the model).

## Scenario comparison
- **A: Water, sanitation & electricity** (basic_water_pct 80.86 -> 95.0, basic_sanitation_pct 77.06 -> 90.0, electricity_access_pct 92.3 -> 95.0): predicted 71.2 years, a change of +0.8 years versus the modelled baseline.
  - Caveat: Predicted change (+0.8y) is small relative to model MAE (2.4y).
- **B: Double health spending per capita** (health_exp_per_capita 103.9 -> 207.8): predicted 70.4 years, a change of +0.0 years versus the modelled baseline.
  - Caveat: Predicted change (+0.0y) is small relative to model MAE (2.4y).

## Reading the evidence
The largest modelled gain comes from **A: Water, sanitation & electricity** (+0.8 years). Differences smaller than the model error of 2.4 years should not be treated as meaningful. The model is associational: it describes countries that already have these indicator levels and cannot prove that a programme would cause the change.

## Confidence
A regularised neural network gives an independent estimate of 70.1 years (Monte-Carlo dropout spread 4.27 years) against 70.4 years from the boosting model; overall confidence is **normal**.

## Recommended next steps
Validate the indicator values with the national statistics office, cost each option, and pilot before scaling. This brief was drafted by an AI assistant and must be reviewed by a qualified analyst before use.
