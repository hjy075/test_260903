# Retail Forecast Survivorship Gate-0.5

Research spike for the question:

**Do Surviving Products Make Retail Forecasts Look Better?**  
**살아남은 상품만 보면 소매 수요예측 성능이 더 좋아 보이는가?**

This repository runs a low-cost Gate-0.5 falsification test on the public M5 retail dataset before training forecasting models.

## What Gate-0.5 tests

The gate asks whether, at multiple forecast origins, item-store series that continue to be sold 26/52 weeks later are:

1. sufficiently numerous to support a real comparison, and
2. measurably different **using only information available at the forecast origin** from series whose selling presence does not continue.

No forecasting model is trained at this stage.

## Definitions fixed before seeing the M5 result

- Unit: M5 item-store series.
- Current assortment: price observed in at least **10 of the 13 completed weeks** before each origin.
- Future continuation: price observed in at least **10 of the 13 weeks ending 26 or 52 weeks after** the origin.
- Rolling origins: **6 origins**, spaced **26 weeks** apart.
- Pre-origin features: log mean sales, zero-sales rate, recent sales momentum, series age, observed-history weeks.
- Diagnostic predictability: 5-fold cross-validated logistic-regression AUC using only those pre-origin features.
- M5 has no permanent-discontinuation label. `Continuing` means sustained selling presence inferred from weekly price observations; it does **not** assert permanent product survival/discontinuation.

## Pre-registered decision rule

The primary gate uses the 52-week definition.

- **HARD_KILL**: median minority-group share across origins < 5%.
- **WEAK_HOLD**: median minority-group share is 5% to <10%.
- **PASS_GATE_0_5**: median minority-group share >=10% **and** either:
  - maximum median absolute standardized mean difference (|SMD|) across the pre-origin features >=0.20, or
  - median 5-fold AUC >=0.60.
- Otherwise: **HOLD**.

A 7/13 and 13/13 future-presence threshold is also reported as sensitivity analysis; it does not replace the pre-registered 10/13 main rule.

## Outputs

`results/` contains:

- `gate05_summary.csv`
- `gate05_feature_separation.csv`
- `gate05_threshold_sensitivity.csv`
- `gate05_decision.json`
- `gate05_report.md`

## Reproduction

The GitHub Actions workflow downloads the public M5 archive at runtime from Zenodo, with a public GitHub mirror as fallback. Data are not committed to this repository.

Local execution after placing/extracting the M5 CSVs under `data/m5/`:

```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/run_gate05.py --data-dir data/m5 --out-dir results
```

Unit tests:

```bash
python -m unittest discover -s tests -v
```
