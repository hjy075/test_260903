from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.gate05 import gate05_decision, standardized_mean_difference, window_bounds

CURRENT_WINDOW_WEEKS = 13
CURRENT_MIN_WEEKS = 10
HORIZONS = (26, 52)
ORIGIN_COUNT = 6
ORIGIN_STEP_WEEKS = 26
FEATURES = (
    "log_mean_sales",
    "zero_rate",
    "momentum",
    "age_weeks",
    "history_weeks",
)


def locate(data_dir: Path, name: str) -> Path:
    matches = list(data_dir.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {name} under {data_dir}, found {len(matches)}")
    return matches[0]


def load_m5(data_dir: Path):
    calendar = pd.read_csv(locate(data_dir, "calendar.csv"), parse_dates=["date"])
    prices = pd.read_csv(
        locate(data_dir, "sell_prices.csv"),
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    sales = pd.read_csv(locate(data_dir, "sales_train_evaluation.csv"))
    return calendar, prices, sales


def canonicalize_sales(sales: pd.DataFrame):
    store_levels = sorted(sales["store_id"].unique().tolist())
    item_levels = sorted(sales["item_id"].unique().tolist())
    n_items = len(item_levels)
    n_stores = len(store_levels)
    expected = n_items * n_stores
    if len(sales) != expected:
        raise ValueError(f"Expected complete item-store grid of {expected}, found {len(sales)}")

    store_codes = pd.Categorical(sales["store_id"], categories=store_levels).codes
    item_codes = pd.Categorical(sales["item_id"], categories=item_levels).codes
    series_idx = store_codes.astype(np.int64) * n_items + item_codes.astype(np.int64)
    order = np.argsort(series_idx)
    sorted_idx = series_idx[order]
    if not np.array_equal(sorted_idx, np.arange(expected)):
        raise ValueError("M5 sales rows do not form the expected complete canonical grid")

    sales = sales.iloc[order].reset_index(drop=True)
    return sales, store_levels, item_levels


def build_availability(prices, calendar, store_levels, item_levels):
    week_table = (
        calendar[["wm_yr_wk", "date"]]
        .groupby("wm_yr_wk", as_index=False)
        .agg(week_start=("date", "min"), week_end=("date", "max"))
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    week_table["week_idx"] = np.arange(len(week_table), dtype=np.int32)
    week_to_idx = dict(zip(week_table["wm_yr_wk"].astype(int), week_table["week_idx"].astype(int)))

    store_code = pd.Categorical(prices["store_id"].astype(str), categories=store_levels).codes
    item_code = pd.Categorical(prices["item_id"].astype(str), categories=item_levels).codes
    if (store_code < 0).any() or (item_code < 0).any():
        raise ValueError("sell_prices contains store/item IDs absent from sales data")
    n_items = len(item_levels)
    series_idx = store_code.astype(np.int64) * n_items + item_code.astype(np.int64)
    week_idx = prices["wm_yr_wk"].map(week_to_idx).to_numpy()
    if pd.isna(week_idx).any():
        raise ValueError("sell_prices contains week IDs absent from calendar")
    week_idx = week_idx.astype(np.int32)

    avail = np.zeros((len(store_levels) * len(item_levels), len(week_table)), dtype=np.uint8)
    avail[series_idx, week_idx] = 1
    return avail, week_table, week_to_idx


def day_metadata(calendar: pd.DataFrame, day_cols: list[str], week_to_idx: dict[int, int]):
    lookup = calendar.set_index("d")
    week_idx = np.array([week_to_idx[int(lookup.at[d, "wm_yr_wk"])] for d in day_cols], dtype=np.int32)
    return week_idx


def select_origins(last_sales_week_idx: int, n_weeks: int):
    latest = min(last_sales_week_idx, n_weeks - 1) - max(HORIZONS)
    candidates = [latest - ORIGIN_STEP_WEEKS * k for k in range(ORIGIN_COUNT - 1, -1, -1)]
    origins = [x for x in candidates if x >= CURRENT_WINDOW_WEEKS]
    if len(origins) < 4:
        raise ValueError(f"Too few valid origins: {origins}")
    return origins


def first_observed_week(avail: np.ndarray):
    has_any = avail.any(axis=1)
    first = np.full(avail.shape[0], -1, dtype=np.int32)
    first[has_any] = np.argmax(avail[has_any] > 0, axis=1)
    return first


def compute_features(
    sales_values: np.ndarray,
    day_week_idx: np.ndarray,
    avail: np.ndarray,
    first_week: np.ndarray,
    origin: int,
):
    start, end = window_bounds(origin, CURRENT_WINDOW_WEEKS, -1)
    day_pos = np.where((day_week_idx >= start) & (day_week_idx < end))[0]
    if len(day_pos) == 0:
        raise ValueError(f"No daily observations for origin {origin}")

    weeks = day_week_idx[day_pos]
    y = sales_values[:, day_pos]
    daily_avail = avail[:, weeks].astype(bool)
    obs_days = daily_avail.sum(axis=1)

    observed_sales_sum = np.where(daily_avail, y, 0.0).sum(axis=1)
    mean_sales = np.divide(
        observed_sales_sum,
        obs_days,
        out=np.full(y.shape[0], np.nan, dtype=np.float64),
        where=obs_days > 0,
    )
    zero_count = ((y == 0) & daily_avail).sum(axis=1)
    zero_rate = np.divide(
        zero_count,
        obs_days,
        out=np.full(y.shape[0], np.nan, dtype=np.float64),
        where=obs_days > 0,
    )

    weekly = np.zeros((y.shape[0], CURRENT_WINDOW_WEEKS), dtype=np.float32)
    for j, week in enumerate(range(start, end)):
        pos = np.where(weeks == week)[0]
        if len(pos):
            weekly[:, j] = y[:, pos].sum(axis=1)
    first4 = weekly[:, :4].mean(axis=1)
    last4 = weekly[:, -4:].mean(axis=1)
    momentum = np.log1p(last4) - np.log1p(first4)

    history_weeks = avail[:, :origin].sum(axis=1).astype(np.float32)
    age_weeks = np.where(first_week >= 0, origin - first_week, np.nan).astype(np.float32)

    return pd.DataFrame(
        {
            "log_mean_sales": np.log1p(mean_sales),
            "zero_rate": zero_rate,
            "momentum": momentum,
            "age_weeks": age_weeks,
            "history_weeks": history_weeks,
        }
    )


def cv_auc(x: pd.DataFrame, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or min(np.bincount(y.astype(int))) < 5:
        return float("nan")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(model, x, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    return float(roc_auc_score(y, proba))


def summarize_groups(features, current_mask, continuation_mask, origin, horizon, week_table):
    idx = np.where(current_mask)[0]
    y = continuation_mask[idx].astype(np.uint8)
    f = features.iloc[idx].reset_index(drop=True)

    n = len(idx)
    n_cont = int(y.sum())
    n_non = n - n_cont
    cont_share = n_cont / n if n else float("nan")
    minority_share = min(cont_share, 1 - cont_share) if n else float("nan")

    summary = {
        "origin_idx": origin,
        "origin_week": int(week_table.at[origin, "wm_yr_wk"]),
        "origin_date": str(pd.Timestamp(week_table.at[origin, "week_start"]).date()),
        "horizon_weeks": horizon,
        "n_current": n,
        "n_continuing": n_cont,
        "n_noncontinuing": n_non,
        "continuing_share": cont_share,
        "minority_share": minority_share,
        "cv_auc": cv_auc(f[list(FEATURES)], y),
    }

    sep_rows = []
    for feature in FEATURES:
        non = f.loc[y == 0, feature].dropna().to_numpy()
        cont = f.loc[y == 1, feature].dropna().to_numpy()
        smd = standardized_mean_difference(non, cont)
        sep_rows.append(
            {
                "origin_idx": origin,
                "origin_week": summary["origin_week"],
                "origin_date": summary["origin_date"],
                "horizon_weeks": horizon,
                "feature": feature,
                "noncontinuing_mean": float(np.nanmean(non)) if len(non) else float("nan"),
                "continuing_mean": float(np.nanmean(cont)) if len(cont) else float("nan"),
                "noncontinuing_median": float(np.nanmedian(non)) if len(non) else float("nan"),
                "continuing_median": float(np.nanmedian(cont)) if len(cont) else float("nan"),
                "smd_continuing_minus_non": float(smd),
                "abs_smd": float(abs(smd)),
            }
        )
    return summary, sep_rows


def threshold_sensitivity(avail, origins, week_table):
    rows = []
    for origin in origins:
        cur_start, cur_end = window_bounds(origin, CURRENT_WINDOW_WEEKS, -1)
        current = avail[:, cur_start:cur_end].sum(axis=1) >= CURRENT_MIN_WEEKS
        for min_weeks in (7, 10, 13):
            fut_start, fut_end = window_bounds(origin, CURRENT_WINDOW_WEEKS, 52)
            cont = avail[:, fut_start:fut_end].sum(axis=1) >= min_weeks
            vals = cont[current]
            rows.append(
                {
                    "origin_idx": origin,
                    "origin_week": int(week_table.at[origin, "wm_yr_wk"]),
                    "origin_date": str(pd.Timestamp(week_table.at[origin, "week_start"]).date()),
                    "future_min_weeks": min_weeks,
                    "n_current": int(current.sum()),
                    "continuing_share": float(vals.mean()) if len(vals) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def render_report(summary_df, sep_df, decision, sensitivity_df):
    med_smd = (
        sep_df[sep_df["horizon_weeks"] == 52]
        .groupby("feature")["abs_smd"]
        .median()
        .sort_values(ascending=False)
    )
    lines = [
        "# Gate-0.5 result: Retail forecast survivorship",
        "",
        "**Research question:** Do Surviving Products Make Retail Forecasts Look Better?  ",
        "**국문:** 살아남은 상품만 보면 소매 수요예측 성능이 더 좋아 보이는가?",
        "",
        "## Pre-registered Gate-0.5 design",
        "",
        "- Unit: M5 item-store series.",
        "- Current assortment: price observed in at least 10 of the 13 completed weeks before each origin.",
        "- Future continuation: price observed in at least 10 of the 13 weeks ending 26 or 52 weeks after the origin.",
        "- Six rolling origins, 26 weeks apart.",
        "- No forecasting models are trained at this gate.",
        "- HARD KILL if the median 52-week minority-group share is below 5%.",
        "- PASS if the median minority-group share is at least 10% and either max median |SMD| >= 0.20 or median 5-fold AUC >= 0.60.",
        "",
        "## Composition by origin",
        "",
        summary_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 52-week feature separation (median absolute SMD across origins)",
        "",
        med_smd.rename("median_abs_smd").to_frame().to_markdown(floatfmt=".3f"),
        "",
        "## 52-week continuation threshold sensitivity",
        "",
        sensitivity_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Gate decision",
        "",
        f"**{decision['status']}**",
        "",
        "```json",
        json.dumps(decision, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Interpretation guardrail",
        "",
        "M5 has no permanent-discontinuation label. `Continuing` here means sustained selling presence inferred from weekly price observations, not confirmed permanent product survival or discontinuation.",
    ]
    return "\n".join(lines)


def run(data_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    calendar, prices, sales = load_m5(data_dir)
    sales, store_levels, item_levels = canonicalize_sales(sales)
    avail, week_table, week_to_idx = build_availability(prices, calendar, store_levels, item_levels)

    day_cols = sorted([c for c in sales.columns if c.startswith("d_")], key=lambda x: int(x.split("_")[1]))
    day_week_idx = day_metadata(calendar, day_cols, week_to_idx)
    sales_values = sales[day_cols].to_numpy(dtype=np.float32, copy=True)
    last_sales_week_idx = int(day_week_idx.max())
    origins = select_origins(last_sales_week_idx, avail.shape[1])
    first_week = first_observed_week(avail)

    summary_rows = []
    sep_rows = []
    for origin in origins:
        cur_start, cur_end = window_bounds(origin, CURRENT_WINDOW_WEEKS, -1)
        current_mask = avail[:, cur_start:cur_end].sum(axis=1) >= CURRENT_MIN_WEEKS
        features = compute_features(sales_values, day_week_idx, avail, first_week, origin)

        for horizon in HORIZONS:
            fut_start, fut_end = window_bounds(origin, CURRENT_WINDOW_WEEKS, horizon)
            continuation = avail[:, fut_start:fut_end].sum(axis=1) >= CURRENT_MIN_WEEKS
            summary, sep = summarize_groups(features, current_mask, continuation, origin, horizon, week_table)
            summary_rows.append(summary)
            sep_rows.extend(sep)

    summary_df = pd.DataFrame(summary_rows)
    sep_df = pd.DataFrame(sep_rows)
    sensitivity_df = threshold_sensitivity(avail, origins, week_table)

    s52 = summary_df[summary_df["horizon_weeks"] == 52]
    feature_medians = (
        sep_df[sep_df["horizon_weeks"] == 52]
        .groupby("feature")["abs_smd"]
        .median()
        .to_dict()
    )
    decision = gate05_decision(
        minority_shares=s52["minority_share"].tolist(),
        median_abs_smd_by_feature={k: float(v) for k, v in feature_medians.items()},
        aucs=s52["cv_auc"].dropna().tolist(),
    )
    decision["origins"] = [int(x) for x in origins]
    decision["n_series"] = int(avail.shape[0])
    decision["n_weeks"] = int(avail.shape[1])

    summary_df.to_csv(out_dir / "gate05_summary.csv", index=False)
    sep_df.to_csv(out_dir / "gate05_feature_separation.csv", index=False)
    sensitivity_df.to_csv(out_dir / "gate05_threshold_sensitivity.csv", index=False)
    (out_dir / "gate05_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "gate05_report.md").write_text(render_report(summary_df, sep_df, decision, sensitivity_df), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
