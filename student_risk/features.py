"""
Shared feature-engineering logic for the student pass/fail risk model.

This module is the single source of truth for turning one student's
*currently known* state into the feature vector the model expects. Both the
training notebook (student-performance-prediction.ipynb, one row per
point-in-course snapshot) and the FastAPI service (one row per live request)
import this module, so training and serving can never drift apart.

Deliberately kept out of this module:
- code_module / code_presentation / region are never turned into model
  features. The model must generalize beyond the specific OULAD courses (and
  their UK regions) it was trained on.
- Population-level statistics (mean/std/median) are never computed inside
  this module — they must come from a training split only (see
  compute_training_stats) and be passed in explicitly, bucketed by
  elapsed_fraction (see below), so a single-row inference request reuses the
  exact numbers seen during training for a comparable point in the course.
- Historical, cutoff-aware snapshot construction (turning full-hindsight
  OULAD records into "as observed at day X" training rows) lives in
  student_risk/snapshots.py, not here — that's a training-data-construction
  concern the live API never faces, since a live request is already only
  ever describing the present moment.

Why bucketed by elapsed_fraction: total_clicks at 10% into a course is
naturally far smaller than at 100%. Comparing every row against one pooled
population average would flag nearly every early-course request as
"low activity" regardless of whether the student is actually behind their
own-stage peers — so engagement stats are computed and looked up per
elapsed_fraction bucket, not globally.

recent_clicks / recent_click_share give the model a rough trend signal
(rising vs. falling engagement) without asking a caller to split their
history into arbitrary before/after windows — recent_clicks is just
"clicks in the last RECENT_WINDOW_DAYS" (see snapshots.py), a number any
backend can compute easily, alongside total_clicks (the full so-far total).
"""

import numpy as np
import pandas as pd

# Raw fields a caller (the training pipeline or an API request) must supply.
RAW_INPUT_COLUMNS = [
    "gender",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "total_score",
    "avg_score",
    "num_assessments",
    "total_clicks",
    "recent_clicks",
    "date_registration",
    "date_unregistration",
    "elapsed_fraction",
]

CATEGORICAL_COLUMNS = ["credit_load_category"]

FINAL_FEATURE_COLUMNS = [
    "gender_m",
    "edu_level",
    "imd_num",
    "age_num",
    "num_of_prev_attempts",
    "studied_credits",
    "disability_flag",
    "total_score",
    "avg_score",
    "num_assessments",
    "total_clicks",
    "recent_clicks",
    "recent_click_share",
    "elapsed_fraction",
    "registration_lead_days",
    "withdrawn_by_snapshot_flag",
    "registration_duration",
    "clicks_per_credit",
    "score_per_assess",
    "assessment_density",
    "score_per_credit",
    "activity_efficiency",
    "credit_load_category",
    "engagement_intensity",
    "low_activity_flag",
    "late_registration_flag",
    "high_click_but_low_score_flag",
]

NUMERIC_COLUMNS = [c for c in FINAL_FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

_AGE_MIDPOINT = {"0-35": 17.5, "35-55": 45.0, "55<=": 60.0}

_EDUCATION_LEVEL = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Postgraduate Qualification": 4,
}
_EDUCATION_LEVEL_DEFAULT = 2  # "A Level or Equivalent" — neutral fallback for an unrecognized value


def _imd_band_to_midpoint(value) -> float:
    """Turn an IMD band string like '10-20%' into its numeric midpoint."""
    try:
        if pd.isna(value) or value == "Unknown":
            return np.nan
        cleaned = str(value).replace("%", "")
        parts = cleaned.replace("<=", "").split("-")
        parts = [p for p in parts if p != ""]
        if len(parts) == 1:
            return float(parts[0])
        return (float(parts[0]) + float(parts[-1])) / 2.0
    except (ValueError, TypeError):
        return np.nan


def nearest_cutoff(fraction: float, cutoffs: list) -> float:
    """The trained elapsed_fraction bucket closest to an arbitrary fraction."""
    return min(cutoffs, key=lambda c: abs(c - fraction))


def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-row feature engineering that needs nothing beyond the row itself —
    no dataset-wide statistics involved. Safe to call on a single-row
    DataFrame (an API request) or a full training set. `date_unregistration`
    and `elapsed_fraction` must already reflect "as observed at this
    snapshot" — see student_risk/snapshots.py for how training rows ensure
    that; a live API request already only describes the present moment.
    """
    df = df.copy()
    df["date_unregistration"] = df["date_unregistration"].fillna(-1)
    df["elapsed_fraction"] = df["elapsed_fraction"].clip(0.0, 1.0)
    df["recent_clicks"] = df["recent_clicks"].fillna(0)

    df["imd_num"] = df["imd_band"].apply(_imd_band_to_midpoint)

    df["age_num"] = df["age_band"].astype(str).str.strip().map(_AGE_MIDPOINT)

    df["edu_level"] = (
        df["highest_education"].map(_EDUCATION_LEVEL).fillna(_EDUCATION_LEVEL_DEFAULT).astype(int)
    )

    df["gender_m"] = df["gender"].map({"M": 1, "F": 0}).fillna(0).astype(int)
    df["disability_flag"] = df["disability"].map({"Y": 1, "N": 0}).fillna(0).astype(int)

    df["registration_lead_days"] = -df["date_registration"]
    # 1 = the student had already withdrawn as of this snapshot, 0 = still
    # enrolled at this point (renamed from the old "registered_flag", which
    # read backwards — 1 meant "unregistered", not "is registered").
    df["withdrawn_by_snapshot_flag"] = (df["date_unregistration"] != -1).astype(int)
    df["registration_duration"] = np.where(
        df["date_unregistration"] != -1,
        df["date_unregistration"] - df["date_registration"],
        -1,
    )

    # what share of a student's clicks-so-far happened recently (last
    # RECENT_WINDOW_DAYS, see snapshots.py) -- a rough trend signal (rising
    # vs. falling engagement) without needing a full time series.
    total_clicks_safe = df["total_clicks"].replace(0, np.nan)
    df["recent_click_share"] = (df["recent_clicks"] / total_clicks_safe).fillna(0).clip(0.0, 1.0)

    studied_credits_safe = df["studied_credits"].replace(0, np.nan)
    df["clicks_per_credit"] = (df["total_clicks"] / studied_credits_safe).fillna(0)
    df["assessment_density"] = (df["num_assessments"] / studied_credits_safe).fillna(0)
    df["score_per_credit"] = (df["total_score"] / studied_credits_safe).fillna(0)

    num_assessments_safe = df["num_assessments"].replace(0, np.nan)
    df["score_per_assess"] = (df["total_score"] / num_assessments_safe).fillna(0)

    clicks_per_credit_safe = df["clicks_per_credit"].replace(0, np.nan)
    df["activity_efficiency"] = (df["score_per_assess"] / clicks_per_credit_safe).fillna(0)

    df["credit_load_category"] = pd.cut(
        df["studied_credits"],
        bins=[0, 60, 90, 120],
        labels=["light", "medium", "heavy"],
        include_lowest=True,
    ).astype(str)

    return df


def compute_training_stats(df_train: pd.DataFrame, cutoffs: list) -> dict:
    """
    Compute per-elapsed_fraction-bucket population statistics from the
    training split ONLY. df_train must already have add_row_features applied
    (so total_clicks, total_score, registration_lead_days, imd_num exist)
    and must contain the elapsed_fraction column. Never call this on test
    data or the full dataset — that would leak test-set information into the
    features the model is trained on.
    """
    nearest = df_train["elapsed_fraction"].apply(lambda f: nearest_cutoff(f, cutoffs))

    by_cutoff = {}
    for cutoff in cutoffs:
        bucket = df_train[nearest == cutoff]
        if bucket.empty:
            continue
        by_cutoff[str(cutoff)] = {
            "total_clicks_mean": float(bucket["total_clicks"].mean()),
            "total_clicks_std": float(bucket["total_clicks"].std() or 1.0),
            "total_clicks_median": float(bucket["total_clicks"].median()),
            "total_score_median": float(bucket["total_score"].median()),
            "registration_lead_days_median": float(bucket["registration_lead_days"].median()),
            "imd_num_median": float(bucket["imd_num"].median()),
            "row_count": int(len(bucket)),
        }

    return {"cutoffs": list(cutoffs), "by_cutoff": by_cutoff}


def add_population_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Features defined relative to training-set statistics. Each row is
    matched to its nearest trained elapsed_fraction bucket in `stats`, so a
    dataframe spanning many buckets (the full training set) and a single
    live request (one bucket) are both handled by the same code path.
    """
    df = df.copy()
    cutoffs = stats["cutoffs"]
    nearest = df["elapsed_fraction"].apply(lambda f: nearest_cutoff(f, cutoffs))

    imd_num = df["imd_num"].copy()
    engagement_intensity = pd.Series(np.nan, index=df.index, dtype=float)
    low_activity_flag = pd.Series(0, index=df.index, dtype=int)
    late_registration_flag = pd.Series(0, index=df.index, dtype=int)
    high_click_but_low_score_flag = pd.Series(0, index=df.index, dtype=int)

    for cutoff in cutoffs:
        mask = nearest == cutoff
        bucket_stats = stats["by_cutoff"].get(str(cutoff))
        if not mask.any() or bucket_stats is None:
            continue

        std = bucket_stats["total_clicks_std"] or 1.0
        engagement_intensity.loc[mask] = (
            df.loc[mask, "total_clicks"] - bucket_stats["total_clicks_mean"]
        ) / std
        low_activity_flag.loc[mask] = (
            df.loc[mask, "total_clicks"] < bucket_stats["total_clicks_median"]
        ).astype(int)
        late_registration_flag.loc[mask] = (
            df.loc[mask, "registration_lead_days"] < bucket_stats["registration_lead_days_median"]
        ).astype(int)
        high_click_but_low_score_flag.loc[mask] = (
            (df.loc[mask, "total_clicks"] > bucket_stats["total_clicks_median"])
            & (df.loc[mask, "total_score"] < bucket_stats["total_score_median"])
        ).astype(int)
        imd_num.loc[mask] = df.loc[mask, "imd_num"].fillna(bucket_stats["imd_num_median"])

    df["imd_num"] = imd_num
    df["engagement_intensity"] = engagement_intensity
    df["low_activity_flag"] = low_activity_flag
    df["late_registration_flag"] = late_registration_flag
    df["high_click_but_low_score_flag"] = high_click_but_low_score_flag

    return df


def build_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Full feature pipeline: raw input columns in, model-ready feature columns
    out, in a fixed column order (FINAL_FEATURE_COLUMNS). `stats` must come
    from compute_training_stats(X_train, cutoffs) — see module docstring.
    Never emits code_module / code_presentation / region.
    """
    featured = add_row_features(df)
    featured = add_population_features(featured, stats)

    featured[NUMERIC_COLUMNS] = (
        featured[NUMERIC_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    return featured[FINAL_FEATURE_COLUMNS]
