"""
Historical training-snapshot construction — notebook-only, never imported by
the API.

The live API only ever describes the present moment, so it never needs to
simulate "what was known as of day X" from full-hindsight data. Building
*training* rows that are valid for point-in-course prediction does need
that simulation: build_snapshot_table() takes full-course OULAD records and
produces several rows per student attempt, each one truncated to only the
activity that had actually happened by a given fraction of the course.

Two temporal-correctness rules this enforces:
- Assessment/VLE activity is aggregated only from events on or before each
  snapshot's cutoff day — never the student's full-course totals.
- date_unregistration is only treated as "known" in a snapshot if the
  student had actually unregistered by that snapshot's cutoff day.
  Otherwise the snapshot must treat them as still enrolled, even though the
  historical record shows they withdrew later — using that later date would
  leak future information into an earlier snapshot.

Also computes recent_clicks per snapshot: clicks in the RECENT_WINDOW_DAYS
immediately before that snapshot's cutoff day, as opposed to total_clicks
(everything since the course started). This is what lets
student_risk.features derive recent_click_share, a rough trend signal.
"""

import numpy as np
import pandas as pd

CUTOFF_FRACTIONS = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
RECENT_WINDOW_DAYS = 14

_ATTEMPT_KEYS = ["id_student", "code_module", "code_presentation"]


def build_snapshot_table(
    student_base: pd.DataFrame,
    student_reg: pd.DataFrame,
    student_assess: pd.DataFrame,
    student_vle: pd.DataFrame,
    courses: pd.DataFrame,
    cutoffs: list = CUTOFF_FRACTIONS,
) -> pd.DataFrame:
    """
    student_base: one row per (id_student, code_module, code_presentation)
        attempt, with demographics and target_pass/target_cgpa already
        merged in (but not yet registration dates or activity).
    student_assess: raw per-assessment-submission rows (id_student,
        id_assessment, date_submitted, score) — NOT pre-aggregated.
        Assessment records aren't attributable to a specific module
        attempt without also loading assessments.csv (out of scope here),
        so a student's assessment history is pooled across their attempts —
        the same simplification the non-snapshot version of this notebook
        already made; this only adds the cutoff-day filter on top of it.
    student_vle: raw per-day click rows (id_student, code_module,
        code_presentation, date, sum_click) — properly scoped per attempt
        since studentVle does carry the module keys.
    courses: code_module, code_presentation, module_presentation_length.

    Returns one row per (attempt, cutoff) — len(student_base) * len(cutoffs)
    rows total.
    """
    base = student_base.merge(
        student_reg[["id_student", "code_module", "code_presentation", "date_registration", "date_unregistration"]],
        on=_ATTEMPT_KEYS,
        how="left",
    )
    base = base.merge(
        courses[["code_module", "code_presentation", "module_presentation_length"]],
        on=["code_module", "code_presentation"],
        how="left",
    )
    base["module_presentation_length"] = base["module_presentation_length"].fillna(
        base["module_presentation_length"].median()
    )

    snapshots = []
    for cutoff in cutoffs:
        snap = base.copy()
        snap["elapsed_fraction"] = cutoff
        snap["cutoff_day"] = cutoff * snap["module_presentation_length"]

        assess_with_cutoff = student_assess.merge(
            snap[_ATTEMPT_KEYS + ["cutoff_day"]], on="id_student", how="inner"
        )
        assess_observed = assess_with_cutoff[
            assess_with_cutoff["date_submitted"] <= assess_with_cutoff["cutoff_day"]
        ]
        assess_agg = (
            assess_observed.groupby(_ATTEMPT_KEYS)
            .agg(
                total_score=("score", "sum"),
                avg_score=("score", "mean"),
                num_assessments=("id_assessment", "count"),
            )
            .reset_index()
        )

        vle_with_cutoff = student_vle.merge(snap[_ATTEMPT_KEYS + ["cutoff_day"]], on=_ATTEMPT_KEYS, how="inner")
        vle_observed = vle_with_cutoff[vle_with_cutoff["date"] <= vle_with_cutoff["cutoff_day"]]
        vle_agg = vle_observed.groupby(_ATTEMPT_KEYS).agg(total_clicks=("sum_click", "sum")).reset_index()

        vle_recent = vle_observed[
            vle_observed["date"] > (vle_observed["cutoff_day"] - RECENT_WINDOW_DAYS)
        ]
        vle_recent_agg = (
            vle_recent.groupby(_ATTEMPT_KEYS).agg(recent_clicks=("sum_click", "sum")).reset_index()
        )

        snap = snap.merge(assess_agg, on=_ATTEMPT_KEYS, how="left")
        snap = snap.merge(vle_agg, on=_ATTEMPT_KEYS, how="left")
        snap = snap.merge(vle_recent_agg, on=_ATTEMPT_KEYS, how="left")
        snap["total_score"] = snap["total_score"].fillna(0)
        snap["avg_score"] = snap["avg_score"].fillna(0)
        snap["num_assessments"] = snap["num_assessments"].fillna(0)
        snap["total_clicks"] = snap["total_clicks"].fillna(0)
        snap["recent_clicks"] = snap["recent_clicks"].fillna(0)

        already_unregistered = snap["date_unregistration"].notna() & (
            snap["date_unregistration"] <= snap["cutoff_day"]
        )
        snap["date_unregistration"] = np.where(already_unregistered, snap["date_unregistration"], -1)

        snapshots.append(snap.drop(columns=["cutoff_day", "module_presentation_length"]))

    return pd.concat(snapshots, ignore_index=True)
