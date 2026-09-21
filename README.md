# Smart E-Learning Assistant — AI API

FastAPI service backing the Smart E-Learning Assistant Flutter app's two AI
features:

- `POST /predict/performance` — pass/fail risk prediction from a student's
  demographic profile and their assessment/engagement activity **so far**
  (see "Point-in-course prediction" below).
- `POST /feedback/analyze` — sentiment + key-issue extraction from free-text
  student feedback, via Gemini.

## Setup

```
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY
```

The prediction endpoint needs trained model artifacts in `models/`:
`pass_fail_pipeline.pkl`, `feature_stats.json`, `metadata.json`. These are
produced by running `../student-performance-prediction.ipynb` end to end —
run the notebook first.

```
uvicorn app.main:app --reload
```

## Project layout

- `student_risk/features.py` — the single feature-engineering implementation
  shared by the notebook (training) and this API (serving): turns one
  student's *currently known* state into a feature vector. Never edit the
  feature logic in one place without the other; import from here in both.
- `student_risk/snapshots.py` — notebook-only, **not imported by the API**.
  Builds point-in-course training rows from OULAD's full-hindsight records;
  the live API never needs this since a live request already only describes
  the present moment.
- `app/services/performance_model.py` — loads the saved pipeline once and
  validates its expected feature columns against `metadata.json` at startup.
- `app/services/gemini_feedback.py` — Gemini call with structured JSON
  output, timeout/retry, a graceful fallback on repeated failure, and a
  short-TTL cache to avoid paying for duplicate submissions.

## Point-in-course prediction

The model is trained on snapshots taken at several points through a course
(10% / 25% / 40% / 55% / 70% / 85% / 100% complete), not just full-course
totals — so it can be queried mid-course without misreading naturally-small
early activity as a risk signal. `/predict/performance` requires
`elapsed_fraction` (0.0-1.0, how far into the course this request is) along
with `total_score`/`avg_score`/`num_assessments`/`total_clicks` **observed so
far**, not full-course totals, plus `recent_clicks` — clicks in just the
last 14 days, alongside the full-so-far `total_clicks` — giving the model a
rough trend signal (rising vs. falling engagement) without needing a real
time series. The service snaps `elapsed_fraction` to the
nearest bucket the model was actually trained on and uses that bucket's
statistics (`feature_stats.json`'s `by_cutoff`) to judge whether the
reported activity is high or low *for that stage of the course* — comparing
a week-2 snapshot against week-30 norms would make almost anyone look
disengaged.

`region`, `code_module`, and `code_presentation` are not part of the request
— they were dropped as model inputs; the model must generalize beyond the
specific OULAD courses and UK regions it was trained on.

## Known limitation: OULAD calibration

Engagement features (`total_clicks` and everything derived from it,
`engagement_intensity`/`low_activity_flag`/`high_click_but_low_score_flag`)
are calibrated **per elapsed_fraction bucket** using OULAD's own
click-logging volume at each stage of a course. If this model is ever
pointed at a different LMS, or a course whose engagement pattern differs a
lot from OULAD's, these features — and the probabilities the model outputs —
should not be trusted without re-checking `models/feature_stats.json`'s
per-bucket statistics against the new platform, and likely retraining.

## Data leakage safeguards

- `feature_stats.json` (per-bucket means/medians used by engineered
  features) and the pipeline's `OneHotEncoder` are both fit on the training
  split only, never on test data or the full dataset.
- The train/test split happens on `id_student`, not on individual snapshot
  rows — each student contributes up to 7 rows (one per cutoff), and
  splitting by row would let the same student's early and late snapshots
  land on opposite sides of the split.
- A snapshot never knows about a student's withdrawal date unless it had
  already happened as of that snapshot's cutoff — an eventual dropout is not
  visible to an earlier, still-enrolled snapshot.
- `region` / `code_module` / `code_presentation` are excluded from the model
  entirely (not even accepted by the API) — the model must generalize
  beyond the specific OULAD courses and regions it was trained on.
