from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class PerformancePredictionRequest(BaseModel):
    gender: Literal["M", "F"]
    highest_education: Literal[
        "No Formal quals",
        "Lower Than A Level",
        "A Level or Equivalent",
        "HE Qualification",
        "Postgraduate Qualification",
    ]
    imd_band: Optional[str] = None
    age_band: Literal["0-35", "35-55", "55<="]
    num_of_prev_attempts: int = Field(ge=0)
    studied_credits: int = Field(gt=0)
    disability: Literal["Y", "N"]

    # How far into the course this snapshot is, as a fraction (0.0 = just
    # started, 1.0 = course complete). The model was trained on snapshots
    # bucketed at several points in a course, not just the end, specifically
    # so it can be called mid-course without misreading naturally-small
    # early activity as a risk signal — see feature_stats.json's per-bucket
    # statistics. The service snaps this to the nearest trained bucket.
    elapsed_fraction: float = Field(ge=0.0, le=1.0)

    # Activity observed SO FAR, up to elapsed_fraction — not full-course
    # totals.
    total_score: float = Field(ge=0)
    avg_score: float = Field(ge=0)
    num_assessments: int = Field(ge=0)
    total_clicks: int = Field(ge=0)
    # Clicks in the last 14 days only, as a rough trend signal (rising vs.
    # falling engagement) alongside the full-so-far total_clicks above.
    recent_clicks: int = Field(ge=0)

    date_registration: int
    # Only set this if the student has ALREADY unregistered as of now —
    # leave null otherwise. Never a future/anticipated date.
    date_unregistration: Optional[int] = None

    @model_validator(mode="after")
    def _recent_clicks_within_total(self):
        if self.recent_clicks > self.total_clicks:
            raise ValueError("recent_clicks cannot exceed total_clicks")
        return self


class PerformancePredictionResponse(BaseModel):
    prediction: Literal["Pass", "Fail"]
    probability: float
    risk_level: Literal["Low", "Medium", "High"]


class FeedbackAnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class FeedbackAnalysisResponse(BaseModel):
    sentiment: Literal["Positive", "Negative", "Neutral", "Unknown"]
    issues: list[str]
    summary: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_schema_valid: bool
    gemini_configured: bool
