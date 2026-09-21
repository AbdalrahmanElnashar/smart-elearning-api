import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from student_risk.features import RAW_INPUT_COLUMNS, build_features

from app.config import settings
from app.schemas import PerformancePredictionRequest

logger = logging.getLogger(__name__)

_DUMMY_ROW = {
    "gender": "M",
    "highest_education": "A Level or Equivalent",
    "imd_band": "50-60%",
    "age_band": "0-35",
    "num_of_prev_attempts": 0,
    "studied_credits": 60,
    "disability": "N",
    "total_score": 0.0,
    "avg_score": 0.0,
    "num_assessments": 0,
    "total_clicks": 0,
    "recent_clicks": 0,
    "date_registration": -30,
    "date_unregistration": None,
    "elapsed_fraction": 1.0,
}


class PerformanceModelError(RuntimeError):
    """Raised when a prediction is requested but the model isn't ready to serve."""


class PerformancePredictionResult:
    def __init__(self, prediction: str, probability: float, risk_level: str):
        self.prediction = prediction
        self.probability = probability
        self.risk_level = risk_level


def _risk_level(pass_probability: float) -> str:
    fail_probability = 1.0 - pass_probability
    if fail_probability < 0.3:
        return "Low"
    if fail_probability < 0.6:
        return "Medium"
    return "High"


class PerformanceModelService:
    """
    Loads the trained pass/fail pipeline once and keeps it in memory.

    Validates at startup that build_features() actually produces the columns
    the pipeline was trained on (metadata.json's recorded feature_columns).
    If the notebook and this service ever drift apart, this fails loudly at
    startup instead of returning silently wrong predictions later.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.pipeline = None
        self.stats: Optional[dict] = None
        self.metadata: Optional[dict] = None
        self.loaded = False
        self.schema_valid = False
        self._load()

    def _load(self) -> None:
        pipeline_path = self.model_dir / "pass_fail_pipeline.pkl"
        stats_path = self.model_dir / "feature_stats.json"
        metadata_path = self.model_dir / "metadata.json"

        missing = [p.name for p in (pipeline_path, stats_path, metadata_path) if not p.exists()]
        if missing:
            logger.error(
                "Model artifacts missing %s in %s — run the training notebook first.",
                missing,
                self.model_dir,
            )
            return

        try:
            self.pipeline = joblib.load(pipeline_path)
            self.stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load model artifacts from %s", self.model_dir)
            return

        self.loaded = True
        self._validate_schema()

    def _validate_schema(self) -> None:
        expected = (self.metadata or {}).get("feature_columns", [])
        dummy_df = pd.DataFrame([{col: _DUMMY_ROW[col] for col in RAW_INPUT_COLUMNS}])

        try:
            produced = list(build_features(dummy_df, self.stats).columns)
        except Exception:
            logger.exception("build_features() raised while validating the model schema at startup")
            self.schema_valid = False
            return

        self.schema_valid = produced == expected
        if not self.schema_valid:
            logger.error(
                "Model schema mismatch — pipeline expects %s, build_features() currently produces %s. "
                "The notebook and this service have drifted apart; retrain or fix student_risk/features.py.",
                expected,
                produced,
            )

    def predict(self, request: PerformancePredictionRequest) -> PerformancePredictionResult:
        if not self.loaded:
            raise PerformanceModelError("Model artifacts are not loaded — check /health.")
        if not self.schema_valid:
            raise PerformanceModelError("Model feature schema is invalid — check /health and server logs.")

        row = request.model_dump()
        df = pd.DataFrame([row])
        features = build_features(df, self.stats)

        proba = self.pipeline.predict_proba(features)[0]
        classes = list(self.pipeline.classes_)
        pass_index = classes.index(1) if 1 in classes else int(proba.argmax())
        pass_probability = float(proba[pass_index])

        prediction = "Pass" if pass_probability >= 0.5 else "Fail"
        return PerformancePredictionResult(
            prediction=prediction,
            probability=pass_probability,
            risk_level=_risk_level(pass_probability),
        )


performance_model_service = PerformanceModelService(settings.model_dir)
