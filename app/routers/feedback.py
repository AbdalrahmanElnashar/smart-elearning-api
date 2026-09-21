from fastapi import APIRouter, Request

from app.config import settings
from app.limiter import limiter
from app.schemas import FeedbackAnalysisRequest, FeedbackAnalysisResponse
from app.services.gemini_feedback import feedback_service

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("/analyze", response_model=FeedbackAnalysisResponse)
@limiter.limit(settings.feedback_rate_limit)
def analyze_feedback(request: Request, payload: FeedbackAnalysisRequest) -> FeedbackAnalysisResponse:
    # feedback_service.analyze() never raises — Gemini timeouts, malformed
    # responses, or a missing API key all resolve to a graceful fallback,
    # so this endpoint always returns 200 unless something outside Gemini
    # itself goes wrong.
    result = feedback_service.analyze(payload.text)
    return FeedbackAnalysisResponse(
        sentiment=result.sentiment,
        issues=result.issues,
        summary=result.summary,
    )
