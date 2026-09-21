import logging

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.limiter import limiter
from app.routers import feedback, predict
from app.schemas import HealthResponse
from app.services.gemini_feedback import feedback_service
from app.services.performance_model import performance_model_service

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Smart E-Learning Assistant API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(predict.router)
app.include_router(feedback.router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_ready = performance_model_service.loaded and performance_model_service.schema_valid
    return HealthResponse(
        status="ok" if model_ready else "degraded",
        model_loaded=performance_model_service.loaded,
        model_schema_valid=performance_model_service.schema_valid,
        gemini_configured=feedback_service.configured,
    )
