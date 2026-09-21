from fastapi import APIRouter, HTTPException

from app.schemas import PerformancePredictionRequest, PerformancePredictionResponse
from app.services.performance_model import PerformanceModelError, performance_model_service

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/performance", response_model=PerformancePredictionResponse)
def predict_performance(request: PerformancePredictionRequest) -> PerformancePredictionResponse:
    try:
        result = performance_model_service.predict(request)
    except PerformanceModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PerformancePredictionResponse(
        prediction=result.prediction,
        probability=result.probability,
        risk_level=result.risk_level,
    )
