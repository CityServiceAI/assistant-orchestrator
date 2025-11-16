from fastapi import APIRouter
from app.schemas.agents import NormalizerRequest, NormalizerResponse
from app.agents.normalizer import NormalizerAgent

router = APIRouter(prefix="/agents", tags=["agents"])

normalizer = NormalizerAgent()


@router.post("/normalize", response_model=NormalizerResponse)
async def normalize_text(body: NormalizerRequest):
    return await normalizer.run(body)
