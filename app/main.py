import logging
from fastapi import FastAPI
import os
from dotenv import load_dotenv

load_dotenv()

from app.routers import routers
from app.data.loader import init_categories

CATEGORIES_DATA_FILE = os.getenv("CATEGORIES_DATA_FILE", "app/data/categories.csv")

# handler_ = [logging.FileHandler(LOG_FILE), logging.StreamHandler()]
handler_ = [logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s',
    handlers=handler_,
)


def create_app() -> FastAPI:
    init_categories(CATEGORIES_DATA_FILE)

    result = FastAPI(title="CityServiceAI Orchestrator")

    @result.get("/health")
    def health_check():
        """Endpoint для перевірки стану Додатку. Повертає 200 OK."""
        return {"status": "ok", "service": "cityserviceai-orchestrator"}

    result.include_router(routers.router)
    return result


app = create_app()
