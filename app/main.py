import sys
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# handler_ = [logging.FileHandler(LOG_FILE), logging.StreamHandler()]
handler_ = [logging.StreamHandler(sys.stdout)]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s',
    handlers=handler_,
)

from fastapi import FastAPI

from app.routers import routers
from app.data.loader import init_categories

CATEGORIES_DATA_FILE = os.getenv("CATEGORIES_DATA_FILE", "app/data/categories.csv")

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
