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
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=handler_,
)


def create_app() -> FastAPI:
    init_categories(CATEGORIES_DATA_FILE)

    result = FastAPI(title="CityServiceAI Orchestrator")
    result.include_router(routers.router)
    return result


app = create_app()
