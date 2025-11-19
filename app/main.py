import logging
from fastapi import FastAPI
from app.routers import routers
from app.data.loader import init_categories

# handler_ = [logging.FileHandler(LOG_FILE), logging.StreamHandler()]
handler_ = [logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=handler_,
)


def create_app() -> FastAPI:
    app = FastAPI(title="CityServiceAI Orchestrator")
    init_categories("app/data/categories.csv")
    app.include_router(routers.router)

    return app


app = create_app()
