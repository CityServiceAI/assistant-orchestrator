import logging

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

# handler_ = [logging.FileHandler(LOG_FILE), logging.StreamHandler()]
handler_ = [logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=handler_
)

from app.routers import agents_router, pipeline_router, routers


def create_app() -> FastAPI:
    app = FastAPI(title="CityServiceAI Orchestrator")

    app.include_router(agents_router.router)
    app.include_router(pipeline_router.router)
    app.include_router(routers.router)

    return app


app = create_app()
