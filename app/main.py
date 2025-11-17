from fastapi import FastAPI
from app.routers import agents_router, pipeline_router, routers


def create_app() -> FastAPI:
    app = FastAPI(title="CityServiceAI Orchestrator")

    app.include_router(agents_router.router)
    app.include_router(pipeline_router.router)
    app.include_router(routers.router)

    return app


app = create_app()
