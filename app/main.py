from fastapi import FastAPI
from app.routers import agents_router
from app.routers import debug_router


def create_app() -> FastAPI:
    app = FastAPI(title="CityServiceAI")
    app.include_router(agents_router.router)
    app.include_router(debug_router.router)
    return app


app = create_app()
