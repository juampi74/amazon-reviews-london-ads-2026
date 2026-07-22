from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import inference
from app.core.config import get_settings
from app.core.errors import RepositoryError
from app.routers import analyses, store, system, datasets


settings = get_settings()
app = FastAPI(
    title="Launchly API",
    version=inference.MODEL_VERSION,
    description="Authenticated decision-support API for Launchly.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["Cache-Control"] = "private, no-store"
    return response


@app.exception_handler(RepositoryError)
async def repository_error_handler(_: Request, exc: RepositoryError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


app.include_router(system.router)
app.include_router(analyses.router)
app.include_router(store.router)
app.include_router(datasets.router)