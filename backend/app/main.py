"""FastAPI application setup for the 3D box checker server."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.errors import ApiError
from app.routes import router as api_router
from app.schemas import ErrorResponse
from app.store import (
    DEFAULT_STORAGE_SETTINGS,
    StorageSettings,
    create_storage_directories,
    initialize_database,
)
from app.web import router as web_router


__all__ = ["app", "create_app"]


LOGGER = logging.getLogger(__name__)
APP_TITLE = "3D Box Checker API"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = (
    "Local-network API for analyzing whether a video shows a physical "
    "3D box or a flat representation."
)


def _error_response(status_code: int, detail: str) -> JSONResponse:
    body = ErrorResponse(detail=detail)
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
    )


def create_app(
    storage_settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
) -> FastAPI:
    """Build an independently configured FastAPI application.

    Supplying storage settings lets tests and local tools use an isolated data
    directory without changing the production module-level application.
    """
    if not isinstance(storage_settings, StorageSettings):
        raise TypeError("storage_settings must be a StorageSettings")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        # Pass the root path so startup remains stable even in development
        # tooling that reloads the storage module and recreates its class.
        create_storage_directories(storage_settings.data_directory)
        initialize_database(storage_settings)
        yield

    application = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
        lifespan=lifespan,
    )
    application.state.storage_settings = storage_settings
    application.include_router(api_router)
    application.include_router(web_router)

    @application.exception_handler(ApiError)
    async def handle_api_error(request: Request, error: ApiError) -> JSONResponse:
        del request
        return _error_response(error.status_code, error.detail)

    @application.exception_handler(RequestValidationError)
    async def handle_request_validation(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        del request, error
        return _error_response(422, "Request validation failed.")

    @application.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        request: Request,
        error: StarletteHTTPException,
    ) -> JSONResponse:
        del request
        detail = error.detail if isinstance(error.detail, str) else "Request failed."
        return _error_response(error.status_code, detail)

    @application.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        LOGGER.exception(
            "Unhandled API error for %s %s",
            request.method,
            request.url.path,
            exc_info=error,
        )
        return _error_response(500, "Internal server error.")

    return application


app = create_app()
