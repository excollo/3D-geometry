"""HTTP routes for the versioned box-checker API."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, File, Form, Path as PathParameter, Query, Request, UploadFile
from pydantic import UUID4
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse

from app.errors import BadRequestError, NotFoundError
from app.policies import apply_review_policy
from app.schemas import (
    AnalysisListResponse,
    AnalysisResponse,
    ErrorResponse,
    analysis_list_response,
    analysis_response_from_result,
    analysis_response_from_stored,
)
from app.store import (
    SUPPORTED_VIDEO_EXTENSIONS,
    StorageSettings,
    create_analysis,
    get_debug_frame,
    get_result,
    list_analyses,
    rerun_analysis,
)
from boxcheck.types import GroundTruthLabel
from boxcheck.video import VideoMetadataError, VideoOpenError, VideoReadError


__all__ = ["router"]


router = APIRouter(prefix="/api/v1")
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_VIDEO_INPUT_ERRORS = (VideoOpenError, VideoMetadataError, VideoReadError)


def _upload_extension(video: UploadFile) -> str:
    filename = video.filename
    if not filename or not filename.strip():
        raise BadRequestError("A video filename is required.")
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise BadRequestError(f"Unsupported video extension; use {supported}.")
    return extension


async def _save_temporary_upload(video: UploadFile, extension: str) -> Path:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix="boxcheck-upload-",
            suffix=extension,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            while chunk := await video.read(_UPLOAD_CHUNK_BYTES):
                temporary_file.write(chunk)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze_upload(
    request: Request,
    video: Annotated[UploadFile, File(description="MP4 or MOV video to analyze")],
    label: Annotated[GroundTruthLabel | None, Form()] = None,
    phone_model: Annotated[str | None, Form(min_length=1)] = None,
    attempt: Annotated[int, Form(ge=1)] = 1,
) -> AnalysisResponse:
    """Analyze and persist one uploaded video in the same request."""
    extension = _upload_extension(video)
    temporary_path: Path | None = None
    try:
        temporary_path = await _save_temporary_upload(video, extension)
        settings: StorageSettings = request.app.state.storage_settings
        operation = partial(
            create_analysis,
            temporary_path,
            label=label,
            phone_model=phone_model,
            attempt=attempt,
            settings=settings,
            result_policy=apply_review_policy,
        )
        try:
            stored_analysis = await run_in_threadpool(operation)
        except _VIDEO_INPUT_ERRORS as error:
            raise BadRequestError(
                "The video could not be read. Upload a valid MP4 or MOV file."
            ) from error
        return analysis_response_from_stored(stored_analysis)
    finally:
        await video.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@router.get(
    "/analyses",
    response_model=AnalysisListResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def analysis_history(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AnalysisListResponse:
    """Return one newest-first page of saved analysis summaries."""
    settings: StorageSettings = request.app.state.storage_settings
    summaries = await run_in_threadpool(
        partial(list_analyses, limit=limit, offset=offset, settings=settings)
    )
    return analysis_list_response(summaries, limit=limit, offset=offset)


@router.get(
    "/analyses/{analysis_id}",
    response_model=AnalysisResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analysis_detail(
    request: Request,
    analysis_id: Annotated[UUID4, PathParameter()],
) -> AnalysisResponse:
    """Return the complete public result for one saved analysis."""
    canonical_id = str(analysis_id)
    settings: StorageSettings = request.app.state.storage_settings
    result = await run_in_threadpool(
        partial(get_result, canonical_id, settings=settings)
    )
    if result is None:
        raise NotFoundError("Analysis not found.")
    return analysis_response_from_result(result)


@router.get(
    "/analyses/{analysis_id}/debug/{frame_number}.jpg",
    response_class=FileResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analysis_debug_frame(
    request: Request,
    analysis_id: Annotated[UUID4, PathParameter()],
    frame_number: Annotated[int, PathParameter(ge=0)],
) -> FileResponse:
    """Return one verified checker-generated debug JPEG by zero-based index."""
    canonical_id = str(analysis_id)
    settings: StorageSettings = request.app.state.storage_settings
    frame_path = await run_in_threadpool(
        partial(
            get_debug_frame,
            canonical_id,
            frame_number,
            settings=settings,
        )
    )
    if frame_path is None:
        raise NotFoundError("Analysis or debug frame not found.")
    return FileResponse(frame_path, media_type="image/jpeg")


@router.post(
    "/analyses/{analysis_id}/rerun",
    response_model=AnalysisResponse,
    status_code=201,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def rerun_saved_analysis(
    request: Request,
    analysis_id: Annotated[UUID4, PathParameter()],
) -> AnalysisResponse:
    """Create a new analysis from a saved video's original bytes."""
    canonical_id = str(analysis_id)
    settings: StorageSettings = request.app.state.storage_settings
    rerun = await run_in_threadpool(
        partial(
            rerun_analysis,
            canonical_id,
            settings=settings,
            result_policy=apply_review_policy,
        )
    )
    if rerun is None:
        raise NotFoundError("Analysis not found.")
    return analysis_response_from_stored(rerun)
