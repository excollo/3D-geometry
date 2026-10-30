"""Persistence boundary for saved box-check analyses.

This module owns local files and, in later Part 2 stages, will own the SQLite
index.  The standalone :mod:`boxcheck` package must not import this module;
application code calls the checker and then uses this module to persist its
outputs.

Directory initialization, video copying, the checker/debug handoff, result
JSON, SQLite schema initialization, and database row insertion are implemented
along with the complete analyze-and-save workflow, read operations, and
original-preserving re-runs.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfileobj, rmtree
from tempfile import NamedTemporaryFile
from typing import Any, Callable, TypeAlias
from uuid import UUID, uuid4

from boxcheck.pipeline import analyze_video
from boxcheck.types import (
    AnalysisMetrics,
    AnalysisResult,
    GroundTruthLabel,
    PairMetrics,
    Verdict,
)


__all__ = [
    "DEFAULT_DATA_DIRECTORY",
    "DEFAULT_STORAGE_SETTINGS",
    "DATABASE_SCHEMA_VERSION",
    "STORAGE_SUBDIRECTORIES",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "AnalysisSummary",
    "ResultPolicy",
    "StoredAnalysis",
    "StorageSettings",
    "analyze_saved_video",
    "build_stored_analysis",
    "create_storage_directories",
    "create_analysis",
    "generate_analysis_id",
    "get_analysis",
    "get_debug_frame",
    "get_result",
    "index_analysis",
    "initialize_database",
    "list_analyses",
    "prepare_debug_directory",
    "save_result",
    "save_video",
    "rerun_analysis",
    "validate_analysis_id",
]


DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
STORAGE_SUBDIRECTORIES = ("uploads", "results", "debug")
SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov"})
STORED_VERDICTS = frozenset({"3d", "flat", "insufficient", "review"})
DATABASE_SCHEMA_VERSION = 1
_COPY_BUFFER_BYTES = 1024 * 1024
_SQLITE_TIMEOUT_SECONDS = 5.0
_ANALYSES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY NOT NULL CHECK (length(id) = 36),
    created_at TEXT NOT NULL CHECK (length(created_at) > 0),
    label TEXT CHECK (
        label IS NULL OR label IN (
            'real',
            'flat_printout',
            'folded_printout',
            'screen',
            'bad_recording',
            'other'
        )
    ),
    phone_model TEXT CHECK (phone_model IS NULL OR length(phone_model) > 0),
    attempt INTEGER NOT NULL CHECK (attempt >= 1),
    verdict TEXT NOT NULL CHECK (
        verdict IN ('3d', 'flat', 'insufficient', 'review')
    ),
    score REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    video_path TEXT NOT NULL CHECK (length(video_path) > 0),
    result_path TEXT NOT NULL CHECK (length(result_path) > 0),
    debug_directory TEXT NOT NULL CHECK (length(debug_directory) > 0)
)
"""
_ANALYSES_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_analyses_label ON analyses(label)",
    "CREATE INDEX IF NOT EXISTS idx_analyses_phone_model ON analyses(phone_model)",
    "CREATE INDEX IF NOT EXISTS idx_analyses_verdict ON analyses(verdict)",
)
_INSERT_ANALYSIS_SQL = """
INSERT INTO analyses (
    id,
    created_at,
    label,
    phone_model,
    attempt,
    verdict,
    score,
    confidence,
    video_path,
    result_path,
    debug_directory
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
_SELECT_ANALYSIS_SQL = """
SELECT id, created_at, label, phone_model, attempt, verdict, score, confidence,
       video_path, result_path, debug_directory
FROM analyses
"""


@dataclass(frozen=True, slots=True, init=False)
class StorageSettings:
    """All filesystem locations used by the local persistence layer."""

    data_directory: Path
    uploads_directory: Path = field(init=False)
    results_directory: Path = field(init=False)
    debug_directory: Path = field(init=False)
    database_path: Path = field(init=False)

    def __init__(self, data_directory: str | Path) -> None:
        data_directory = Path(data_directory)
        object.__setattr__(self, "data_directory", data_directory)
        object.__setattr__(self, "uploads_directory", data_directory / "uploads")
        object.__setattr__(self, "results_directory", data_directory / "results")
        object.__setattr__(self, "debug_directory", data_directory / "debug")
        object.__setattr__(self, "database_path", data_directory / "analyses.sqlite3")

    @property
    def directories(self) -> dict[str, Path]:
        """Return the directories that must exist before files are stored."""
        return {
            "uploads": self.uploads_directory,
            "results": self.results_directory,
            "debug": self.debug_directory,
        }


@dataclass(frozen=True, slots=True)
class AnalysisSummary:
    """Lightweight SQLite-backed information used by analysis history."""

    id: str
    created_at: datetime
    label: GroundTruthLabel | None
    phone_model: str | None
    attempt: int
    verdict: str
    score: float
    confidence: float
    video_path: Path
    result_path: Path
    debug_directory: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "label": self.label.value if self.label is not None else None,
            "phone_model": self.phone_model,
            "attempt": self.attempt,
            "verdict": self.verdict,
            "score": self.score,
            "confidence": self.confidence,
            "video_path": str(self.video_path),
            "result_path": str(self.result_path),
            "debug_directory": str(self.debug_directory),
        }


@dataclass(frozen=True, slots=True)
class StoredAnalysis:
    """A complete checker result together with its persistence metadata."""

    id: str
    created_at: datetime
    attempt: int
    video_path: Path
    result_path: Path
    debug_directory: Path
    result: AnalysisResult
    verdict: str | None = None

    def __post_init__(self) -> None:
        validate_analysis_id(self.id)
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int):
            raise TypeError("attempt must be an integer")
        if self.attempt < 1:
            raise ValueError("attempt must be at least 1")
        if not isinstance(self.result, AnalysisResult):
            raise TypeError("result must be an AnalysisResult")
        if self.verdict is None:
            object.__setattr__(self, "verdict", self.result.verdict.value)
        elif self.verdict not in STORED_VERDICTS:
            raise ValueError(f"unsupported stored verdict: {self.verdict}")

        for field_name in ("video_path", "result_path", "debug_directory"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))

        if (
            self.video_path.stem != self.id
            or self.video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS
        ):
            raise ValueError("video_path must use the analysis ID and a supported extension")
        if self.result_path.name != f"{self.id}.json":
            raise ValueError("result_path must use the analysis ID with a .json extension")
        if self.debug_directory.name != self.id:
            raise ValueError("debug_directory must use the analysis ID")

    def to_dict(self) -> dict[str, Any]:
        """Return the full record in a JSON-compatible shape."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "attempt": self.attempt,
            **self.result.to_dict(),
            "verdict": self.verdict,
            "video_path": str(self.video_path),
            "result_path": str(self.result_path),
            "debug_directory": str(self.debug_directory),
        }


DEFAULT_STORAGE_SETTINGS = StorageSettings(DEFAULT_DATA_DIRECTORY)


ResultPolicy: TypeAlias = Callable[[AnalysisResult, int], str]
"""Choose the stored verdict from a checker result and its attempt."""


def generate_analysis_id() -> str:
    """Return a new canonical UUID v4 for one stored analysis."""
    return str(uuid4())


def initialize_database(
    settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
) -> Path:
    """Create or verify the current SQLite schema and return its path."""
    settings.data_directory.mkdir(parents=True, exist_ok=True)

    with closing(
        sqlite3.connect(
            settings.database_path,
            timeout=_SQLITE_TIMEOUT_SECONDS,
        )
    ) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        current_version = int(
            connection.execute("PRAGMA user_version").fetchone()[0]
        )
        if current_version > DATABASE_SCHEMA_VERSION:
            raise RuntimeError(
                "Database schema version "
                f"{current_version} is newer than supported version "
                f"{DATABASE_SCHEMA_VERSION}"
            )

        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(_ANALYSES_TABLE_SQL)
            for index_sql in _ANALYSES_INDEX_SQL:
                connection.execute(index_sql)
            if current_version < DATABASE_SCHEMA_VERSION:
                connection.execute(
                    f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}"
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    return settings.database_path


def validate_analysis_id(analysis_id: str) -> str:
    """Validate and return a canonical UUID v4 analysis ID.

    IDs are validated before later storage steps use them in filenames or
    database lookups. Non-canonical forms, other UUID versions, and path-like
    values are rejected.
    """
    if not isinstance(analysis_id, str):
        raise ValueError("analysis_id must be a string UUID")

    try:
        parsed_id = UUID(analysis_id)
    except (ValueError, AttributeError) as error:
        raise ValueError("analysis_id must be a valid UUID v4") from error

    if parsed_id.version != 4 or str(parsed_id) != analysis_id:
        raise ValueError("analysis_id must be a canonical lowercase UUID v4")

    return analysis_id


def create_analysis(
    source_video_path: str | Path,
    label: GroundTruthLabel | str | None = None,
    phone_model: str | None = None,
    attempt: int = 1,
    settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
    result_policy: ResultPolicy | None = None,
) -> StoredAnalysis:
    """Analyze one video and persist all of its Part 2 artifacts.

    When supplied, ``result_policy`` runs once after the standalone checker
    returns and before the result JSON or SQLite row is created. This keeps
    server-only verdict rules outside :mod:`boxcheck` while ensuring every
    persisted representation receives the same result.
    """
    if not isinstance(settings, StorageSettings):
        raise TypeError("settings must be a StorageSettings")
    if result_policy is not None and not callable(result_policy):
        raise TypeError("result_policy must be callable or None")
    source, _ = _validate_video_source(source_video_path)
    if isinstance(label, str):
        typed_label = GroundTruthLabel(label)
    elif label is None or isinstance(label, GroundTruthLabel):
        typed_label = label
    else:
        raise TypeError("label must be a GroundTruthLabel, string, or None")
    if phone_model is not None:
        if not isinstance(phone_model, str):
            raise TypeError("phone_model must be a string or None")
        if not phone_model.strip():
            raise ValueError("phone_model cannot be blank")
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise TypeError("attempt must be an integer")
    if attempt < 1:
        raise ValueError("attempt must be at least 1")

    analysis_id = generate_analysis_id()
    expected_result_path = settings.results_directory / f"{analysis_id}.json"
    expected_debug_directory = settings.debug_directory / analysis_id
    result_existed_before = expected_result_path.exists()
    debug_existed_before = expected_debug_directory.exists()
    saved_video: Path | None = None

    try:
        saved_video = save_video(source, analysis_id, settings)
        result = analyze_saved_video(
            saved_video,
            analysis_id,
            label=typed_label,
            phone_model=phone_model,
            settings=settings,
        )
        stored_verdict = result.verdict.value
        if result_policy is not None:
            stored_verdict = result_policy(result, attempt)
            if stored_verdict not in STORED_VERDICTS:
                raise ValueError(
                    "result_policy must return one of: "
                    + ", ".join(sorted(STORED_VERDICTS))
                )
        stored_analysis = build_stored_analysis(
            analysis_id,
            saved_video,
            result,
            attempt=attempt,
            settings=settings,
            verdict=stored_verdict,
        )
        save_result(stored_analysis, settings)
        index_analysis(stored_analysis, settings)
        return stored_analysis
    except Exception:
        _cleanup_failed_analysis(
            saved_video=saved_video,
            result_path=(None if result_existed_before else expected_result_path),
            debug_directory=(
                None if debug_existed_before else expected_debug_directory
            ),
        )
        raise


def rerun_analysis(
    analysis_id: str,
    settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
    result_policy: ResultPolicy | None = None,
) -> StoredAnalysis | None:
    """Run a saved video's original input again with current checker settings."""
    if result_policy is not None and not callable(result_policy):
        raise TypeError("result_policy must be callable or None")
    original = get_analysis(analysis_id, settings)
    if original is None:
        return None
    return create_analysis(
        original.video_path,
        label=original.result.label,
        phone_model=original.result.phone_model,
        attempt=original.attempt,
        settings=settings,
        result_policy=result_policy,
    )


def _cleanup_failed_analysis(
    *,
    saved_video: Path | None,
    result_path: Path | None,
    debug_directory: Path | None,
) -> None:
    """Best-effort cleanup limited to artifacts owned by a failed new analysis."""
    if result_path is not None:
        try:
            result_path.unlink(missing_ok=True)
        except OSError:
            pass
    if debug_directory is not None and debug_directory.is_dir():
        try:
            rmtree(debug_directory)
        except OSError:
            pass
    if saved_video is not None:
        try:
            saved_video.unlink(missing_ok=True)
        except OSError:
            pass


def build_stored_analysis(
    analysis_id: str,
    saved_video_path: str | Path,
    result: AnalysisResult,
    attempt: int = 1,
    settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
    created_at: datetime | None = None,
    verdict: str | None = None,
) -> StoredAnalysis:
    """Build a consistent storage record for a completed checker result."""
    validated_id = validate_analysis_id(analysis_id)
    saved_video = Path(saved_video_path)
    if not saved_video.is_file():
        raise FileNotFoundError(f"Saved video is not a file: {saved_video}")
    if saved_video.resolve().parent != settings.uploads_directory.resolve():
        raise ValueError("saved_video_path must be in the configured uploads directory")

    debug_directory = settings.debug_directory / validated_id
    if not debug_directory.is_dir():
        raise FileNotFoundError(
            f"Analysis debug directory does not exist: {debug_directory}"
        )

    return StoredAnalysis(
        id=validated_id,
        created_at=created_at or datetime.now(timezone.utc),
        attempt=attempt,
        video_path=saved_video,
        result_path=settings.results_directory / f"{validated_id}.json",
        debug_directory=debug_directory,
        result=result,
        verdict=verdict,
    )


def save_result(
    stored_analysis: StoredAnalysis,
    settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
) -> Path:
    """Atomically save a complete analysis as portable JSON without overwrite."""
    if not isinstance(stored_analysis, StoredAnalysis):
        raise TypeError("stored_analysis must be a StoredAnalysis")

    _validate_stored_analysis_paths(stored_analysis, settings)

    data = stored_analysis.to_dict()
    data["video_path"] = _storage_relative_path(stored_analysis.video_path, settings)
    data["result_path"] = _storage_relative_path(stored_analysis.result_path, settings)
    data["debug_directory"] = _storage_relative_path(
        stored_analysis.debug_directory,
        settings,
    )
    data["debug_frames"] = [
        _storage_relative_path(frame_path, settings)
        for frame_path in stored_analysis.result.debug_frames
    ]

    serialized = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"

    settings.results_directory.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=settings.results_directory,
            prefix=f".{stored_analysis.id}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(serialized)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.link(temporary_path, stored_analysis.result_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return stored_analysis.result_path


def index_analysis(
    stored_analysis: StoredAnalysis,
    settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
) -> str:
    """Insert one completed analysis summary into the SQLite index."""
    if not isinstance(stored_analysis, StoredAnalysis):
        raise TypeError("stored_analysis must be a StoredAnalysis")

    _validate_stored_analysis_paths(stored_analysis, settings)
    if not stored_analysis.video_path.is_file():
        raise FileNotFoundError(f"Stored video does not exist: {stored_analysis.video_path}")
    if not stored_analysis.result_path.is_file():
        raise FileNotFoundError(f"Stored result does not exist: {stored_analysis.result_path}")
    if not stored_analysis.debug_directory.is_dir():
        raise FileNotFoundError(
            f"Stored debug directory does not exist: {stored_analysis.debug_directory}"
        )

    initialize_database(settings)
    label = (
        stored_analysis.result.label.value
        if stored_analysis.result.label is not None
        else None
    )
    values = (
        stored_analysis.id,
        stored_analysis.created_at.isoformat(),
        label,
        stored_analysis.result.phone_model,
        stored_analysis.attempt,
        stored_analysis.verdict,
        stored_analysis.result.score,
        stored_analysis.result.confidence,
        _storage_relative_path(stored_analysis.video_path, settings),
        _storage_relative_path(stored_analysis.result_path, settings),
        _storage_relative_path(stored_analysis.debug_directory, settings),
    )

    with closing(
        sqlite3.connect(
            settings.database_path,
            timeout=_SQLITE_TIMEOUT_SECONDS,
        )
    ) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(_INSERT_ANALYSIS_SQL, values)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    return stored_analysis.id


def list_analyses(
    limit: int = 100,
    offset: int = 0,
    settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
) -> list[AnalysisSummary]:
    """Return analysis summaries newest first with bounded pagination."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("offset must be an integer")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    initialize_database(settings)
    with closing(sqlite3.connect(settings.database_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            _SELECT_ANALYSIS_SQL
            + " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [_summary_from_row(row, settings) for row in rows]


def get_result(
    analysis_id: str,
    settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
) -> dict[str, Any] | None:
    """Load the complete portable JSON result for one indexed analysis."""
    validated_id = validate_analysis_id(analysis_id)
    row = _fetch_analysis_row(validated_id, settings)
    if row is None:
        return None
    return _load_and_validate_result_data(row, settings)


def get_analysis(
    analysis_id: str,
    settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
) -> StoredAnalysis | None:
    """Reconstruct a complete StoredAnalysis from SQLite and result JSON."""
    validated_id = validate_analysis_id(analysis_id)
    row = _fetch_analysis_row(validated_id, settings)
    if row is None:
        return None
    data = _load_and_validate_result_data(row, settings)

    try:
        metrics = AnalysisMetrics(**data["metrics"])
        pairs = [PairMetrics(**pair) for pair in data["pairs"]]
        label = GroundTruthLabel(data["label"]) if data["label"] is not None else None
        debug_frames = [
            _resolve_storage_path(path, settings)
            for path in data["debug_frames"]
        ]
        checker_verdict = (
            Verdict.INSUFFICIENT
            if data["verdict"] == "review"
            else Verdict(data["verdict"])
        )
        result = AnalysisResult(
            verdict=checker_verdict,
            score=data["score"],
            confidence=data["confidence"],
            reason=data["reason"],
            metrics=metrics,
            pairs=pairs,
            processing_ms=data["processing_ms"],
            label=label,
            phone_model=data["phone_model"],
            debug_frames=debug_frames,
        )
        return StoredAnalysis(
            id=validated_id,
            created_at=datetime.fromisoformat(data["created_at"]),
            attempt=data["attempt"],
            video_path=_resolve_storage_path(data["video_path"], settings),
            result_path=_resolve_storage_path(data["result_path"], settings),
            debug_directory=_resolve_storage_path(data["debug_directory"], settings),
            result=result,
            verdict=data["verdict"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Stored analysis {validated_id} is invalid") from error


def get_debug_frame(
    analysis_id: str,
    frame_number: int,
    settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
) -> Path | None:
    """Return one verified debug image path by its zero-based list index."""
    if isinstance(frame_number, bool) or not isinstance(frame_number, int):
        raise TypeError("frame_number must be an integer")
    if frame_number < 0:
        raise ValueError("frame_number must be non-negative")

    data = get_result(analysis_id, settings)
    if data is None:
        return None
    debug_frames = data.get("debug_frames")
    if not isinstance(debug_frames, list):
        raise ValueError("Stored debug_frames must be a list")
    if frame_number >= len(debug_frames):
        return None

    frame_path = _resolve_storage_path(debug_frames[frame_number], settings)
    expected_directory = (settings.debug_directory / analysis_id).resolve()
    if frame_path.resolve().parent != expected_directory:
        raise ValueError("Stored debug frame is outside its analysis directory")
    if not frame_path.is_file():
        raise FileNotFoundError(f"Stored debug frame does not exist: {frame_path}")
    return frame_path


def _fetch_analysis_row(
    analysis_id: str,
    settings: StorageSettings,
) -> sqlite3.Row | None:
    initialize_database(settings)
    with closing(sqlite3.connect(settings.database_path)) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            _SELECT_ANALYSIS_SQL + " WHERE id = ?",
            (analysis_id,),
        ).fetchone()


def _summary_from_row(
    row: sqlite3.Row,
    settings: StorageSettings,
) -> AnalysisSummary:
    try:
        created_at = datetime.fromisoformat(row["created_at"])
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        return AnalysisSummary(
            id=validate_analysis_id(row["id"]),
            created_at=created_at,
            label=(GroundTruthLabel(row["label"]) if row["label"] is not None else None),
            phone_model=row["phone_model"],
            attempt=row["attempt"],
            verdict=row["verdict"],
            score=row["score"],
            confidence=row["confidence"],
            video_path=_resolve_storage_path(row["video_path"], settings),
            result_path=_resolve_storage_path(row["result_path"], settings),
            debug_directory=_resolve_storage_path(row["debug_directory"], settings),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Stored analysis summary is invalid") from error


def _load_and_validate_result_data(
    row: sqlite3.Row,
    settings: StorageSettings,
) -> dict[str, Any]:
    result_path = _resolve_storage_path(row["result_path"], settings)
    if not result_path.is_file():
        raise FileNotFoundError(f"Stored result does not exist: {result_path}")
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Stored result JSON is invalid: {result_path}") from error
    if not isinstance(data, dict):
        raise ValueError("Stored result JSON must contain an object")

    comparisons = {
        "id": row["id"],
        "created_at": row["created_at"],
        "attempt": row["attempt"],
        "verdict": row["verdict"],
        "score": row["score"],
        "confidence": row["confidence"],
        "label": row["label"],
        "phone_model": row["phone_model"],
        "video_path": row["video_path"],
        "result_path": row["result_path"],
        "debug_directory": row["debug_directory"],
    }
    for key, expected in comparisons.items():
        if data.get(key) != expected:
            raise ValueError(f"SQLite and result JSON disagree on {key}")
    return data


def _resolve_storage_path(path: str, settings: StorageSettings) -> Path:
    """Resolve a stored relative path without allowing escape from data/."""
    if not isinstance(path, str) or not path:
        raise ValueError("Stored path must be a non-empty string")
    stored_path = Path(path)
    if stored_path.is_absolute():
        raise ValueError("Stored path must be relative to the data directory")
    resolved = (settings.data_directory / stored_path).resolve()
    try:
        resolved.relative_to(settings.data_directory.resolve())
    except ValueError as error:
        raise ValueError("Stored path escapes the data directory") from error
    return resolved


def _validate_stored_analysis_paths(
    stored_analysis: StoredAnalysis,
    settings: StorageSettings,
) -> None:
    """Ensure a record belongs entirely to one storage configuration."""
    expected_result_path = settings.results_directory / f"{stored_analysis.id}.json"
    if stored_analysis.result_path.resolve() != expected_result_path.resolve():
        raise ValueError("result_path must be inside the configured results directory")
    if stored_analysis.video_path.resolve().parent != settings.uploads_directory.resolve():
        raise ValueError("video_path must be inside the configured uploads directory")
    expected_debug_directory = settings.debug_directory / stored_analysis.id
    if stored_analysis.debug_directory.resolve() != expected_debug_directory.resolve():
        raise ValueError("debug_directory must match the configured analysis directory")


def _storage_relative_path(path: str | Path, settings: StorageSettings) -> str:
    """Return a portable POSIX path below the configured data directory."""
    candidate = Path(path).resolve()
    try:
        relative = candidate.relative_to(settings.data_directory.resolve())
    except ValueError as error:
        raise ValueError(f"Storage path is outside the data directory: {path}") from error
    return relative.as_posix()


def save_video(
    source_path: str | Path,
    analysis_id: str,
    settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
) -> Path:
    """Copy an original video into storage without replacing existing data."""
    source, extension = _validate_video_source(source_path)
    validated_id = validate_analysis_id(analysis_id)

    directories = create_storage_directories(settings)
    destination = directories["uploads"] / f"{validated_id}{extension}"
    destination_created = False

    try:
        with source.open("rb") as source_file:
            with destination.open("xb") as destination_file:
                destination_created = True
                copyfileobj(
                    source_file,
                    destination_file,
                    length=_COPY_BUFFER_BYTES,
                )
    except Exception:
        if destination_created:
            destination.unlink(missing_ok=True)
        raise

    return destination


def _validate_video_source(source_path: str | Path) -> tuple[Path, str]:
    """Validate a source video without changing storage."""
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Video source is not a file: {source}")
    extension = source.suffix.lower()
    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video extension {source.suffix!r}; use {supported}")
    return source, extension


def prepare_debug_directory(
    analysis_id: str,
    settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
) -> Path:
    """Create a new, empty debug directory for one analysis."""
    validated_id = validate_analysis_id(analysis_id)
    settings.debug_directory.mkdir(parents=True, exist_ok=True)
    debug_directory = settings.debug_directory / validated_id
    debug_directory.mkdir(exist_ok=False)
    return debug_directory


def analyze_saved_video(
    saved_video_path: str | Path,
    analysis_id: str,
    label: GroundTruthLabel | str | None = None,
    phone_model: str | None = None,
    settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
) -> AnalysisResult:
    """Run Part 1 on a stored video and direct debug images into storage."""
    validated_id = validate_analysis_id(analysis_id)
    saved_video = Path(saved_video_path)
    if not saved_video.is_file():
        raise FileNotFoundError(f"Saved video is not a file: {saved_video}")

    expected_parent = settings.uploads_directory.resolve()
    if (
        saved_video.resolve().parent != expected_parent
        or saved_video.stem != validated_id
        or saved_video.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS
    ):
        raise ValueError(
            "saved_video_path must be the matching video in the configured uploads directory"
        )

    debug_directory = prepare_debug_directory(validated_id, settings)
    return analyze_video(
        saved_video,
        label=label,
        phone_model=phone_model,
        debug_output_directory=debug_directory,
    )


def create_storage_directories(
    settings: StorageSettings | str | Path = DEFAULT_STORAGE_SETTINGS,
) -> dict[str, Path]:
    """Create and return the directories used by persistent storage.

    Existing directories are left untouched, so this function is safe to call
    every time the future API starts. A root path remains accepted as a
    convenience; it is converted to :class:`StorageSettings` internally.
    """
    storage_settings = (
        settings
        if isinstance(settings, StorageSettings)
        else StorageSettings(Path(settings))
    )
    directories = storage_settings.directories

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories
