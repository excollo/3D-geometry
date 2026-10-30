"""Tests for local analysis storage setup."""

import ast
import importlib
import json
import sqlite3
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid1

import app.store as store_module
import pytest
from app.policies import apply_review_policy
from app.store import STORAGE_SUBDIRECTORIES, create_storage_directories
from boxcheck.types import (
    AnalysisMetrics,
    AnalysisResult,
    GroundTruthLabel,
    PairMetrics,
    Verdict,
)


def test_storage_module_has_an_explicit_public_interface() -> None:
    assert store_module.__all__ == [
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


def test_importing_storage_module_does_not_create_directories(
    monkeypatch,
) -> None:
    def fail_if_called(*args, **kwargs) -> None:
        raise AssertionError("Importing app.store must not create directories")

    monkeypatch.setattr(Path, "mkdir", fail_if_called)

    importlib.reload(store_module)


def test_checker_package_does_not_import_application_storage() -> None:
    checker_directory = Path(__file__).parents[1] / "boxcheck"

    for source_path in checker_directory.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )

        assert not any(
            module == "app" or module.startswith("app.")
            for module in imported_modules
        ), f"{source_path.name} must remain independent of application storage"


def test_storage_settings_derives_every_path_from_one_data_directory(
    tmp_path: Path,
) -> None:
    data_directory = tmp_path / "custom-data"

    settings = store_module.StorageSettings(data_directory)

    assert settings.data_directory == data_directory
    assert settings.uploads_directory == data_directory / "uploads"
    assert settings.results_directory == data_directory / "results"
    assert settings.debug_directory == data_directory / "debug"
    assert settings.database_path == data_directory / "analyses.sqlite3"
    assert settings.directories == {
        "uploads": data_directory / "uploads",
        "results": data_directory / "results",
        "debug": data_directory / "debug",
    }


def test_storage_settings_accepts_a_string_and_does_not_touch_disk(
    tmp_path: Path,
) -> None:
    data_directory = tmp_path / "not-created"

    settings = store_module.StorageSettings(str(data_directory))

    assert settings.data_directory == data_directory
    assert not data_directory.exists()


def test_storage_settings_cannot_be_changed_after_creation(tmp_path: Path) -> None:
    settings = store_module.StorageSettings(tmp_path / "data")

    with pytest.raises(FrozenInstanceError):
        settings.data_directory = tmp_path / "somewhere-else"


def test_default_storage_settings_uses_backend_data_directory() -> None:
    settings = store_module.DEFAULT_STORAGE_SETTINGS

    assert settings.data_directory == store_module.DEFAULT_DATA_DIRECTORY
    assert settings.database_path.name == "analyses.sqlite3"


def test_generate_analysis_id_returns_canonical_uuid4() -> None:
    analysis_id = store_module.generate_analysis_id()
    parsed_id = UUID(analysis_id)

    assert parsed_id.version == 4
    assert str(parsed_id) == analysis_id


def test_generate_analysis_id_returns_unique_values() -> None:
    analysis_ids = {
        store_module.generate_analysis_id()
        for _ in range(100)
    }

    assert len(analysis_ids) == 100


def test_validate_analysis_id_accepts_generated_id() -> None:
    analysis_id = store_module.generate_analysis_id()

    assert store_module.validate_analysis_id(analysis_id) == analysis_id


@pytest.mark.parametrize(
    "invalid_id",
    [
        "",
        "not-a-uuid",
        "../results",
        str(uuid1()),
        "76042442-239A-4677-96C2-93F95506F5A0",
        "{76042442-239a-4677-96c2-93f95506f5a0}",
        None,
    ],
)
def test_validate_analysis_id_rejects_invalid_or_unsafe_values(
    invalid_id: object,
) -> None:
    with pytest.raises(ValueError, match="analysis_id"):
        store_module.validate_analysis_id(invalid_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("extension", [".mp4", ".mov", ".MP4", ".MOV"])
def test_save_video_copies_original_into_configured_uploads(
    tmp_path: Path,
    extension: str,
) -> None:
    source = tmp_path / f"recording{extension}"
    source.write_bytes(b"original video bytes")
    settings = store_module.StorageSettings(tmp_path / "storage")
    analysis_id = store_module.generate_analysis_id()

    saved_path = store_module.save_video(source, analysis_id, settings)

    assert saved_path == settings.uploads_directory / f"{analysis_id}{extension.lower()}"
    assert saved_path.read_bytes() == b"original video bytes"
    assert source.read_bytes() == b"original video bytes"


def test_save_video_rejects_missing_or_non_file_source(tmp_path: Path) -> None:
    settings = store_module.StorageSettings(tmp_path / "storage")
    analysis_id = store_module.generate_analysis_id()

    with pytest.raises(FileNotFoundError, match="not a file"):
        store_module.save_video(tmp_path / "missing.mp4", analysis_id, settings)

    directory_with_video_suffix = tmp_path / "directory.mp4"
    directory_with_video_suffix.mkdir()
    with pytest.raises(FileNotFoundError, match="not a file"):
        store_module.save_video(directory_with_video_suffix, analysis_id, settings)


def test_save_video_rejects_unsupported_extension(tmp_path: Path) -> None:
    source = tmp_path / "recording.txt"
    source.write_bytes(b"not a supported video")
    settings = store_module.StorageSettings(tmp_path / "storage")

    with pytest.raises(ValueError, match="Unsupported video extension"):
        store_module.save_video(
            source,
            store_module.generate_analysis_id(),
            settings,
        )


def test_save_video_rejects_invalid_analysis_id(tmp_path: Path) -> None:
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"video")
    settings = store_module.StorageSettings(tmp_path / "storage")

    with pytest.raises(ValueError, match="analysis_id"):
        store_module.save_video(source, "../unsafe", settings)

    assert not settings.data_directory.exists()


def test_save_video_never_overwrites_existing_upload(tmp_path: Path) -> None:
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"new video")
    settings = store_module.StorageSettings(tmp_path / "storage")
    analysis_id = store_module.generate_analysis_id()
    settings.uploads_directory.mkdir(parents=True)
    existing = settings.uploads_directory / f"{analysis_id}.mp4"
    existing.write_bytes(b"existing video")

    with pytest.raises(FileExistsError):
        store_module.save_video(source, analysis_id, settings)

    assert existing.read_bytes() == b"existing video"


def test_save_video_removes_partial_file_when_copy_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"video content")
    settings = store_module.StorageSettings(tmp_path / "storage")
    analysis_id = store_module.generate_analysis_id()
    expected_destination = settings.uploads_directory / f"{analysis_id}.mp4"

    def interrupted_copy(source_file, destination_file, length) -> None:
        destination_file.write(b"partial")
        raise OSError("copy interrupted")

    monkeypatch.setattr(store_module, "copyfileobj", interrupted_copy)

    with pytest.raises(OSError, match="copy interrupted"):
        store_module.save_video(source, analysis_id, settings)

    assert not expected_destination.exists()


def test_prepare_debug_directory_creates_one_analysis_directory(
    tmp_path: Path,
) -> None:
    settings = store_module.StorageSettings(tmp_path / "storage")
    analysis_id = store_module.generate_analysis_id()

    debug_directory = store_module.prepare_debug_directory(analysis_id, settings)

    assert debug_directory == settings.debug_directory / analysis_id
    assert debug_directory.is_dir()
    assert list(debug_directory.iterdir()) == []


def test_prepare_debug_directory_rejects_invalid_id_without_creating_storage(
    tmp_path: Path,
) -> None:
    settings = store_module.StorageSettings(tmp_path / "storage")

    with pytest.raises(ValueError, match="analysis_id"):
        store_module.prepare_debug_directory("../unsafe", settings)

    assert not settings.data_directory.exists()


def test_prepare_debug_directory_never_reuses_existing_analysis_directory(
    tmp_path: Path,
) -> None:
    settings = store_module.StorageSettings(tmp_path / "storage")
    analysis_id = store_module.generate_analysis_id()
    existing = settings.debug_directory / analysis_id
    existing.mkdir(parents=True)
    marker = existing / "existing.jpg"
    marker.write_bytes(b"existing debug image")

    with pytest.raises(FileExistsError):
        store_module.prepare_debug_directory(analysis_id, settings)

    assert marker.read_bytes() == b"existing debug image"


def test_analyze_saved_video_passes_storage_debug_directory_to_checker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    settings = store_module.StorageSettings(tmp_path / "storage")
    analysis_id = store_module.generate_analysis_id()
    saved_video = store_module.save_video(source, analysis_id, settings)
    expected_result = object()
    received: dict[str, object] = {}

    def fake_analyze_video(video_path, **kwargs):
        received["video_path"] = video_path
        received.update(kwargs)
        return expected_result

    monkeypatch.setattr(store_module, "analyze_video", fake_analyze_video)

    result = store_module.analyze_saved_video(
        saved_video,
        analysis_id,
        label="real",
        phone_model="Pixel 7a",
        settings=settings,
    )

    assert result is expected_result
    assert received == {
        "video_path": saved_video,
        "label": "real",
        "phone_model": "Pixel 7a",
        "debug_output_directory": settings.debug_directory / analysis_id,
    }
    assert (settings.debug_directory / analysis_id).is_dir()


@pytest.mark.parametrize("mismatch", ["id", "directory"])
def test_analyze_saved_video_rejects_video_that_does_not_match_analysis(
    tmp_path: Path,
    mismatch: str,
) -> None:
    settings = store_module.StorageSettings(tmp_path / "storage")
    analysis_id = store_module.generate_analysis_id()
    other_id = store_module.generate_analysis_id()

    if mismatch == "id":
        video = settings.uploads_directory / f"{other_id}.mp4"
    else:
        video = tmp_path / f"{analysis_id}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")

    with pytest.raises(ValueError, match="matching video"):
        store_module.analyze_saved_video(video, analysis_id, settings=settings)

    assert not (settings.debug_directory / analysis_id).exists()


def _sample_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        verdict=Verdict.THREE_D,
        score=0.81,
        confidence=0.74,
        reason="Two surfaces at different angles were detected.",
        metrics=AnalysisMetrics(frames_sampled=32, frames_used=27),
        pairs=[PairMetrics(0, 4, 1.0, match_count=86, pair_score=0.82)],
        processing_ms=2800,
        label=GroundTruthLabel.REAL,
        phone_model="Pixel 7a",
    )


def _build_sample_stored_analysis(
    tmp_path: Path,
    *,
    attempt: int = 1,
    created_at: datetime | None = None,
) -> tuple[store_module.StoredAnalysis, store_module.StorageSettings]:
    settings = store_module.StorageSettings(tmp_path / "storage")
    analysis_id = store_module.generate_analysis_id()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    saved_video = store_module.save_video(source, analysis_id, settings)
    debug_directory = store_module.prepare_debug_directory(analysis_id, settings)
    debug_frame = debug_directory / "pair_000.jpg"
    debug_frame.write_bytes(b"debug image")
    result = _sample_analysis_result()
    result.debug_frames = [debug_frame]
    record = store_module.build_stored_analysis(
        analysis_id,
        saved_video,
        result,
        attempt=attempt,
        settings=settings,
        created_at=created_at,
    )
    return record, settings


def test_build_stored_analysis_derives_consistent_metadata(tmp_path: Path) -> None:
    created_at = datetime(2026, 10, 29, 12, 30, tzinfo=timezone.utc)

    record, settings = _build_sample_stored_analysis(
        tmp_path,
        attempt=2,
        created_at=created_at,
    )

    assert record.created_at == created_at
    assert record.attempt == 2
    assert record.video_path == settings.uploads_directory / f"{record.id}.mp4"
    assert record.result_path == settings.results_directory / f"{record.id}.json"
    assert record.debug_directory == settings.debug_directory / record.id
    assert record.result.verdict is Verdict.THREE_D


def test_build_stored_analysis_uses_timezone_aware_utc_timestamp(
    tmp_path: Path,
) -> None:
    record, _ = _build_sample_stored_analysis(tmp_path)

    assert record.created_at.tzinfo is not None
    assert record.created_at.utcoffset() == timezone.utc.utcoffset(record.created_at)


def test_stored_analysis_converts_to_complete_json_compatible_record(
    tmp_path: Path,
) -> None:
    record, _ = _build_sample_stored_analysis(tmp_path, attempt=2)

    data = record.to_dict()

    assert data["id"] == record.id
    assert data["attempt"] == 2
    assert data["verdict"] == "3d"
    assert data["label"] == "real"
    assert data["metrics"]["frames_used"] == 27
    assert data["video_path"] == str(record.video_path)
    assert data["result_path"] == str(record.result_path)
    assert data["debug_directory"] == str(record.debug_directory)
    datetime.fromisoformat(data["created_at"])
    json.dumps(data)


@pytest.mark.parametrize("attempt", [0, -1])
def test_stored_analysis_rejects_attempt_below_one(
    tmp_path: Path,
    attempt: int,
) -> None:
    with pytest.raises(ValueError, match="attempt must be at least 1"):
        _build_sample_stored_analysis(tmp_path, attempt=attempt)


@pytest.mark.parametrize("attempt", [True, 1.5, "1"])
def test_stored_analysis_rejects_non_integer_attempt(
    tmp_path: Path,
    attempt: object,
) -> None:
    with pytest.raises(TypeError, match="attempt must be an integer"):
        _build_sample_stored_analysis(tmp_path, attempt=attempt)  # type: ignore[arg-type]


def test_stored_analysis_rejects_timestamp_without_timezone(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="created_at must include timezone"):
        _build_sample_stored_analysis(
            tmp_path,
            created_at=datetime(2026, 10, 29, 12, 30),
        )


def test_build_stored_analysis_requires_saved_video_in_configured_uploads(
    tmp_path: Path,
) -> None:
    settings = store_module.StorageSettings(tmp_path / "storage")
    analysis_id = store_module.generate_analysis_id()
    outside_video = tmp_path / f"{analysis_id}.mp4"
    outside_video.write_bytes(b"video")
    store_module.prepare_debug_directory(analysis_id, settings)

    with pytest.raises(ValueError, match="configured uploads"):
        store_module.build_stored_analysis(
            analysis_id,
            outside_video,
            _sample_analysis_result(),
            settings=settings,
        )


def test_build_stored_analysis_requires_prepared_debug_directory(
    tmp_path: Path,
) -> None:
    settings = store_module.StorageSettings(tmp_path / "storage")
    analysis_id = store_module.generate_analysis_id()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    saved_video = store_module.save_video(source, analysis_id, settings)

    with pytest.raises(FileNotFoundError, match="debug directory"):
        store_module.build_stored_analysis(
            analysis_id,
            saved_video,
            _sample_analysis_result(),
            settings=settings,
        )


def test_save_result_writes_complete_portable_json(tmp_path: Path) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path, attempt=2)
    assert list(settings.results_directory.iterdir()) == []

    saved_path = store_module.save_result(record, settings)
    data = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path == record.result_path
    assert data["id"] == record.id
    assert data["attempt"] == 2
    assert data["verdict"] == "3d"
    assert data["score"] == 0.81
    assert data["metrics"]["frames_used"] == 27
    assert data["pairs"][0]["match_count"] == 86
    assert data["label"] == "real"
    assert data["phone_model"] == "Pixel 7a"
    assert data["video_path"] == f"uploads/{record.id}.mp4"
    assert data["result_path"] == f"results/{record.id}.json"
    assert data["debug_directory"] == f"debug/{record.id}"
    assert data["debug_frames"] == [f"debug/{record.id}/pair_000.jpg"]
    assert saved_path.read_bytes().endswith(b"\n")


def test_save_result_preserves_unicode_text(tmp_path: Path) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path)
    record.result.reason = "Box corner detected — परीक्षण सफल"

    saved_path = store_module.save_result(record, settings)
    saved_text = saved_path.read_text(encoding="utf-8")

    assert "Box corner detected — परीक्षण सफल" in saved_text


def test_save_result_never_overwrites_existing_json(tmp_path: Path) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path)
    settings.results_directory.mkdir(parents=True, exist_ok=True)
    record.result_path.write_text("existing result", encoding="utf-8")

    with pytest.raises(FileExistsError):
        store_module.save_result(record, settings)

    assert record.result_path.read_text(encoding="utf-8") == "existing result"
    assert list(settings.results_directory.glob("*.tmp")) == []


def test_save_result_rejects_result_path_outside_configuration(
    tmp_path: Path,
) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path)
    outside_record = store_module.StoredAnalysis(
        id=record.id,
        created_at=record.created_at,
        attempt=record.attempt,
        video_path=record.video_path,
        result_path=tmp_path / f"{record.id}.json",
        debug_directory=record.debug_directory,
        result=record.result,
    )

    with pytest.raises(ValueError, match="configured results"):
        store_module.save_result(outside_record, settings)

    assert not outside_record.result_path.exists()


def test_save_result_serialization_failure_creates_no_result_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path)

    def fail_serialization(*args, **kwargs):
        raise TypeError("cannot serialize")

    monkeypatch.setattr(store_module.json, "dumps", fail_serialization)

    with pytest.raises(TypeError, match="cannot serialize"):
        store_module.save_result(record, settings)

    assert not record.result_path.exists()
    assert list(settings.results_directory.iterdir()) == []


def test_save_result_publication_failure_removes_temporary_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path)

    def fail_publication(source, destination):
        raise OSError("publication interrupted")

    monkeypatch.setattr(store_module.os, "link", fail_publication)

    with pytest.raises(OSError, match="publication interrupted"):
        store_module.save_result(record, settings)

    assert not record.result_path.exists()
    assert list(settings.results_directory.glob("*.tmp")) == []


def _insert_test_analysis(
    connection: sqlite3.Connection,
    *,
    analysis_id: str | None = None,
    label: str | None = "real",
    phone_model: str | None = "Pixel 7a",
    attempt: object = 1,
    verdict: str = "3d",
    score: float = 0.81,
    confidence: float = 0.74,
) -> str:
    row_id = analysis_id or store_module.generate_analysis_id()
    connection.execute(
        """
        INSERT INTO analyses (
            id, created_at, label, phone_model, attempt, verdict, score,
            confidence, video_path, result_path, debug_directory
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row_id,
            "2026-10-30T10:00:00+00:00",
            label,
            phone_model,
            attempt,
            verdict,
            score,
            confidence,
            f"uploads/{row_id}.mp4",
            f"results/{row_id}.json",
            f"debug/{row_id}",
        ),
    )
    return row_id


def test_initialize_database_creates_expected_schema_and_version(
    tmp_path: Path,
) -> None:
    settings = store_module.StorageSettings(tmp_path / "custom-data")

    database_path = store_module.initialize_database(settings)

    assert database_path == settings.database_path
    assert database_path.is_file()
    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1]: {"type": row[2], "not_null": bool(row[3]), "primary_key": bool(row[5])}
            for row in connection.execute("PRAGMA table_info(analyses)")
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        indexes = {
            row[1]
            for row in connection.execute("PRAGMA index_list(analyses)")
        }

    assert list(columns) == [
        "id",
        "created_at",
        "label",
        "phone_model",
        "attempt",
        "verdict",
        "score",
        "confidence",
        "video_path",
        "result_path",
        "debug_directory",
    ]
    assert columns["id"]["primary_key"] is True
    assert columns["attempt"]["not_null"] is True
    assert columns["score"]["type"] == "REAL"
    assert version == store_module.DATABASE_SCHEMA_VERSION == 1
    assert {
        "idx_analyses_created_at",
        "idx_analyses_label",
        "idx_analyses_phone_model",
        "idx_analyses_verdict",
    }.issubset(indexes)


@pytest.mark.parametrize(
    "overrides",
    [
        {"analysis_id": "short-id"},
        {"label": "unknown"},
        {"phone_model": ""},
        {"attempt": 0},
        {"verdict": "unknown"},
        {"score": -0.1},
        {"score": 1.1},
        {"confidence": -0.1},
        {"confidence": 1.1},
    ],
)
def test_database_constraints_reject_invalid_analysis_rows(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    settings = store_module.StorageSettings(tmp_path / "data")
    store_module.initialize_database(settings)

    with sqlite3.connect(settings.database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            _insert_test_analysis(connection, **overrides)  # type: ignore[arg-type]


def test_initialize_database_is_repeatable_and_preserves_rows(
    tmp_path: Path,
) -> None:
    settings = store_module.StorageSettings(tmp_path / "data")
    store_module.initialize_database(settings)
    with sqlite3.connect(settings.database_path) as connection:
        row_id = _insert_test_analysis(connection)
        connection.commit()

    second_path = store_module.initialize_database(settings)

    with sqlite3.connect(second_path) as connection:
        stored = connection.execute(
            "SELECT id, verdict, score FROM analyses WHERE id = ?",
            (row_id,),
        ).fetchone()
    assert stored == (row_id, "3d", 0.81)


def test_initialize_database_rejects_newer_schema_version(tmp_path: Path) -> None:
    settings = store_module.StorageSettings(tmp_path / "data")
    settings.data_directory.mkdir(parents=True)
    newer_version = store_module.DATABASE_SCHEMA_VERSION + 1
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(f"PRAGMA user_version = {newer_version}")

    with pytest.raises(RuntimeError, match="newer than supported"):
        store_module.initialize_database(settings)

    with sqlite3.connect(settings.database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert version == newer_version


def test_initialize_database_rolls_back_partial_schema_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = store_module.StorageSettings(tmp_path / "data")
    monkeypatch.setattr(
        store_module,
        "_ANALYSES_INDEX_SQL",
        ("CREATE INDEX broken SQL",),
    )

    with pytest.raises(sqlite3.OperationalError):
        store_module.initialize_database(settings)

    with sqlite3.connect(settings.database_path) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'analyses'"
        ).fetchone()
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert table is None
    assert version == 0


def test_index_analysis_inserts_complete_summary_with_portable_paths(
    tmp_path: Path,
) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path, attempt=2)
    store_module.save_result(record, settings)

    inserted_id = store_module.index_analysis(record, settings)

    assert inserted_id == record.id
    with sqlite3.connect(settings.database_path) as connection:
        row = connection.execute(
            """
            SELECT id, created_at, label, phone_model, attempt, verdict, score,
                   confidence, video_path, result_path, debug_directory
            FROM analyses WHERE id = ?
            """,
            (record.id,),
        ).fetchone()
    assert row == (
        record.id,
        record.created_at.isoformat(),
        "real",
        "Pixel 7a",
        2,
        "3d",
        0.81,
        0.74,
        f"uploads/{record.id}.mp4",
        f"results/{record.id}.json",
        f"debug/{record.id}",
    )


def test_index_analysis_allows_unlabelled_analysis_without_phone_model(
    tmp_path: Path,
) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path)
    record.result.label = None
    record.result.phone_model = None
    store_module.save_result(record, settings)

    store_module.index_analysis(record, settings)

    with sqlite3.connect(settings.database_path) as connection:
        row = connection.execute(
            "SELECT label, phone_model FROM analyses WHERE id = ?",
            (record.id,),
        ).fetchone()
    assert row == (None, None)


def test_index_analysis_requires_saved_result_before_database_creation(
    tmp_path: Path,
) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path)

    with pytest.raises(FileNotFoundError, match="Stored result"):
        store_module.index_analysis(record, settings)

    assert not settings.database_path.exists()


@pytest.mark.parametrize("missing_artifact", ["video", "debug"])
def test_index_analysis_requires_all_saved_artifacts(
    tmp_path: Path,
    missing_artifact: str,
) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path)
    store_module.save_result(record, settings)
    if missing_artifact == "video":
        record.video_path.unlink()
        expected_message = "Stored video"
    else:
        for child in record.debug_directory.iterdir():
            child.unlink()
        record.debug_directory.rmdir()
        expected_message = "Stored debug directory"

    with pytest.raises(FileNotFoundError, match=expected_message):
        store_module.index_analysis(record, settings)

    assert not settings.database_path.exists()


def test_index_analysis_rejects_duplicate_without_changing_existing_row(
    tmp_path: Path,
) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path)
    store_module.save_result(record, settings)
    store_module.index_analysis(record, settings)
    record.result.score = 0.12

    with pytest.raises(sqlite3.IntegrityError):
        store_module.index_analysis(record, settings)

    with sqlite3.connect(settings.database_path) as connection:
        rows = connection.execute(
            "SELECT id, score FROM analyses WHERE id = ?",
            (record.id,),
        ).fetchall()
    assert rows == [(record.id, 0.81)]


def test_index_analysis_rejects_record_from_different_storage_configuration(
    tmp_path: Path,
) -> None:
    record, original_settings = _build_sample_stored_analysis(tmp_path)
    store_module.save_result(record, original_settings)
    other_settings = store_module.StorageSettings(tmp_path / "other-storage")

    with pytest.raises(ValueError, match="configured results"):
        store_module.index_analysis(record, other_settings)

    assert not other_settings.database_path.exists()


def _install_fake_checker(
    monkeypatch,
    *,
    verdict: Verdict = Verdict.THREE_D,
) -> None:
    def fake_analyze_video(
        video_path,
        label=None,
        phone_model=None,
        debug_output_directory=None,
    ) -> AnalysisResult:
        debug_frame = Path(debug_output_directory) / "pair_000.jpg"
        debug_frame.write_bytes(b"debug image")
        return AnalysisResult(
            verdict=verdict,
            score=0.81 if verdict is Verdict.THREE_D else 0.5,
            confidence=0.74 if verdict is Verdict.THREE_D else 0.0,
            reason=(
                "Two surfaces detected."
                if verdict is Verdict.THREE_D
                else "Not enough movement."
            ),
            metrics=AnalysisMetrics(frames_sampled=12, frames_used=10),
            processing_ms=120,
            label=label,
            phone_model=phone_model,
            debug_frames=[debug_frame],
        )

    monkeypatch.setattr(store_module, "analyze_video", fake_analyze_video)


def test_create_analysis_completes_full_storage_workflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_checker(monkeypatch)
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"original video")
    settings = store_module.StorageSettings(tmp_path / "storage")

    record = store_module.create_analysis(
        source,
        label="real",
        phone_model="Pixel 7a",
        attempt=2,
        settings=settings,
    )

    assert isinstance(record, store_module.StoredAnalysis)
    assert record.attempt == 2
    assert record.result.label is GroundTruthLabel.REAL
    assert record.result.phone_model == "Pixel 7a"
    assert record.video_path.read_bytes() == b"original video"
    assert record.result_path.is_file()
    assert record.debug_directory.is_dir()
    assert record.result.debug_frames[0].is_file()
    assert record.video_path.stem == record.id
    assert record.result_path.stem == record.id
    assert record.debug_directory.name == record.id

    saved_json = json.loads(record.result_path.read_text(encoding="utf-8"))
    assert saved_json["id"] == record.id
    assert saved_json["attempt"] == 2
    with sqlite3.connect(settings.database_path) as connection:
        indexed = connection.execute(
            "SELECT id, label, phone_model, attempt, verdict FROM analyses"
        ).fetchall()
    assert indexed == [(record.id, "real", "Pixel 7a", 2, "3d")]


def test_create_analysis_saves_unlabelled_insufficient_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_checker(monkeypatch, verdict=Verdict.INSUFFICIENT)
    source = tmp_path / "recording.mov"
    source.write_bytes(b"video")
    settings = store_module.StorageSettings(tmp_path / "storage")

    record = store_module.create_analysis(source, settings=settings)

    assert record.result.verdict is Verdict.INSUFFICIENT
    assert record.result.label is None
    assert record.result.phone_model is None
    with sqlite3.connect(settings.database_path) as connection:
        indexed = connection.execute(
            "SELECT label, phone_model, verdict FROM analyses WHERE id = ?",
            (record.id,),
        ).fetchone()
    assert indexed == (None, None, "insufficient")


def test_create_analysis_applies_result_policy_before_every_persistence_step(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_checker(monkeypatch, verdict=Verdict.INSUFFICIENT)
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"video")
    settings = store_module.StorageSettings(tmp_path / "storage")

    record = store_module.create_analysis(
        source,
        attempt=2,
        settings=settings,
        result_policy=apply_review_policy,
    )

    assert record.result.verdict is Verdict.INSUFFICIENT
    assert record.verdict == "review"
    saved_json = json.loads(record.result_path.read_text(encoding="utf-8"))
    assert saved_json["verdict"] == "review"
    with sqlite3.connect(settings.database_path) as connection:
        indexed_verdict = connection.execute(
            "SELECT verdict FROM analyses WHERE id = ?",
            (record.id,),
        ).fetchone()
    assert indexed_verdict == ("review",)
    restored = store_module.get_analysis(record.id, settings)
    assert restored is not None
    assert restored.verdict == "review"
    assert restored.result.verdict is Verdict.INSUFFICIENT


def test_create_analysis_passes_result_and_attempt_to_policy_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_checker(monkeypatch, verdict=Verdict.INSUFFICIENT)
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"video")
    settings = store_module.StorageSettings(tmp_path / "storage")
    calls: list[tuple[AnalysisResult, int]] = []

    def recording_policy(result: AnalysisResult, attempt: int) -> str:
        calls.append((result, attempt))
        return result.verdict.value

    record = store_module.create_analysis(
        source,
        attempt=3,
        settings=settings,
        result_policy=recording_policy,
    )

    assert calls == [(record.result, 3)]


def test_create_analysis_rejects_non_callable_policy_before_storage_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"video")
    settings = store_module.StorageSettings(tmp_path / "storage")

    with pytest.raises(TypeError, match="result_policy must be callable or None"):
        store_module.create_analysis(
            source,
            settings=settings,
            result_policy=object(),  # type: ignore[arg-type]
        )

    assert not settings.data_directory.exists()


def test_create_analysis_cleans_artifacts_when_policy_returns_invalid_value(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_checker(monkeypatch)
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"video")
    settings = store_module.StorageSettings(tmp_path / "storage")

    with pytest.raises(ValueError, match="result_policy must return one of"):
        store_module.create_analysis(
            source,
            settings=settings,
            result_policy=lambda result, attempt: "unknown",
        )

    assert list(settings.uploads_directory.iterdir()) == []
    assert list(settings.results_directory.iterdir()) == []
    assert list(settings.debug_directory.iterdir()) == []
    assert not settings.database_path.exists()


@pytest.mark.parametrize(
    "arguments,exception_type,message",
    [
        ({"attempt": 0}, ValueError, "attempt must be at least 1"),
        ({"attempt": True}, TypeError, "attempt must be an integer"),
        ({"label": "unknown"}, ValueError, "unknown"),
        ({"phone_model": "   "}, ValueError, "phone_model cannot be blank"),
    ],
)
def test_create_analysis_rejects_invalid_metadata_before_storage_changes(
    tmp_path: Path,
    arguments: dict[str, object],
    exception_type: type[Exception],
    message: str,
) -> None:
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"video")
    settings = store_module.StorageSettings(tmp_path / "storage")

    with pytest.raises(exception_type, match=message):
        store_module.create_analysis(source, settings=settings, **arguments)

    assert not settings.data_directory.exists()


def test_create_analysis_rejects_invalid_source_before_storage_changes(
    tmp_path: Path,
) -> None:
    settings = store_module.StorageSettings(tmp_path / "storage")

    with pytest.raises(FileNotFoundError, match="not a file"):
        store_module.create_analysis(tmp_path / "missing.mp4", settings=settings)

    unsupported = tmp_path / "recording.txt"
    unsupported.write_bytes(b"not video")
    with pytest.raises(ValueError, match="Unsupported video extension"):
        store_module.create_analysis(unsupported, settings=settings)

    assert not settings.data_directory.exists()


def test_create_analysis_cleans_new_artifacts_when_checker_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"video")
    settings = store_module.StorageSettings(tmp_path / "storage")

    def fail_checker(*args, **kwargs):
        raise RuntimeError("checker failed")

    monkeypatch.setattr(store_module, "analyze_video", fail_checker)

    with pytest.raises(RuntimeError, match="checker failed"):
        store_module.create_analysis(source, settings=settings)

    assert list(settings.uploads_directory.iterdir()) == []
    assert list(settings.results_directory.iterdir()) == []
    assert list(settings.debug_directory.iterdir()) == []
    assert not settings.database_path.exists()


@pytest.mark.parametrize("failure_stage", ["json", "database"])
def test_create_analysis_cleans_new_artifacts_after_storage_failure(
    tmp_path: Path,
    monkeypatch,
    failure_stage: str,
) -> None:
    _install_fake_checker(monkeypatch)
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"video")
    settings = store_module.StorageSettings(tmp_path / "storage")

    if failure_stage == "json":
        monkeypatch.setattr(
            store_module,
            "save_result",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("json failed")),
        )
        expected_message = "json failed"
    else:
        monkeypatch.setattr(
            store_module,
            "index_analysis",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("index failed")),
        )
        expected_message = "index failed"

    with pytest.raises(RuntimeError, match=expected_message):
        store_module.create_analysis(source, settings=settings)

    assert list(settings.uploads_directory.iterdir()) == []
    assert list(settings.results_directory.iterdir()) == []
    assert list(settings.debug_directory.iterdir()) == []


def test_create_analysis_failure_never_removes_preexisting_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"new video")
    settings = store_module.StorageSettings(tmp_path / "storage")
    analysis_id = store_module.generate_analysis_id()
    settings.uploads_directory.mkdir(parents=True)
    settings.results_directory.mkdir(parents=True)
    existing_video = settings.uploads_directory / f"{analysis_id}.mp4"
    existing_result = settings.results_directory / f"{analysis_id}.json"
    existing_debug = settings.debug_directory / analysis_id
    existing_debug.mkdir(parents=True)
    existing_video.write_bytes(b"existing video")
    existing_result.write_text("existing result", encoding="utf-8")
    marker = existing_debug / "existing.jpg"
    marker.write_bytes(b"existing debug")
    monkeypatch.setattr(store_module, "generate_analysis_id", lambda: analysis_id)

    with pytest.raises(FileExistsError):
        store_module.create_analysis(source, settings=settings)

    assert existing_video.read_bytes() == b"existing video"
    assert existing_result.read_text(encoding="utf-8") == "existing result"
    assert marker.read_bytes() == b"existing debug"


def test_list_analyses_returns_empty_history_for_fresh_storage(
    tmp_path: Path,
) -> None:
    settings = store_module.StorageSettings(tmp_path / "storage")

    assert store_module.list_analyses(settings=settings) == []
    assert settings.database_path.is_file()


def test_list_analyses_is_newest_first_and_supports_pagination(
    tmp_path: Path,
) -> None:
    older, settings = _build_sample_stored_analysis(
        tmp_path,
        created_at=datetime(2026, 10, 29, 10, 0, tzinfo=timezone.utc),
    )
    store_module.save_result(older, settings)
    store_module.index_analysis(older, settings)
    newer, _ = _build_sample_stored_analysis(
        tmp_path,
        created_at=datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc),
    )
    store_module.save_result(newer, settings)
    store_module.index_analysis(newer, settings)

    all_summaries = store_module.list_analyses(settings=settings)
    second_page = store_module.list_analyses(limit=1, offset=1, settings=settings)

    assert [summary.id for summary in all_summaries] == [newer.id, older.id]
    assert all(isinstance(item, store_module.AnalysisSummary) for item in all_summaries)
    assert second_page[0].id == older.id
    assert all_summaries[0].video_path == newer.video_path.resolve()


@pytest.mark.parametrize(
    "arguments,exception_type,message",
    [
        ({"limit": 0}, ValueError, "limit must be between"),
        ({"limit": 101}, ValueError, "limit must be between"),
        ({"limit": True}, TypeError, "limit must be an integer"),
        ({"offset": -1}, ValueError, "offset must be non-negative"),
        ({"offset": False}, TypeError, "offset must be an integer"),
    ],
)
def test_list_analyses_rejects_invalid_pagination(
    tmp_path: Path,
    arguments: dict[str, object],
    exception_type: type[Exception],
    message: str,
) -> None:
    settings = store_module.StorageSettings(tmp_path / "storage")

    with pytest.raises(exception_type, match=message):
        store_module.list_analyses(settings=settings, **arguments)

    assert not settings.data_directory.exists()


def test_get_result_and_get_analysis_restore_complete_saved_data(
    tmp_path: Path,
) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path, attempt=2)
    store_module.save_result(record, settings)
    store_module.index_analysis(record, settings)

    raw_result = store_module.get_result(record.id, settings)
    restored = store_module.get_analysis(record.id, settings)

    assert raw_result is not None
    assert raw_result["id"] == record.id
    assert raw_result["pairs"][0]["match_count"] == 86
    assert restored is not None
    assert restored.id == record.id
    assert restored.attempt == 2
    assert restored.result.metrics.frames_used == 27
    assert restored.result.pairs[0].match_count == 86
    assert restored.result.label is GroundTruthLabel.REAL
    assert restored.result.debug_frames == record.result.debug_frames


def test_get_operations_return_none_for_unknown_analysis(tmp_path: Path) -> None:
    settings = store_module.StorageSettings(tmp_path / "storage")
    unknown_id = store_module.generate_analysis_id()

    assert store_module.get_result(unknown_id, settings) is None
    assert store_module.get_analysis(unknown_id, settings) is None
    assert store_module.get_debug_frame(unknown_id, 0, settings) is None


@pytest.mark.parametrize("corruption", ["missing", "invalid_json", "mismatch"])
def test_get_result_reports_missing_or_corrupt_result_json(
    tmp_path: Path,
    corruption: str,
) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path)
    store_module.save_result(record, settings)
    store_module.index_analysis(record, settings)
    if corruption == "missing":
        record.result_path.unlink()
        expected_exception = FileNotFoundError
        message = "does not exist"
    elif corruption == "invalid_json":
        record.result_path.write_text("{invalid", encoding="utf-8")
        expected_exception = ValueError
        message = "JSON is invalid"
    else:
        data = json.loads(record.result_path.read_text(encoding="utf-8"))
        data["score"] = 0.01
        record.result_path.write_text(json.dumps(data), encoding="utf-8")
        expected_exception = ValueError
        message = "disagree on score"

    with pytest.raises(expected_exception, match=message):
        store_module.get_result(record.id, settings)


def test_get_result_rejects_unsafe_path_from_database(tmp_path: Path) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path)
    store_module.save_result(record, settings)
    store_module.index_analysis(record, settings)
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(
            "UPDATE analyses SET result_path = ? WHERE id = ?",
            ("../outside.json", record.id),
        )

    with pytest.raises(ValueError, match="escapes the data directory"):
        store_module.get_result(record.id, settings)


def test_get_debug_frame_returns_verified_image_and_handles_range(
    tmp_path: Path,
) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path)
    store_module.save_result(record, settings)
    store_module.index_analysis(record, settings)

    frame = store_module.get_debug_frame(record.id, 0, settings)

    assert frame == record.result.debug_frames[0].resolve()
    assert store_module.get_debug_frame(record.id, 1, settings) is None


@pytest.mark.parametrize("frame_number", [-1, True, 1.5])
def test_get_debug_frame_rejects_invalid_frame_number(
    tmp_path: Path,
    frame_number: object,
) -> None:
    settings = store_module.StorageSettings(tmp_path / "storage")
    exception = TypeError if isinstance(frame_number, (bool, float)) else ValueError

    with pytest.raises(exception):
        store_module.get_debug_frame(
            store_module.generate_analysis_id(),
            frame_number,  # type: ignore[arg-type]
            settings,
        )


def test_get_debug_frame_reports_missing_image(tmp_path: Path) -> None:
    record, settings = _build_sample_stored_analysis(tmp_path)
    store_module.save_result(record, settings)
    store_module.index_analysis(record, settings)
    record.result.debug_frames[0].unlink()

    with pytest.raises(FileNotFoundError, match="debug frame does not exist"):
        store_module.get_debug_frame(record.id, 0, settings)


def test_rerun_analysis_creates_new_result_and_preserves_original(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_checker(monkeypatch)
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"original video")
    settings = store_module.StorageSettings(tmp_path / "storage")
    original = store_module.create_analysis(
        source,
        label="real",
        phone_model="Pixel 7a",
        attempt=2,
        settings=settings,
    )
    original_video = original.video_path.read_bytes()
    original_json = original.result_path.read_bytes()

    rerun = store_module.rerun_analysis(original.id, settings)

    assert rerun is not None
    assert rerun.id != original.id
    assert rerun.video_path != original.video_path
    assert rerun.video_path.read_bytes() == original_video
    assert rerun.result.label is GroundTruthLabel.REAL
    assert rerun.result.phone_model == "Pixel 7a"
    assert rerun.attempt == 2
    assert original.video_path.read_bytes() == original_video
    assert original.result_path.read_bytes() == original_json
    with sqlite3.connect(settings.database_path) as connection:
        ids = {
            row[0]
            for row in connection.execute("SELECT id FROM analyses")
        }
    assert ids == {original.id, rerun.id}


def test_rerun_analysis_forwards_result_policy_and_preserves_original(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_checker(monkeypatch, verdict=Verdict.INSUFFICIENT)
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"original video")
    settings = store_module.StorageSettings(tmp_path / "storage")
    original = store_module.create_analysis(
        source,
        attempt=2,
        settings=settings,
    )
    original_json = original.result_path.read_bytes()

    rerun = store_module.rerun_analysis(
        original.id,
        settings,
        result_policy=apply_review_policy,
    )

    assert rerun is not None
    assert rerun.id != original.id
    assert rerun.result.verdict is Verdict.INSUFFICIENT
    assert rerun.verdict == "review"
    assert original.result.verdict is Verdict.INSUFFICIENT
    assert original.result_path.read_bytes() == original_json
    assert json.loads(rerun.result_path.read_text(encoding="utf-8"))["verdict"] == "review"


def test_rerun_analysis_returns_none_for_unknown_id(tmp_path: Path) -> None:
    settings = store_module.StorageSettings(tmp_path / "storage")

    assert (
        store_module.rerun_analysis(store_module.generate_analysis_id(), settings)
        is None
    )


def test_create_storage_directories_creates_expected_layout(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"

    directories = create_storage_directories(data_directory)

    assert set(directories) == set(STORAGE_SUBDIRECTORIES)
    for name in STORAGE_SUBDIRECTORIES:
        expected_directory = data_directory / name
        assert directories[name] == expected_directory
        assert expected_directory.is_dir()


def test_create_storage_directories_accepts_storage_settings(tmp_path: Path) -> None:
    settings = store_module.StorageSettings(tmp_path / "configured-data")

    directories = create_storage_directories(settings)

    assert directories == settings.directories
    assert all(directory.is_dir() for directory in directories.values())


def test_create_storage_directories_is_safe_to_repeat(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"
    first_result = create_storage_directories(data_directory)
    marker = first_result["uploads"] / "existing-video.mp4"
    marker.write_bytes(b"existing video")

    second_result = create_storage_directories(data_directory)

    assert second_result == first_result
    assert marker.read_bytes() == b"existing video"


def test_create_storage_directories_creates_missing_parent_folders(
    tmp_path: Path,
) -> None:
    nested_data_directory = tmp_path / "backend" / "data"

    create_storage_directories(nested_data_directory)

    assert all(
        (nested_data_directory / name).is_dir()
        for name in STORAGE_SUBDIRECTORIES
    )
