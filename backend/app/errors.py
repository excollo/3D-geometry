"""HTTP-facing application errors shared by API route modules."""

from __future__ import annotations


__all__ = [
    "ApiError",
    "BadRequestError",
    "NotFoundError",
    "UnprocessableEntityError",
]


class ApiError(Exception):
    """A safe error whose detail may be returned to an API client."""

    status_code: int = 500
    default_detail: str = "Request failed."

    def __init__(self, detail: str | None = None) -> None:
        resolved_detail = self.default_detail if detail is None else detail
        if not isinstance(resolved_detail, str):
            raise TypeError("detail must be a string")
        if not resolved_detail.strip():
            raise ValueError("detail cannot be blank")
        self.detail = resolved_detail
        super().__init__(resolved_detail)


class BadRequestError(ApiError):
    """A request whose supplied content cannot be processed."""

    status_code = 400
    default_detail = "The request could not be processed."


class NotFoundError(ApiError):
    """A requested analysis or artifact that does not exist."""

    status_code = 404
    default_detail = "The requested resource was not found."


class UnprocessableEntityError(ApiError):
    """A request with invalid structured fields or identifiers."""

    status_code = 422
    default_detail = "The request fields are invalid."
