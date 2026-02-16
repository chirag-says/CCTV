"""
Custom exception classes for structured error handling.
"""

from fastapi import HTTPException, status


class NotFoundException(HTTPException):
    def __init__(self, resource: str, identifier: str = ""):
        detail = f"{resource} not found"
        if identifier:
            detail = f"{resource} with id '{identifier}' not found"
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class DuplicateException(HTTPException):
    def __init__(self, resource: str, field: str = ""):
        detail = f"{resource} already exists"
        if field:
            detail = f"{resource} with this {field} already exists"
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)


class ValidationException(HTTPException):
    def __init__(self, message: str):
        super().__init__(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=message)


class CameraException(HTTPException):
    def __init__(self, message: str):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=message)


class PipelineException(Exception):
    """Exception raised within the vision pipeline."""
    def __init__(self, message: str, camera_id: str = ""):
        self.camera_id = camera_id
        super().__init__(message)
