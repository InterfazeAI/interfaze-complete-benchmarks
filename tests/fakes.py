class FakeAPIError(Exception):
    """Mimics an SDK error carrying a status code + message."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
