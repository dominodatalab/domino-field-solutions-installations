import os

import jwt
from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider


class DominoExecutionRequestHeaderProvider(RequestHeaderProvider):
    """
    Provides X-Domino-Execution request header based on DOMINO_RUN_ID environment variable.
    """

    def __init__(self):
        run_id = os.getenv("DOMINO_RUN_ID")
        self._domino_execution = jwt.encode({"execution_id": run_id}, "secret", algorithm="HS256")

    def in_context(self):
        return self._domino_execution is not None

    def request_headers(self):
        request_headers = {}

        if self._domino_execution is not None:
            request_headers["X-Domino-Execution"] = self._domino_execution

        return request_headers
