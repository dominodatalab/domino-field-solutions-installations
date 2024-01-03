import os

from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider


class DominoApiKeyRequestHeaderProvider(RequestHeaderProvider):
    """
    Provides X-Domino-Api-Key request header based on DOMINO_USER_API_KEY environment variable.
    """

    def __init__(self):
        self._domino_api_key = os.getenv("DOMINO_USER_API_KEY")

    def in_context(self):
        return self._domino_api_key is not None

    def request_headers(self):
        request_headers = {}

        if self._domino_api_key is not None:
            request_headers["X-Domino-Api-Key"] = self._domino_api_key

        return request_headers
