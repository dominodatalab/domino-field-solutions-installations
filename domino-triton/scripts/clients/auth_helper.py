#!/usr/bin/env python3
"""
Shared authentication helper for Triton client scripts.

See docs/authz.md for complete authentication documentation.

Auth Resolution Order:
1. DOMINO_USER_TOKEN env var - manually set Bearer token (for testing)
2. $DOMINO_API_PROXY/access-token - automatic token fetch (inside Domino workspaces)
3. DOMINO_USER_API_KEY env var - API key fallback

Token Caching:
- Tokens fetched from DOMINO_API_PROXY are cached indefinitely
- On 40x errors, call invalidate_token() to clear cache and refetch on next call

Usage:
    from auth_helper import get_auth_headers, invalidate_token

    headers = get_auth_headers()
    response = client.infer(model_name, inputs, headers=headers)

    # If you get a 401/403, invalidate and retry:
    if response.status_code in (401, 403):
        invalidate_token()
        headers = get_auth_headers()  # Will refetch token
        response = client.infer(model_name, inputs, headers=headers)
"""

import logging
import os
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Cache the fetched token to avoid repeated calls
_cached_token: Optional[str] = None


def _fetch_access_token(force_refresh: bool = False) -> Optional[str]:
    """
    Fetch access token from Domino API proxy.

    Inside Domino workspaces, DOMINO_API_PROXY is set and provides
    an /access-token endpoint that returns a Bearer token.

    Args:
        force_refresh: If True, bypass cache and fetch fresh token

    Returns:
        Token string or None if not available
    """
    global _cached_token

    # Return cached token if available and not forcing refresh
    if _cached_token is not None and not force_refresh:
        return _cached_token

    api_proxy = os.environ.get("DOMINO_API_PROXY")
    if not api_proxy:
        logger.debug("DOMINO_API_PROXY not set, skipping token fetch")
        return None

    url = f"{api_proxy.rstrip('/')}/access-token"
    try:
        resp = requests.get(url, timeout=5.0)
        if resp.status_code == 200:
            # Response is the raw token string
            token = resp.text.strip()
            if token:
                _cached_token = token
                logger.debug("Successfully fetched access token from DOMINO_API_PROXY")
                return token
            else:
                logger.warning("Empty token returned from DOMINO_API_PROXY/access-token")
        else:
            logger.warning(f"Failed to fetch access token: HTTP {resp.status_code}")
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch access token from {url}: {e}")

    return None


def get_auth_headers() -> Optional[Dict[str, str]]:
    """
    Get authentication headers for Triton client requests.

    Resolution order:
    1. DOMINO_USER_TOKEN env var - Bearer token (for manual testing)
    2. DOMINO_API_PROXY/access-token - automatic token fetch (inside Domino)
    3. DOMINO_USER_API_KEY env var - API key fallback

    Returns:
        Dict with auth header, or None if no auth configured.
        For Bearer tokens: {"authorization": "Bearer <token>"}
        For API keys: {"x-domino-api-key": "<key>"}
    """
    # 1. Check for manually set token (testing/development)
    token = os.environ.get("DOMINO_USER_TOKEN")
    if token:
        logger.debug("Using DOMINO_USER_TOKEN for authentication")
        return {"authorization": f"Bearer {token}"}

    # 2. Try to fetch from DOMINO_API_PROXY/access-token (inside Domino)
    token = _fetch_access_token()
    if token:
        return {"authorization": f"Bearer {token}"}

    # 3. Fall back to API key
    api_key = os.environ.get("DOMINO_USER_API_KEY")
    if api_key:
        logger.debug("Using DOMINO_USER_API_KEY for authentication")
        return {"x-domino-api-key": api_key}

    # No auth configured
    logger.debug("No authentication configured")
    return None


def invalidate_token():
    """
    Invalidate the cached token.

    Call this when you receive a 401/403 error to trigger a token refresh
    on the next get_auth_headers() call.

    Example:
        response = make_request(headers=get_auth_headers())
        if response.status_code in (401, 403):
            invalidate_token()
            response = make_request(headers=get_auth_headers())  # Fresh token
    """
    global _cached_token
    _cached_token = None
    logger.debug("Token cache invalidated, will refetch on next request")


# Backwards compatibility alias
clear_token_cache = invalidate_token


def merge_auth_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Merge auth headers into an existing headers dict.

    Useful when you need to add auth to headers that already contain
    other headers like Content-Type.

    Args:
        headers: Existing headers dict to merge into

    Returns:
        Headers dict with auth headers added (if available)
    """
    auth_headers = get_auth_headers()
    if auth_headers:
        headers.update(auth_headers)
    return headers