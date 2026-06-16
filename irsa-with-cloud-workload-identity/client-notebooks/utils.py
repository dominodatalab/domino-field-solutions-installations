import requests
import os

def generate_api_key_based_request_headers():
    domino_user_api_key = os.environ.get("DOMINO_USER_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "X-Domino-Api-Key": domino_user_api_key,
    }
    return headers

def generate_auth_token_based_request_headers():
    access_token_endpoint = "http://localhost:8899/access-token"
    resp = requests.get(access_token_endpoint)
    token = resp.text
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token,
    }
    return headers

headers = generate_api_key_based_request_headers()
DOMINO_NUCLEUS_URI = os.environ.get("NUCLEUS_URI")