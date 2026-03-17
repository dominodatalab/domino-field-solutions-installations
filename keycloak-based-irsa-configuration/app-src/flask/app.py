import os
import requests
from flask import Flask, render_template, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix
from dataclasses import dataclass
from typing import Dict

app = Flask(__name__)

# Configure for reverse proxy
ROOT_PATH = os.getenv('APP_ROOT_PATH', '').strip().rstrip('/')
if ROOT_PATH:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)


@dataclass(frozen=True)
class Settings:
    domino_host: str
    irsa_mgmt_host: str
    http_verify_tls: bool
    default_iam_role_arn: str


def load_settings() -> Settings:
    domino_host = os.getenv("DOMINO_API_HOST", "http://nucleus-frontend.domino-platform:80").strip().rstrip("/")
    irsa_mgmt_host = os.getenv("DOMINO_IRSA_MGMT_HOST",
                               "http://domino-irsa-lite-admin.domino-field.svc.cluster.local:8000").strip().rstrip("/")
    http_verify_tls = os.getenv("HTTP_VERIFY_TLS", "false").strip().lower() in ("1", "true", "yes")
    default_iam_role_arn = os.getenv("DEFAULT_IAM_ROLE_ARN", "").strip()

    return Settings(
        domino_host=domino_host,
        irsa_mgmt_host=irsa_mgmt_host,
        http_verify_tls=http_verify_tls,
        default_iam_role_arn=default_iam_role_arn,
    )


settings = load_settings()


def get_domino_token():
    proxy = os.getenv('DOMINO_API_PROXY')
    if not proxy:
        return ""
    if not proxy.startswith(('http://', 'https://')):
        proxy = f"http://{proxy}"
    return requests.get(f"{proxy}/access-token").text


def get_request_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    api_key = os.getenv('DOMINO_USER_API_KEY')

    if api_key:
        headers["X-Domino-Api-Key"] = api_key
    else:
        token = get_domino_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


@app.route('/')
def index():
    # Pass root path to template for JavaScript API calls
    base_path = ROOT_PATH + '/' if ROOT_PATH else '/'
    return render_template('index.html', base_path=base_path)


@app.route('/presentation')
def presentation():
    return render_template('presentation.html')


@app.route('/docs')
def docs():
    return render_template('docs.html')


@app.route('/api/settings')
def get_settings():
    return jsonify({"default_iam_role_arn": settings.default_iam_role_arn})


@app.route('/api/domino-mappings')
def get_domino_mappings():
    user_url = f"{settings.domino_host}/v4/users"
    irsa_mgmt_url = f"{settings.irsa_mgmt_host}/k8s/serviceaccounts"
    headers = get_request_headers()

    try:
        resp = requests.get(user_url, headers=headers, timeout=15.0, verify=settings.http_verify_tls)
        resp.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to fetch users: {e}"}), 500

    users_payload = resp.json()
    domino_users = [
        u.get("userName").strip()
        for u in users_payload
        if isinstance(u, dict) and u.get("userName")
    ]

    try:
        resp = requests.get(irsa_mgmt_url, headers=headers, timeout=15.0, verify=settings.http_verify_tls)
        resp.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to fetch service accounts: {e}"}), 500

    k8s_items = resp.json().get("items", [])
    sa_map = {sa.get("name", ""): sa for sa in k8s_items if sa.get("name")}

    rows = []
    for user in domino_users:
        sa_obj = sa_map.get(user)
        if sa_obj:
            sa_name = sa_obj.get("name", "")
            role_arn = sa_obj.get("annotations", {}).get("eks.amazonaws.com/role-arn", "")
        else:
            sa_name = ""
            role_arn = ""

        rows.append({
            "domino_user": user,
            "service_account": sa_name,
            "default_aws_iam_role": role_arn
        })

    return jsonify({"mappings": rows})


@app.route('/api/keycloak-data')
def get_keycloak_data():
    groups_url = f"{settings.irsa_mgmt_host}/keycloak/groups"
    trust_url = f"{settings.irsa_mgmt_host}/iam/irsa/trust"
    headers = get_request_headers()

    try:
        resp_groups = requests.get(groups_url, headers=headers, timeout=15.0, verify=settings.http_verify_tls)
        resp_groups.raise_for_status()
        groups_data = resp_groups.json()
    except requests.RequestException as e:
        return jsonify({"error": f"Groups API Error: {e}"}), 500

    trust_data = {"items": []}
    try:
        resp_trust = requests.get(trust_url, headers=headers, timeout=15.0, verify=settings.http_verify_tls)
        resp_trust.raise_for_status()
        trust_data = resp_trust.json()
    except requests.RequestException:
        pass

    trust_map = {item.get('group', ''): item.get('subs', []) for item in trust_data.get('items', [])}
    groups_raw = groups_data.get("groups", [])

    group_rows = []
    all_members = []

    for g in groups_raw:
        raw_name = g.get("name", "")
        g_id = g.get("id", "")
        display_name = raw_name

        if "->" in display_name:
            parts = display_name.split("->")
            proxy_arn = parts[0].strip()
            target_arn = parts[1].strip()
        else:
            proxy_arn = ""
            target_arn = display_name

        trusted_subs = trust_map.get(raw_name, [])

        group_rows.append({
            "group_name": display_name,
            "group_id": g_id,
            "target_role_arn": target_arn,
            "proxy_role_arn": proxy_arn,
            "trusted_subs": trusted_subs
        })

        for u_name in g.get("members", []):
            if u_name:
                all_members.append({
                    "group_id": g_id,
                    "domino_user_name": u_name,
                    "sub": f"system:serviceaccount:domino-compute:{u_name.replace('.', '-')}"
                })

    return jsonify({"groups": group_rows, "members": all_members})


@app.route('/api/sync-users', methods=['POST'])
def sync_users():
    data = request.json
    aws_arn = data.get('default_aws_role_arn', '')

    if not aws_arn:
        return jsonify({"error": "Please provide a Target IAM Role ARN"}), 400

    api_url = f"{settings.irsa_mgmt_host}/k8s/users/sync"
    payload = {"default_aws_role_arn": aws_arn, "dry_run": False}

    try:
        response = requests.post(api_url, headers=get_request_headers(), json=payload, verify=settings.http_verify_tls)
        if response.status_code == 200:
            return jsonify({"success": True, "message": "Sync Complete!"})
        else:
            return jsonify({"error": f"Sync failed: {response.text}"}), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Connection error: {e}"}), 500


@app.route('/api/irsa-sync', methods=['POST'])
def irsa_sync():
    sync_url = f"{settings.irsa_mgmt_host}/iam/irsa/sync"
    payload = {"dry_run": False}

    try:
        response = requests.post(sync_url, headers=get_request_headers(), json=payload, verify=settings.http_verify_tls)
        if response.status_code == 200:
            return jsonify({"success": True, "message": "IRSA Sync initiated successfully!"})
        else:
            return jsonify({"error": f"IRSA Sync failed: {response.text}"}), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Connection error: {e}"}), 500


@app.route('/api/keycloak/member', methods=['POST'])
def add_member():
    data = request.json
    api_url = f"{settings.irsa_mgmt_host}/keycloak/groups/member"

    try:
        response = requests.post(api_url, headers=get_request_headers(), json=data, verify=settings.http_verify_tls)
        if response.status_code == 201:
            return jsonify({"success": True, "message": f"Added {data.get('username')} successfully"})
        else:
            return jsonify({"error": f"Failed to add: {response.text}"}), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Connection error: {e}"}), 500


@app.route('/api/keycloak/member', methods=['DELETE'])
def remove_member():
    data = request.json
    api_url = f"{settings.irsa_mgmt_host}/keycloak/groups/member"

    try:
        response = requests.delete(api_url, headers=get_request_headers(), json=data, verify=settings.http_verify_tls)
        if response.status_code == 200:
            return jsonify({"success": True, "message": f"Removed {data.get('username')}"})
        else:
            return jsonify({"error": f"Failed to remove: {response.text}"}), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Connection error: {e}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=False)