#!/usr/bin/env python3
"""
kc_apply_group_spec.py

Apply Keycloak IRSA groups from a JSON spec (flat under root) and reconcile memberships.

Spec shape:
{
  "root": "domino-irsa-roles-test",
  "children": [
    {
      "target-role-name":"arn:aws-us-gov:iam::123456789012:role/role1",
      "members": ["integration-test"]
    },
    {
      "proxy-role-name":"arn:aws-us-gov:iam::123456789012:role/role4",
      "target-role-name":"arn:aws-us-gov:iam::210987654321:role/role1",
      "members": ["integration-test"]
    }
  ]
}

Group naming rule:
  - If 'proxy-role-name' exists:   "{proxy-role-name}->{target-role-name}"
  - Else:                          "{target-role-name}"

Behavior:
  - Ensures the root group exists (top-level group named spec['root']).
  - For each child entry:
      * Computes child group name using the rule above.
      * Ensures a direct child group exists under root with name == computed name.
      * Reconciles membership:
          - Adds users present in spec but missing in group.
          - Removes users present in group but NOT in spec.
  - Idempotent.

Auth (env):
  KC_URL (required)         e.g. https://<host>/auth
  REALM (required)          e.g. DominoRealm
  KC_ADMIN_USER (required)
  KC_ADMIN_PASS (required)
  ADMIN_REALM (optional)    default: master
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import requests


class KCError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise KCError(f"Missing required env var: {name}")
    return v


def rstrip_slash(u: str) -> str:
    return (u or "").rstrip("/")


@dataclass(frozen=True)
class KCConfig:
    base_url: str
    realm: str
    admin_realm: str
    admin_user: str
    admin_pass: str


def _headers_json(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def post_form(url: str, data: Dict[str, str], timeout: int = 30) -> requests.Response:
    return requests.post(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        timeout=timeout,
        verify=False  # Disables SSL certificate verification
    )


def mint_admin_token(cfg: KCConfig) -> str:
    token_url = f"{cfg.base_url}/realms/{cfg.admin_realm}/protocol/openid-connect/token"
    r = post_form(
        token_url,
        {
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": cfg.admin_user,
            "password": cfg.admin_pass,
        },
    )
    if r.status_code >= 400:
        raise KCError(f"{r.status_code} error minting admin token at {token_url}: {r.text[:800]}")
    payload = r.json()
    tok = payload.get("access_token")
    if not tok:
        raise KCError(f"Admin token response missing access_token: {payload}")
    return tok


def kc_get(cfg: KCConfig, token: str, path: str, params: Optional[Dict[str, str]] = None) -> Any:
    url = f"{cfg.base_url}{path}"
    r = requests.get(url, headers=_headers_json(token), params=params, timeout=30,verify=False)
    if r.status_code >= 400:
        raise KCError(f"{r.status_code} error GET {url}: {r.text[:800]}")
    return r.json()


def kc_post_json(cfg: KCConfig, token: str, path: str, payload: Dict[str, Any]) -> requests.Response:
    url = f"{cfg.base_url}{path}"
    return requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"},
        json=payload,
        timeout=30,
        verify=False
    )


def kc_put(cfg: KCConfig, token: str, path: str, payload: Optional[Dict[str, Any]] = None) -> requests.Response:
    url = f"{cfg.base_url}{path}"
    return requests.put(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"},
        json=payload,
        timeout=30,
        verify=False
    )


def kc_delete(cfg: KCConfig, token: str, path: str) -> requests.Response:
    url = f"{cfg.base_url}{path}"
    return requests.delete(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, timeout=30)


# -------------------------- Keycloak group APIs -----------------------------

def get_top_groups(cfg: KCConfig, token: str) -> List[Dict[str, Any]]:
    data = kc_get(cfg, token, f"/admin/realms/{cfg.realm}/groups")
    if not isinstance(data, list):
        raise KCError(f"Unexpected response for top groups: {type(data)}")
    return data


def get_children(cfg: KCConfig, token: str, gid: str) -> List[Dict[str, Any]]:
    data = kc_get(cfg, token, f"/admin/realms/{cfg.realm}/groups/{gid}/children")
    if not isinstance(data, list):
        raise KCError(f"Unexpected response for children of {gid}: {type(data)}")
    return data


def list_group_members(cfg: KCConfig, token: str, gid: str, max_results: int = 1000) -> List[Dict[str, Any]]:
    # Keycloak supports pagination via first/max; we keep it simple for typical group sizes.
    params = {"first": "0", "max": str(max_results)}
    data = kc_get(cfg, token, f"/admin/realms/{cfg.realm}/groups/{gid}/members", params=params)
    if not isinstance(data, list):
        raise KCError(f"Unexpected response for members of {gid}: {type(data)}")
    return data


def create_group(cfg: KCConfig, token: str, name: str, parent_id: str = "") -> str:
    if parent_id:
        path = f"/admin/realms/{cfg.realm}/groups/{parent_id}/children"
    else:
        path = f"/admin/realms/{cfg.realm}/groups"

    r = kc_post_json(cfg, token, path, {"name": name})
    if r.status_code not in (201, 204):
        raise KCError(f"{r.status_code} error creating group '{name}' at {cfg.base_url}{path}: {r.text[:800]}")
    loc = (r.headers.get("Location") or r.headers.get("location") or "").strip()
    if not loc:
        return ""
    return loc.rstrip("/").split("/")[-1]


def ensure_root_group(cfg: KCConfig, token: str, root_name: str, dry_run: bool) -> str:
    if not root_name or not isinstance(root_name, str):
        raise KCError("Spec 'root' must be a non-empty string")
    if "/" in root_name:
        raise KCError("Spec 'root' must be a single segment (no '/')")

    tops = get_top_groups(cfg, token)
    existing = next((g for g in tops if g.get("name") == root_name), None)
    if existing and existing.get("id"):
        return str(existing["id"])

    if dry_run:
        log(f"[dry-run] would create root group: {root_name}")
        return "__dry_run_root_id__"

    gid = create_group(cfg, token, root_name, parent_id="")
    if not gid:
        tops = get_top_groups(cfg, token)
        existing = next((g for g in tops if g.get("name") == root_name), None)
        gid = str((existing or {}).get("id") or "")
    if not gid:
        raise KCError(f"Created root group '{root_name}' but could not resolve its id")
    return gid


def ensure_child_group(cfg: KCConfig, token: str, root_id: str, child_name: str, dry_run: bool) -> str:
    if not child_name or not isinstance(child_name, str):
        raise KCError("Each child must resolve to a non-empty group name")

    kids = get_children(cfg, token, root_id) if root_id != "__dry_run_root_id__" else []
    existing = next((g for g in kids if g.get("name") == child_name), None)
    if existing and existing.get("id"):
        return str(existing["id"])

    if dry_run:
        log(f"[dry-run] would create child group under root: {child_name}")
        return "__dry_run_child_id__:" + child_name

    gid = create_group(cfg, token, child_name, parent_id=root_id)
    if not gid:
        kids = get_children(cfg, token, root_id)
        existing = next((g for g in kids if g.get("name") == child_name), None)
        gid = str((existing or {}).get("id") or "")
    if not gid:
        raise KCError(f"Created child group '{child_name}' but could not resolve its id")
    return gid


# -------------------------- Keycloak user APIs -------------------------------

def resolve_user_id(cfg: KCConfig, token: str, username: str) -> str:
    if not username:
        raise KCError("Username must be non-empty")
    params = {"username": username, "max": "20"}
    data = kc_get(cfg, token, f"/admin/realms/{cfg.realm}/users", params=params)
    if not isinstance(data, list):
        raise KCError(f"Unexpected response searching users: {type(data)}")

    exact = next((u for u in data if (u.get("username") or "") == username), None)
    if exact and exact.get("id"):
        return str(exact["id"])

    if len(data) == 1 and (data[0].get("id") and data[0].get("username")):
        return str(data[0]["id"])

    raise KCError(f"Could not uniquely resolve user '{username}'. Matches={len(data)}")


def add_user_to_group(cfg: KCConfig, token: str, user_id: str, group_id: str) -> None:
    path = f"/admin/realms/{cfg.realm}/users/{user_id}/groups/{group_id}"
    r = kc_put(cfg, token, path, payload=None)
    if r.status_code not in (200, 201, 204):
        raise KCError(f"{r.status_code} error adding user {user_id} to group {group_id}: {r.text[:800]}")


def remove_user_from_group(cfg: KCConfig, token: str, user_id: str, group_id: str) -> None:
    path = f"/admin/realms/{cfg.realm}/users/{user_id}/groups/{group_id}"
    r = kc_delete(cfg, token, path)
    if r.status_code not in (200, 204):
        raise KCError(f"{r.status_code} error removing user {user_id} from group {group_id}: {r.text[:800]}")


# ------------------------------ spec handling --------------------------------

def load_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def child_group_name(entry: Dict[str, Any]) -> str:
    target = entry.get("target-role-name")
    proxy = entry.get("proxy-role-name")

    if not target or not isinstance(target, str):
        raise KCError("Each child must have a non-empty string 'target-role-name'")

    if proxy is None or proxy == "":
        return target

    if not isinstance(proxy, str):
        raise KCError("If present, 'proxy-role-name' must be a string")
    return f"{proxy}->{target}"


def validate_spec(spec: Dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise KCError("Spec must be a JSON object")
    root = spec.get("root")
    if not root or not isinstance(root, str):
        raise KCError("Spec must contain top-level string field 'root'")
    children = spec.get("children")
    if children is None:
        raise KCError("Spec must contain top-level field 'children'")
    if not isinstance(children, list):
        raise KCError("Spec 'children' must be a list")

    for i, entry in enumerate(children):
        if not isinstance(entry, dict):
            raise KCError(f"Spec children[{i}] must be an object")
        _ = child_group_name(entry)
        members = entry.get("members", [])
        if members is None:
            members = []
        if not isinstance(members, list) or any((not isinstance(m, str) or not m) for m in members):
            raise KCError(f"Spec children[{i}] 'members' must be a list of non-empty strings")


# ------------------------------ apply logic ----------------------------------

def apply(cfg: KCConfig, token: str, spec: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    validate_spec(spec)

    root_name = spec["root"]
    root_id = ensure_root_group(cfg, token, root_name, dry_run=dry_run)

    ensured_groups: List[str] = []
    added_memberships: List[Dict[str, str]] = []
    removed_memberships: List[Dict[str, str]] = []

    for entry in spec.get("children", []):
        gname = child_group_name(entry)
        desired_members: Set[str] = set(entry.get("members") or [])

        gid = ensure_child_group(cfg, token, root_id, gname, dry_run=dry_run)
        ensured_groups.append(f"/{root_name}/{gname}")

        if dry_run:
            # In dry-run, we only show planned adds/removes (best-effort)
            log(f"[dry-run] would reconcile membership for group '{gname}' desired={sorted(desired_members)}")
            for username in sorted(desired_members):
                added_memberships.append({"username": username, "group": gname, "dry_run": "maybe"})
            continue

        # Current members in KC
        current = list_group_members(cfg, token, gid)
        current_by_username = { (m.get("username") or ""): m for m in (current or []) if m.get("username") }
        current_set = set(current_by_username.keys())

        to_add = sorted(desired_members - current_set)
        to_remove = sorted(current_set - desired_members)

        # Add missing
        for username in to_add:
            uid = resolve_user_id(cfg, token, username)
            add_user_to_group(cfg, token, uid, gid)
            added_memberships.append({"username": username, "group": gname})

        # Remove extra
        for username in to_remove:
            m = current_by_username.get(username) or {}
            uid = m.get("id") or resolve_user_id(cfg, token, username)
            remove_user_from_group(cfg, token, str(uid), gid)
            removed_memberships.append({"username": username, "group": gname})

    return {
        "root": root_name,
        "ensured_groups": ensured_groups,
        "added_memberships": added_memberships,
        "removed_memberships": removed_memberships,
        "dry_run": dry_run,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply Keycloak groups + reconcile memberships from a JSON spec.")
    ap.add_argument("--spec", required=True, help="Path to spec JSON file")
    ap.add_argument("--dry-run", action="store_true", help="Print planned actions; do not modify Keycloak.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    cfg = KCConfig(
        base_url=rstrip_slash(require_env("KC_URL")),
        realm=require_env("REALM"),
        admin_realm=os.getenv("ADMIN_REALM", "master"),
        admin_user=require_env("KC_ADMIN_USER"),
        admin_pass=require_env("KC_ADMIN_PASS"),
    )

    spec = load_spec(args.spec)

    log(f">> Minting admin token (realm={cfg.admin_realm}, client=admin-cli)…")
    token = mint_admin_token(cfg)

    result = apply(cfg, token, spec, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KCError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
