import os
import requests
import pandas as pd
import streamlit as st
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# 1. Page Configuration
st.set_page_config(page_title="IRSA Admin", layout="wide")
st.title("Domino Users -> IRSA IAM Role Mapping Tool")


# --- INITIALIZE SESSION STATE ---
@dataclass(frozen=True)
class Settings:
    domino_host: str
    irsa_mgmt_host: str
    http_verify_tls: bool


def load_settings() -> Settings:
    # Use getenv with defaults to prevent crashes
    domino_host = os.getenv("DOMINO_API_HOST", "http://nucleus-frontend.domino-platform:80").strip().rstrip("/")
    irsa_mgmt_host = os.getenv("DOMINO_IRSA_MGMT_HOST",
                               "http://domino-irsa-lite-admin.domino-field.svc.cluster.local:8000").strip().rstrip("/")

    http_verify_tls = os.getenv("HTTP_VERIFY_TLS", "false").strip().lower() in ("1", "true", "yes")

    return Settings(
        domino_host=domino_host,
        irsa_mgmt_host=irsa_mgmt_host,
        http_verify_tls=http_verify_tls,
    )


settings = load_settings()


def get_domino_token():
    # Use getenv for safety
    proxy = os.getenv('DOMINO_API_PROXY')
    if not proxy:
        return ""  # Handle gracefully or raise specific error
    return requests.get(f"{proxy}/access-token").text


def get_request_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    # FIX: Use os.getenv to avoid KeyError if variable is missing
    api_key = os.getenv('DOMINO_API_KEY_OVERRIDE')

    if api_key:
        headers["X-Domino-Api-Key"] = api_key
    else:
        token = get_domino_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


# --- HELPER FUNCTIONS ---

def run_irsa_sync():
    """
    Helper function to trigger the IRSA Sync.
    Used by Manual Button and Auto-Trigger on Add/Remove.
    """
    sync_url = f"{settings.irsa_mgmt_host}/iam/irsa/sync"
    payload = {"dry_run": False}

    # We use a toast here because it persists slightly better across the immediate rerun
    try:
        response = requests.post(sync_url, headers=get_request_headers(), json=payload, verify=settings.http_verify_tls)
        if response.status_code == 200:
            st.toast("✅ IRSA Sync initiated successfully!", icon="🔄")
        else:
            st.error(f"IRSA Sync failed. Status: {response.status_code}, Detail: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"IRSA Sync Connection error: {e}")


def fetch_domino_user_mappings() -> pd.DataFrame:
    """
    Fetches users and service accounts, returns a DataFrame ready for display.
    Does NOT interact with st.session_state directly.
    """
    user_url = f"{settings.domino_host}/v4/users"
    irsa_mgmt_url = f"{settings.irsa_mgmt_host}/k8s/serviceaccounts"
    headers = get_request_headers()

    # 1. Fetch Domino Users
    try:
        resp = requests.get(user_url, headers=headers, timeout=15.0, verify=settings.http_verify_tls)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch users: {e}")

    users_payload = resp.json()
    # Create a set of valid usernames for O(1) lookups if needed,
    # though here we just need the list of names.
    domino_users = [
        u.get("userName").strip()
        for u in users_payload
        if isinstance(u, dict) and u.get("userName")
    ]

    # 2. Fetch IRSA Mappings
    try:
        resp = requests.get(irsa_mgmt_url, headers=headers, timeout=15.0, verify=settings.http_verify_tls)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch service accounts: {e}")

    k8s_items = resp.json().get("items", [])

    # Map SA name -> ServiceAccount object (SA name equals username)
    sa_map = {sa.get("name", ""): sa for sa in k8s_items if sa.get("name")}

    # 3. Build the Data Rows
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
            "Domino User": user,
            "Service Account": sa_name,
            "Default AWS IAM Role": role_arn
        })

    return pd.DataFrame(rows)


def fetch_keycloak_data() -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Fetches Keycloak groups and correlates them with IAM Trust Policy (IRSA) data.
    Returns: (groups_dataframe, members_list_of_dicts)
    """
    groups_url = f"{settings.irsa_mgmt_host}/keycloak/groups"
    trust_url = f"{settings.irsa_mgmt_host}/iam/irsa/trust"
    headers = get_request_headers()

    # 1. Fetch Groups
    try:
        resp_groups = requests.get(groups_url, headers=headers, timeout=15.0, verify=settings.http_verify_tls)
        resp_groups.raise_for_status()
        groups_data = resp_groups.json()
    except requests.RequestException as e:
        st.error(f"Groups API Error: {e}")
        groups_data = {}

    # 2. Fetch Trust Data (Subs attached to IAM roles)
    trust_data = {"items": []}
    try:
        resp_trust = requests.get(trust_url, headers=headers, timeout=15.0, verify=settings.http_verify_tls)
        resp_trust.raise_for_status()
        trust_data = resp_trust.json()
    except requests.RequestException as e:
        st.warning(f"Trust API Error: {e}")

    # 3. Create Map of Trusted Subs: {group_name -> [list_of_subs]}
    # The API returns {"items": [{"group": "arn:...", "role": "...", "subs": [...]}]}
    trust_map = {item.get('group', ''): item.get('subs', []) for item in trust_data.get('items', [])}
    groups_raw = groups_data.get("groups", [])

    group_rows = []
    all_members = []

    for g in groups_raw:
        raw_name = g.get("name", "")
        g_id = g.get("id", "")
        display_name = raw_name

        # Parse ARNs
        if "->" in display_name:
            parts = display_name.split("->")
            proxy_arn = parts[0].strip()
            target_arn = parts[1].strip()
        else:
            proxy_arn = ""
            target_arn = display_name

        # Lookup Trusted Subs from the Trust API correlation (keyed by group name)
        trusted_subs = trust_map.get(raw_name, [])

        group_rows.append({
            "Group Name": display_name,
            "Group Id": g_id,
            "Target Role ARN": target_arn,
            "Proxy Role ARN": proxy_arn,
            "Trusted Subs": trusted_subs
        })

        # Process Keycloak Members
        for u_name in g.get("members", []):

            if u_name:
                all_members.append({
                    "group_id": g_id,
                    "domino_user_name": u_name,
                    "sub": f"system:serviceaccount:domino-compute:{u_name.replace('.', '-')}"
                })

    return pd.DataFrame(group_rows), all_members


# --- SESSION STATE MANAGEMENT ---
def refresh_data():
    with st.spinner("Fetching latest data..."):
        st.session_state['domino_mappings'] = fetch_domino_user_mappings()
        kc_groups_df, kc_members_list = fetch_keycloak_data()
        st.session_state['kc_groups'] = kc_groups_df
        st.session_state['kc_members'] = kc_members_list


if 'domino_mappings' not in st.session_state:
    refresh_data()

# --- SESSION STATE SETUP ---
if 'kc_groups' not in st.session_state:
    st.session_state['kc_groups'] = pd.DataFrame(
        columns=['Group Id', 'Group Name', 'Target Role ARN', 'Proxy Role ARN', 'Trusted Subs'])

if 'kc_members' not in st.session_state:
    st.session_state['kc_members'] = []

# 2. Tabs
tab1, tab2 = st.tabs(["Domino User SA Mappings", "Keycloak Group Management"])

# --- TAB 1: DOMINO MAPPINGS ---
with tab1:
    col_header, col_btn = st.columns([4, 1])
    with col_header:
        st.header("Domino Users -> Service Account Mappings")
    with col_btn:
        # Add a manual refresh button
        if st.button("🔄 Refresh Data"):
            refresh_data()
            st.rerun()

    st.subheader("Mappings")

    # Access the DataFrame directly from state
    df_mappings = st.session_state.get('domino_mappings', pd.DataFrame())

    st.dataframe(df_mappings, use_container_width=True)
    st.write("---")

    # Determine default ARN from the data (safely)
    default_arn = ""
    if not df_mappings.empty:
        # Get the first non-empty ARN found in the table
        valid_arns = df_mappings[df_mappings["Default AWS IAM Role"] != ""]["Default AWS IAM Role"]
        if not valid_arns.empty:
            default_arn = valid_arns.iloc[0]

    st.subheader("Configure Default IAM Role for Sync")
    aws_arn = st.text_input(
        "Default IAM Role ARN",
        value=default_arn,
        placeholder="arn:aws:iam::123456789012:role/YourRoleName",
        help="This IAM Role ARN will be applied to all Domino users during sync."
    )

    if st.button("Sync Users", key="btn_tab1"):
        if not aws_arn:
            st.warning("⚠️ Please enter a Target IAM Role ARN before syncing.")
        else:
            api_url = f"{settings.irsa_mgmt_host}/k8s/users/sync"
            payload = {"default_aws_role_arn": aws_arn, "dry_run": False}

            with st.spinner("Syncing user mappings..."):
                try:
                    response = requests.post(api_url, headers=get_request_headers(), json=payload, verify=settings.http_verify_tls)
                    if response.status_code == 200:
                        st.success("Sync Complete!")
                        refresh_data()
                        st.rerun()
                    else:
                        st.error(f"Sync failed: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

# --- TAB 2: KEYCLOAK GROUP MANAGEMENT ---
with tab2:
    st.header("Keycloak Group Management")

    groups_df = st.session_state.get('kc_groups', pd.DataFrame())

    if groups_df.empty:
        st.warning("No Keycloak groups found.")
    else:
        # 1. Group Selector
        group_names = groups_df['Group Name'].tolist()
        col_sel, col_info = st.columns([1, 2])
        with col_sel:
            selected_group_name = st.selectbox("Select Keycloak Group", group_names)

        # Retrieve details
        group_row = groups_df[groups_df['Group Name'] == selected_group_name].iloc[0]
        selected_group_id = group_row['Group Id']
        target_arn = group_row['Target Role ARN']
        proxy_arn = group_row['Proxy Role ARN']
        trusted_subs = group_row['Trusted Subs']  # List of strings from the Trust endpoint

        with col_info:
            st.markdown(f"**Group Id:** `{selected_group_id}`")
            st.markdown(f"**Target Role ARN:** `{target_arn}`")
            if proxy_arn:
                st.markdown(f"**Proxy Role ARN:** `{proxy_arn}`")

        st.divider()

        # 2. Members Table
        display_header_name = proxy_arn if (proxy_arn and proxy_arn.strip()) else target_arn
        st.subheader(f"Members of '{display_header_name}'")

        all_members = st.session_state.get('kc_members', [])
        current_group_members = [m for m in all_members if m['group_id'] == selected_group_id]
        current_member_names = [m['domino_user_name'] for m in current_group_members]

        # Calculate available users from Tab 1 data
        df_mappings_src = st.session_state.get('domino_mappings', pd.DataFrame())

        # Get list of ALL valid users from the mappings dataframe
        if not df_mappings_src.empty and 'Domino User' in df_mappings_src.columns:
            all_domino_users = df_mappings_src['Domino User'].tolist()
        else:
            all_domino_users = []

        # Filter: All users from Tab 1 minus those already in the group
        available_users = sorted([u for u in all_domino_users if u not in current_member_names])

        # Table Header
        h1, h2, h3 = st.columns([2, 4, 1])
        h1.markdown("**Domino User**")
        h2.markdown("**Service Account Subject (sub)**")
        h3.markdown("**Action**")
        st.write("---")

        # Existing Members
        if not current_group_members:
            st.info("No Keycloak members in this group.")

        for member in current_group_members:
            r1, r2, r3 = st.columns([2, 4, 1])
            r1.write(member['domino_user_name'])
            r2.code(member['sub'], language="text")

            # Remove Button
            btn_key = f"del_{selected_group_id}_{member['domino_user_name']}"
            if r3.button("Remove", key=btn_key):
                api_url = f"{settings.irsa_mgmt_host}/keycloak/groups/member"
                payload = {
                    "username": member['domino_user_name'],
                    "group_name": selected_group_name
                }

                with st.spinner(f"Removing {member['domino_user_name']}..."):
                    try:
                        response = requests.delete(api_url, headers=get_request_headers(), json=payload, verify=settings.http_verify_tls)
                        if response.status_code == 200:
                            st.success(f"Removed {member['domino_user_name']}")

                            # --- AUTO SYNC ON REMOVE ---
                            with st.spinner("Auto-syncing IRSA..."):
                                run_irsa_sync()
                            # ---------------------------

                            refresh_data()
                            st.rerun()
                        else:
                            st.error(f"Failed to remove. Status: {response.status_code}, Detail: {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"API Request Error: {e}")

        # Add Member Row
        st.write("---")
        if available_users:
            a1, a2, a3 = st.columns([2, 4, 1])
            with a1:
                user_to_add = st.selectbox("Select User", available_users, key="new_user_select",
                                           label_visibility="collapsed")
            with a2:
                st.write("")
            with a3:
                # Add Button
                if st.button("Add", key="add_btn_row"):
                    api_url = f"{settings.irsa_mgmt_host}/keycloak/groups/member"
                    payload = {
                        "username": user_to_add,
                        "group_name": selected_group_name
                    }

                    with st.spinner(f"Adding {user_to_add} to group..."):
                        try:
                            response = requests.post(api_url, headers=get_request_headers(), json=payload, verify=settings.http_verify_tls)

                            if response.status_code == 201:
                                st.success(f"Added {user_to_add} successfully")

                                # --- AUTO SYNC ON ADD ---
                                with st.spinner("Auto-syncing IRSA..."):
                                    run_irsa_sync()
                                # ------------------------

                                refresh_data()
                                st.rerun()
                            else:
                                st.error(f"Failed to add. Status: {response.status_code}, Detail: {response.text}")
                        except requests.exceptions.RequestException as e:
                            st.error(f"API Request Error: {e}")
        else:
            st.caption("All available users are already members.")

    st.write("---")

    # --- ACTION BUTTONS (View Config & Sync) ---

    # 1. Prepare JSON Data
    children_list = []
    if not groups_df.empty:
        all_mbrs = st.session_state.get('kc_members', [])
        for idx, row in groups_df.iterrows():
            g_id = row['Group Id']
            t_arn = row['Target Role ARN']
            p_arn = row['Proxy Role ARN']

            # Find members for this specific group
            group_members_usernames = [
                m['domino_user_name']
                for m in all_mbrs
                if m['group_id'] == g_id
            ]

            # Build the group object
            group_obj = {
                "target-role-name": t_arn,
                "members": group_members_usernames
            }

            # Add proxy role only if it exists
            if p_arn and str(p_arn).strip():
                group_obj["proxy-role-name"] = p_arn

            children_list.append(group_obj)

    final_config = {
        "root": "domino-irsa-roles",
        "children": children_list
    }

    # 2. Display Buttons Side-by-Side
    col_view, col_sync = st.columns([1, 1])

    with col_view:
        # Initialize toggle state if not present
        if 'view_config_open' not in st.session_state:
            st.session_state['view_config_open'] = False

        # Determine button label based on state
        btn_label = "Hide Config" if st.session_state['view_config_open'] else "View Config"

        # The Toggle Button
        if st.button(btn_label, key="btn_toggle_config"):
            st.session_state['view_config_open'] = not st.session_state['view_config_open']
            st.rerun()

    with col_sync:
        if st.button("Sync IRSA", key="btn_sync_irsa"):
            with st.spinner("Syncing IRSA policies..."):
                run_irsa_sync()

    # 3. Conditionally Display Config Content
    if st.session_state['view_config_open']:
        st.info("Configuration Preview:")
        # Showing in a text_area as requested
        st.text_area(
            "JSON Configuration",
            value=json.dumps(final_config, indent=2),
            height=400,
            disabled=True  # Make it read-only
        )