# Cloud Identity Management for Domino

## Demo Overview

---

## Slide 1: How IRSA Works

### EKS OIDC Infrastructure (Built-In)

Every EKS cluster comes equipped with an **OIDC Provider** that is tightly integrated with the Kubernetes API Server:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EKS Cluster Infrastructure                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Kube API Server                                 │   │
│  │                                                                      │   │
│  │   • Configured with --service-account-issuer (OIDC URL)             │   │
│  │   • Issues JWTs via Projected ServiceAccount Token Volume           │   │
│  │   • Tokens contain: iss, sub, aud claims                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      EKS OIDC Provider                               │   │
│  │                                                                      │   │
│  │   URL: oidc.eks.{region}.amazonaws.com/id/{cluster-id}              │   │
│  │   • Publishes /.well-known/openid-configuration                     │   │
│  │   • Publishes /keys (JWKS) for token verification                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    │  Registered as                         │
│                                    ▼  Identity Provider                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         AWS IAM                                      │   │
│  │                                                                      │   │
│  │   • Trusts JWTs signed by the EKS OIDC Provider                     │   │
│  │   • Validates tokens via sts:AssumeRoleWithWebIdentity              │   │
│  │   • Maps JWT claims (sub, aud) to IAM role trust policies           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Pod Identity Webhook (EKS Managed)

The `eks.amazonaws.com/role-arn` annotation on a ServiceAccount is a **hint** to the EKS Pod Identity Webhook:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Pod Identity Webhook Actions                             │
│                                                                             │
│  When a Pod references a ServiceAccount with role-arn annotation:           │
│                                                                             │
│  1. MOUNTS Projected ServiceAccount Token Volume                            │
│     └─ /var/run/secrets/eks.amazonaws.com/serviceaccount/token              │
│                                                                             │
│  2. SETS Environment Variables                                              │
│     └─ AWS_ROLE_ARN=arn:aws:iam::123456789:role/my-role                    │
│     └─ AWS_WEB_IDENTITY_TOKEN_FILE=/var/run/secrets/.../token              │
│                                                                             │
│  3. AWS SDKs automatically use these to call STS AssumeRoleWithWebIdentity │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Complete IRSA Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ┌─────────┐      ┌─────────────┐      ┌──────────────┐      ┌─────────┐  │
│   │  Pod    │      │ Projected   │      │   AWS STS    │      │  IAM    │  │
│   │         │─────▶│ SA Token    │─────▶│              │─────▶│  Role   │  │
│   │         │      │ (JWT)       │      │ AssumeRole   │      │         │  │
│   └─────────┘      └─────────────┘      │ WithWebId    │      └─────────┘  │
│        │                 │              └──────────────┘           │        │
│        │                 │                     │                   │        │
│        │           Contains:                   │             Trust Policy:  │
│        │           • iss: OIDC URL       Validates:        • sub claim     │
│        │           • sub: system:sa:     • Signature       • aud claim     │
│        │             ns:sa-name          • Claims                          │
│        │           • aud: sts.aws                                          │
│        │                                                                    │
│   Reads from:                                                               │
│   AWS_WEB_IDENTITY_TOKEN_FILE                                              │
│   AWS_ROLE_ARN                                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **EKS OIDC Provider** | Issues signed JWTs, registered with AWS IAM as trusted identity provider |
| **Kube API Server** | Configured to issue tokens with OIDC-compatible claims |
| **Projected SA Token** | Short-lived JWT mounted into pod, auto-refreshed by kubelet |
| **Pod Identity Webhook** | Mutates pods to mount token volume and set AWS env vars |
| **IAM Trust Policy** | Validates JWT `sub` claim matches allowed ServiceAccount |

### Trust Policy Structure

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::ACCOUNT:oidc-provider/oidc.eks.REGION.amazonaws.com/id/CLUSTER_ID"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "oidc.eks....:sub": "system:serviceaccount:NAMESPACE:SA_NAME"
      }
    }
  }]
}
```

### The Problem for Domino

**All this infrastructure is already in place** - EKS provides the OIDC provider, the webhook, and the token projection.

**But**: Standard IRSA binds roles to *ServiceAccount names*, and Domino dynamically generates SA names per execution.

**Our Goal**: Leverage the existing EKS IRSA machinery with **minimum Domino interventions** - just one mutation to swap the SA name.

---

## Slide 2: Keycloak-Based User-to-Role Mapping

### Why Keycloak?

**Challenge**: How do we map *Domino users* to *AWS IAM roles* when:
- ServiceAccounts are dynamically created per execution
- User identity must persist across sessions
- Configuration should be manageable without AWS console access

**Solution**: Use Keycloak groups as the source of truth for user-to-role mappings.

### The JWT Contains Everything

When a user authenticates to Domino, their JWT token contains:

```json
{
  "sub": "user-uuid",
  "preferred_username": "alice",
  "groups": [
    "/irsa-mappings/arn:aws:iam::123456789:role/data-scientist-role",
    "/irsa-mappings/arn:aws:iam::123456789:role/ml-engineer-role"
  ],
  "exp": 1234567890
}
```

**Key Insight**: The JWT obtained locally contains the user's group memberships, which encode their allowed IAM roles.

### Configuration Flow

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   Keycloak      │      │   Config File   │      │   IAM Trust     │
│                 │      │                 │      │   Policies      │
│  /irsa-mappings │─────▶│  groups.json    │─────▶│                 │
│    └─ role-arn  │      │                 │      │  sub: sa:alice  │
│       └─ alice  │      │  {              │      │  sub: sa:bob    │
│       └─ bob    │      │    "role": ..., │      │                 │
│                 │      │    "members": []│      │                 │
└─────────────────┘      │  }              │      └─────────────────┘
                         └─────────────────┘
```

### Why This Works Inside Domino Workspaces

1. **JWT Available Locally**: `DOMINO_API_PROXY` provides the user's access token
2. **No External Dependencies**: All config data derivable from JWT + Keycloak API
3. **Dry-Run Support**: Preview changes before applying
4. **IaC Compatible**: CLI + UI both support the same workflow

### IaC: Keycloak Group Specification (`kc_apply_group_spec.py`)

Define your user-to-role mappings as a JSON specification:

```json
{
  "root_group": "/irsa-mappings",
  "groups": [
    {
      "name": "arn:aws:iam::123456789:role/data-scientist-role",
      "members": ["alice", "bob"]
    },
    {
      "name": "arn:aws:iam::123456789:role/ml-engineer-role",
      "members": ["charlie", "alice"]
    }
  ]
}
```

Apply with dry-run support:

```bash
# Preview changes
python kc_apply_group_spec.py --spec groups.json --dry-run

# Apply changes
python kc_apply_group_spec.py --spec groups.json
```

**IaC Benefits**: Version-controlled, auditable, repeatable configuration.

### CLI Interface (`sync.py`)

```bash
# Sync users to K8s ServiceAccounts (dry-run first)
python sync.py users --default-aws-role-arn $ROLE --dry-run
python sync.py users --default-aws-role-arn $ROLE

# Sync Keycloak groups to IAM trust policies
python sync.py irsa --dry-run
python sync.py irsa
```

### Visual Interface: Streamlit App

Making IRSA a **native part of Domino** with a visual admin interface:

| Tab | Function |
|-----|----------|
| **User Mappings** | View Domino users → ServiceAccounts → IAM roles |
| **IRSA Groups** | View Keycloak group memberships and trust policy status |
| **Sync Operations** | Execute sync with dry-run preview |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Streamlit Admin UI                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Tab 1: User Mappings                                                │   │
│  │  ┌──────────────┬──────────────────┬────────────────────────────┐   │   │
│  │  │ Domino User  │ Service Account  │ Default AWS IAM Role       │   │   │
│  │  ├──────────────┼──────────────────┼────────────────────────────┤   │   │
│  │  │ alice        │ alice            │ arn:aws:iam::...:role/...  │   │   │
│  │  │ bob          │ bob              │ arn:aws:iam::...:role/...  │   │   │
│  │  └──────────────┴──────────────────┴────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Tab 2: IRSA Groups                                                  │   │
│  │  Group: arn:aws:iam::123456789:role/data-scientist-role             │   │
│  │  Members: alice, bob                                                 │   │
│  │  Trust Policy Subs: system:sa:domino-compute:alice, ...             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Both CLI and UI support `--dry-run`**: See exactly what will change before committing.

---

## Slide 3: The Mutation - Dynamic to Fixed ServiceAccount

### The Problem

Domino creates **dynamic ServiceAccounts** per execution:

```
run-12345-abcde  ← Changes every run
run-67890-fghij  ← Different SA name
```

IAM trust policies require **exact SA name matching**:

```json
"sub": "system:serviceaccount:domino-compute:run-12345-abcde"  ← Would need constant updates!
```

### The Solution: Username-Based ServiceAccounts

Create **persistent ServiceAccounts** named after Domino usernames:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: alice                    # Fixed, username-based
  namespace: domino-compute
  labels:
    domino/user-sa: "true"
  annotations:
    eks.amazonaws.com/role-arn: "arn:aws:iam::123456789:role/default-role"
```

### The Mutation Webhook

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Pod Creation Request                                │
│                                                                             │
│  ┌──────────────┐    ┌──────────────────────┐    ┌──────────────────────┐  │
│  │  Domino      │    │  Mutating Webhook    │    │  Final Pod Spec      │  │
│  │  Scheduler   │───▶│                      │───▶│                      │  │
│  │              │    │  1. Check labels     │    │  serviceAccountName: │  │
│  │  Pod with:   │    │  2. Get username     │    │    alice             │  │
│  │  sa: run-xxx │    │  3. Swap SA name     │    │                      │  │
│  └──────────────┘    └──────────────────────┘    └──────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Before & After

| Aspect | Before (Dynamic) | After (Username-Based) |
|--------|------------------|------------------------|
| **SA Name** | `run-12345-abcde` | `alice` |
| **Trust Policy** | Must update per run | Fixed: `sub: ...alice` |
| **IAM Role Binding** | Impossible to maintain | Stable, user-centric |
| **Audit Trail** | "Which user was run-12345?" | Clear: "alice" |

### Trust Policy Now Works

```json
{
  "Condition": {
    "StringLike": {
      "oidc.eks....:sub": "system:serviceaccount:*:alice"
    }
  }
}
```

This `sub` claim **never changes** for user `alice`, regardless of how many executions they run.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐         │
│    │  Keycloak   │         │  K8s API    │         │  AWS IAM    │         │
│    │  Groups     │◀───────▶│  (SAs)      │◀───────▶│  (Roles)    │         │
│    └─────────────┘         └─────────────┘         └─────────────┘         │
│           │                       │                       │                 │
│           │                       │                       │                 │
│           ▼                       ▼                       ▼                 │
│    ┌─────────────────────────────────────────────────────────────┐         │
│    │                    Admin Service (FastAPI)                   │         │
│    │                                                              │         │
│    │  POST /k8s/users/sync      - Create/update user SAs         │         │
│    │  POST /iam/irsa/sync       - Sync trust policies            │         │
│    │  GET  /keycloak/groups     - List group memberships         │         │
│    │  POST /keycloak/groups/member - Add/remove user from group  │         │
│    └─────────────────────────────────────────────────────────────┘         │
│                          │                 │                                │
│              ┌───────────┴───────┐   ┌─────┴─────┐                         │
│              ▼                   ▼   ▼           ▼                         │
│    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐        │
│    │  CLI (sync.py)  │   │  Streamlit UI   │   │ Mutation Webhook│        │
│    │  --dry-run      │   │  (app.py)       │   │ (swap SA name)  │        │
│    └─────────────────┘   └─────────────────┘   └─────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Admin Service: Security Model

### `domino-irsa-lite-admin` Deployment

The admin service is the key integration point that ties everything together:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     domino-irsa-lite-admin Deployment                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ServiceAccount: cloud-identity-management-utils                     │   │
│  │                                                                      │   │
│  │  Annotations:                                                        │   │
│  │    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/admin-role │   │
│  │                                                                      │   │
│  │  This role has permission to:                                        │   │
│  │    • iam:GetRole                                                     │   │
│  │    • iam:UpdateAssumeRolePolicy (for managed roles only)            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  The admin service itself uses IRSA to get AWS credentials!                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Safety Guardrails

The service is designed with **strict safety controls** to prevent unauthorized trust policy modifications:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Trust Policy Update Logic                          │
│                                                                             │
│  Environment Variables (configured at deployment):                          │
│    OIDC_PROVIDER_URL = "oidc.eks.us-west-2.amazonaws.com/id/ABC123"        │
│    OIDC_AUDIENCE     = "sts.amazonaws.com"                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Before updating any trust policy, the service VALIDATES:           │   │
│  │                                                                      │   │
│  │  1. OIDC Provider URL in existing policy MUST MATCH env variable    │   │
│  │     └─ Prevents modifying roles from other clusters                 │   │
│  │                                                                      │   │
│  │  2. Audience claim MUST MATCH configured OIDC_AUDIENCE              │   │
│  │     └─ Ensures only legitimate IRSA configurations are touched      │   │
│  │                                                                      │   │
│  │  3. ONLY the `sub` field is modified                                │   │
│  │     └─ Never changes Principal, Action, or other conditions         │   │
│  │     └─ Preserves existing security boundaries                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  If validation fails → Request is REJECTED, no changes made                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What the Service Will NOT Do

| Action | Allowed? | Reason |
|--------|----------|--------|
| Modify `sub` claim for matching OIDC provider | Yes | Core functionality |
| Modify roles with different OIDC provider URL | **NO** | Could affect other clusters |
| Change the OIDC provider in trust policy | **NO** | Could redirect trust |
| Change the audience claim | **NO** | Could weaken security |
| Add new Statement blocks | **NO** | Only modifies existing `sub` |
| Modify IAM policies (permissions) | **NO** | Only touches trust policies |

### Configuration Example

```yaml
# helm/values.yaml
env:
  OIDC_PROVIDER_URL: "oidc.eks.us-west-2.amazonaws.com/id/EXAMPLED539D4633E53DE1B716D3041E"
  OIDC_PROVIDER_ARN: "arn:aws:iam::123456789012:oidc-provider/oidc.eks.us-west-2.amazonaws.com/id/EXAMPLED539D4633E53DE1B716D3041E"
  OIDC_AUDIENCE: "sts.amazonaws.com"
```

The service **only manages trust policies for IAM roles that already have a valid IRSA trust policy for THIS cluster**.

---

## Key Takeaways

1. **IRSA binds IAM roles to ServiceAccount names** - but Domino uses dynamic SA names

2. **Keycloak groups as source of truth** - User-to-role mappings stored in Keycloak, accessible via JWT

3. **Username-based ServiceAccounts** - Mutation webhook swaps dynamic SAs for persistent user SAs

4. **IaC from inside Domino** - `kc_apply_group_spec.py` + CLI + UI with dry-run support enables self-service identity management

5. **All config derives from JWT** - No external credentials needed inside workspace

6. **Admin service uses IRSA itself** - Bootstrapped with its own IRSA role to manage trust policies

7. **Strict safety controls** - Only modifies `sub` claim when OIDC provider and audience match configuration
