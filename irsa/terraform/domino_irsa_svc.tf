
data "aws_iam_policy_document" "domino-eks-irsa-role-trust-policy" {
    statement {
      actions = ["sts:AssumeRoleWithWebIdentity"]
      effect = "Allow"
      principals {
        identifiers = [data.aws_iam_openid_connect_provider.domino-cluster-provider.arn]
        type = "Federated"
      }
      condition {
        test = "StringEquals"
        variable = "${replace(data.aws_eks_cluster.domino-cluster.identity[0].oidc[0].issuer,"https://","")}:sub"
        values = ["system:serviceaccount:${var.domino-irsa-namespace}:irsa"]
      }  
    }
}

resource "aws_iam_role" "domino-irsa-svc" {
    name = var.irsa-svc-role-name
    assume_role_policy = data.aws_iam_policy_document.domino-eks-irsa-role-trust-policy.json
}

data "aws_iam_policy_document" "domino-irsa-svc-policy" {
    statement {
      sid = "IRSAAdmin"
      actions = [
        "iam:ListPolicies",
        "iam:ListPolicyVersions",
        "iam:ListRolePolicies",
        "iam:ListRoles",
        "iam:GetRole",
        "iam:GetRole",
        "iam:PutRolePolicy",
        "iam:UpdateAssumeRolePolicy"        
      ]
      effect = "Allow"
      resources = length(var.irsa-proxy-role-list) > 0 ? var.irsa-proxy-role-list : ["arn:aws:iam::${data.aws_caller_identity.domino-eks-acct.account_id}:role/*"]
    }
}

resource "aws_iam_policy" "domino-irsa-svc-policy" {
  name = "${var.irsa-svc-role-name}-policy"
  policy = data.aws_iam_policy_document.domino-irsa-svc-policy.json
}

resource "aws_iam_role_policy_attachment" "domino-irsa-svc" {
  role = aws_iam_role.domino-irsa-svc.name
  policy_arn = aws_iam_policy.domino-irsa-svc-policy.arn
}