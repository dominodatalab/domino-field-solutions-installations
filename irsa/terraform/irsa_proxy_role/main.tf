
data "aws_iam_policy_document" "irsa-proxy-role-trust-policy" {
    statement {
      actions = ["sts:AssumeRoleWithWebIdentity"]
      effect = "Allow"
      principals {
        identifiers = [data.aws_iam_openid_connect_provider.domino-cluster-provider.arn]
        type = "Federated"
      }
      condition {
        test = "StringLike"
        variable = "${replace(data.aws_eks_cluster.domino-cluster.identity[0].oidc[0].issuer,"https://","")}:sub"
        values = [""]
      }    
    }
}

resource "aws_iam_role" "domino-irsa-proxy" {
    name = "${var.irsa-proxy-role-name}"
    assume_role_policy = data.aws_iam_policy_document.irsa-proxy-role-trust-policy.json
}

data "aws_iam_policy_document" "domino-irsa-proxy-policy" {
    statement {
      actions = ["sts:AssumeRole"]
      resources = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.irsa-workload-role-path}${var.irsa-workload-role-name}"]
    }
}

resource "aws_iam_policy" "domino-irsa-proxy-policy" {
    name = "${var.irsa-proxy-role-name}-policy"
    policy = data.aws_iam_policy_document.domino-irsa-proxy-policy.json
}

resource "aws_iam_role_policy_attachment" "domino-irsa-proxy" {
    role = aws_iam_role.domino-irsa-proxy.name
    policy_arn = aws_iam_policy.domino-irsa-proxy-policy.arn
}