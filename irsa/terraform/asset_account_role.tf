
data "aws_iam_policy_document" "workload-role-trust-policy" {
    provider = aws.asset-acct
    statement {
      actions = ["sts:AssumeRole"]
      effect = "Allow"
      principals {
        type = "AWS"
        identifiers = [aws_iam_role.domino-irsa-proxy.arn]
      }
    }
}

resource "aws_iam_role" "irsa-workload-role" {
    provider = aws.asset-acct
    name = var.irsa-workload-role-name
    assume_role_policy = data.aws_iam_policy_document.workload-role-trust-policy.json
}

data "aws_iam_policy_document" "irsa-workload-example-policy" {
    provider = aws.asset-acct
    statement {
      actions = ["s3:ListBucket"]
      effect = "Allow"
      resources = ["*"]
    }
}

resource "aws_iam_policy" "irsa-workload-example-policy" {
    provider = aws.asset-acct
    name = "${var.irsa-workload-role-name}-policy"
    policy = data.aws_iam_policy_document.irsa-workload-example-policy.json
}

resource "aws_iam_role_policy_attachment" "irsa-workload-example" {
    provider = aws.asset-acct
    role = aws_iam_role.irsa-workload-role.name
    policy_arn = aws_iam_policy.irsa-workload-example-policy.arn
}