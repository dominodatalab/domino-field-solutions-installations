
# These policies are included for demo purposes only. Please comment/remove the below code in a real env.

data "aws_iam_policy_document" "workload-1-policy" {
    provider = aws.asset-acct
    statement {
      actions = ["s3:ListBucket"]
      effect = "Allow"
      resources = ["*"]
    }
}

data "aws_iam_policy_document" "workload-2-policy" {
    provider = aws.asset-acct
    statement {
      actions = ["s3:ListBucket"]
      effect = "Allow"
      resources = ["*"]
    }
}

data "aws_iam_policy_document" "workload-3-policy" {
    provider = aws.asset-acct
    statement {
      actions = ["s3:ListBucket"]
      effect = "Allow"
      resources = ["*"]
    }
}

data "aws_iam_policy_document" "workload-4-policy" {
    provider = aws.asset-acct
    statement {
      actions = ["s3:ListBucket"]
      effect = "Allow"
      resources = ["*"]
    }
}

resource "aws_iam_policy" "irsa-workload-1-policy" {
    provider = aws.asset-acct
    name = "${var.resource-prefix}-project1-policy"
    policy = data.aws_iam_policy_document.workload-1-policy.json
}

resource "aws_iam_role_policy_attachment" "irsa-workload-1" {
    provider = aws.asset-acct
    role = module.irsa_workload_role1.name
    policy_arn = aws_iam_policy.irsa-workload-1-policy.arn
}

resource "aws_iam_policy" "irsa-workload-2-policy" {
    provider = aws.asset-acct
    name = "${var.resource-prefix}-project2-policy"
    policy = data.aws_iam_policy_document.workload-2-policy.json
}

resource "aws_iam_role_policy_attachment" "irsa-workload-2" {
    provider = aws.asset-acct
    role = module.irsa_workload_role2.name
    policy_arn = aws_iam_policy.irsa-workload-2-policy.arn
}

resource "aws_iam_policy" "irsa-workload-3-policy" {
    provider = aws.asset-acct
    name = "${var.resource-prefix}-project3-policy"
    policy = data.aws_iam_policy_document.workload-3-policy.json
}

resource "aws_iam_role_policy_attachment" "irsa-workload-3" {
    provider = aws.asset-acct
    role = module.irsa_workload_role3.name
    policy_arn = aws_iam_policy.irsa-workload-3-policy.arn
}

resource "aws_iam_policy" "irsa-workload-4-policy" {
    provider = aws.asset-acct
    name = "${var.resource-prefix}-project4-policy"
    policy = data.aws_iam_policy_document.workload-4-policy.json
}

resource "aws_iam_role_policy_attachment" "irsa-workload-4" {
    provider = aws.asset-acct
    role = module.irsa_workload_role4.name
    policy_arn = aws_iam_policy.irsa-workload-4-policy.arn
}