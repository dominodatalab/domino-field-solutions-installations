
# These policies are included for demo purposes only. Please comment/remove the below code in a real env.

data "aws_iam_policy_document" "irsa-workload-example-policy" {
    provider = aws.asset-acct
    statement {
      actions = ["s3:ListBucket"]
      effect = "Allow"
      resources = ["*"]
    }
}

#resource "aws_iam_policy" "irsa-workload-example-policy" {
#    provider = aws.asset-acct
#    name = "${local.wl-role-name}-policy"
#    policy = data.aws_iam_policy_document.irsa-workload-example-policy.json
#}