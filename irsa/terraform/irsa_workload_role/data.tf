data "aws_caller_identity" "current" {
}

data "aws_iam_policy" "policy-to-attach-to-role" {
    name = var.policy-to-attach
}