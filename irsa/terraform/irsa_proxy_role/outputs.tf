
output "proxy-role-arn" {
    value = aws_iam_role.domino-irsa-proxy.arn
}

output "proxy-policy-arn" {
    value = aws_iam_policy.domino-irsa-proxy-policy.arn
}