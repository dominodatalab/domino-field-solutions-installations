
output "arn" {
    value = aws_iam_role.irsa-workload-role.arn
}

output "name" {
    value = aws_iam_role.irsa-workload-role.name
}