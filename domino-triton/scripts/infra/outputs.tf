output "bucket_name" {
  description = "Name of the S3 bucket for Triton model storage"
  value       = aws_s3_bucket.triton_models.id
}

output "bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.triton_models.arn
}

output "iam_role_arn" {
  description = "ARN of the IRSA role — annotate the S3 CSI driver service account with this"
  value       = aws_iam_role.triton_s3.arn
}

output "iam_role_name" {
  description = "Name of the IRSA IAM role"
  value       = aws_iam_role.triton_s3.name
}

output "iam_policy_arn" {
  description = "ARN of the S3 access IAM policy"
  value       = aws_iam_policy.triton_s3.arn
}
