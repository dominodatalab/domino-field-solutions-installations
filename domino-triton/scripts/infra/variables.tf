variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "eks_cluster_name" {
  description = "Name of the EKS cluster (used as a prefix for resource names)"
  type        = string
}

variable "s3_csi_driver_namespace" {
  description = "Kubernetes namespace where the S3 CSI driver service account lives"
  type        = string
  default     = "domino-platform"
}

variable "s3_csi_driver_service_account" {
  description = "Kubernetes service account name for the S3 CSI driver"
  type        = string
  default     = "s3-csi-driver-sa"
}

variable "role_name" {
  description = "Override the default IRSA IAM role name (default: <cluster>-triton-s3-role)"
  type        = string
  default     = ""
}

variable "policy_name" {
  description = "Override the default S3 IAM policy name (default: <cluster>-triton-s3-policy)"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags applied to all managed resources"
  type        = map(string)
  default     = {}
}
