locals {
  bucket_name = "${var.eks_cluster_name}-triton-models"
  role_name   = var.role_name != "" ? var.role_name : "${var.eks_cluster_name}-triton-s3-role"
  policy_name = var.policy_name != "" ? var.policy_name : "${var.eks_cluster_name}-triton-s3-policy"

  # EKS OIDC issuer without the https:// prefix (used by IAM condition keys and the OIDC provider lookup)
  oidc_id = trimprefix(data.aws_eks_cluster.this.identity[0].oidc[0].issuer, "https://")
  oidc_url = data.aws_eks_cluster.this.identity[0].oidc[0].issuer
}

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

data "aws_eks_cluster" "this" {
  name = var.eks_cluster_name
}

data "aws_iam_openid_connect_provider" "eks" {
  url = local.oidc_url
}

data "aws_iam_policy_document" "triton_s3" {
  statement {
    actions = [
      "s3:ListBucket",
      "s3:GetBucketLocation",
    ]
    resources = [aws_s3_bucket.triton_models.arn]
  }

  statement {
    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:DeleteObject",
    ]
    resources = ["${aws_s3_bucket.triton_models.arn}/*"]
  }
}

data "aws_iam_policy_document" "triton_s3_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.eks.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_id}:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_id}:sub"
      values   = ["system:serviceaccount:${var.s3_csi_driver_namespace}:${var.s3_csi_driver_service_account}"]
    }
  }
}

# ---------------------------------------------------------------------------
# S3 bucket for Triton model storage
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "triton_models" {
  bucket = local.bucket_name
  tags   = var.tags
}

resource "aws_s3_bucket_public_access_block" "triton_models" {
  bucket                  = aws_s3_bucket.triton_models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "triton_models" {
  bucket = aws_s3_bucket.triton_models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "triton_models" {
  bucket = aws_s3_bucket.triton_models.id

  versioning_configuration {
    status = "Enabled"
  }
}

# ---------------------------------------------------------------------------
# IRSA: IAM role + policy for the S3 CSI driver
# ---------------------------------------------------------------------------

resource "aws_iam_policy" "triton_s3" {
  name        = local.policy_name
  description = "Grants the S3 CSI driver read/write access to the Triton model bucket"
  policy      = data.aws_iam_policy_document.triton_s3.json
  tags        = var.tags
}

resource "aws_iam_role" "triton_s3" {
  name               = local.role_name
  description        = "IRSA role for the S3 CSI driver (domino-triton model storage)"
  assume_role_policy = data.aws_iam_policy_document.triton_s3_trust.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "triton_s3" {
  role       = aws_iam_role.triton_s3.name
  policy_arn = aws_iam_policy.triton_s3.arn
}
