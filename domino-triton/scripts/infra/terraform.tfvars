# Copy this file to terraform.tfvars and fill in the values for your environment.

aws_region       = "us-west-2"
eks_cluster_name = "marcdo126967"

# Defaults match helm-install.md — only override if you changed the CSI driver install
# s3_csi_driver_namespace       = "domino-platform"
# s3_csi_driver_service_account = "s3-csi-driver-sa"

# Optional: override default resource names
# role_name   = "domino-triton-s3-role"
# policy_name = "domino-triton-s3-policy"

tags = {
  Project     = "domino-triton-inference"
  ManagedBy   = "terraform"
  deploy_id   = "marcdo126967"
  customer_name = "navy"
  infosec_classification = "sensitive"
}
