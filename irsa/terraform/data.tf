
data "aws_eks_cluster" "domino-cluster" {
    name = var.eks-cluster-name
}

data "aws_iam_openid_connect_provider" "domino-cluster-provider" {
    url = data.aws_eks_cluster.domino-cluster.identity[0].oidc[0].issuer
}

data "aws_caller_identity" "domino-eks-acct" {
}

data "aws_caller_identity" "aws-asset-acct" {
    provider = aws.asset-acct
}