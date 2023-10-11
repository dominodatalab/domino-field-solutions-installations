
# We're assuming that the account with the Domino EKS cluster will be the default provider.

terraform {
  required_providers {
    aws = {
        source = "hashicorp/aws"
        version = "~> 5.0"
    }
  }
}

provider "aws" {
    region = "us-west-2"
    profile = "domino-eks"
}

provider "aws" {
    alias = "asset-acct"
    region = "us-west-2"
    profile = "asset-acct"
}