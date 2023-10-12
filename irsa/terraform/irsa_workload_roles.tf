

module "irsa_workload_role1" {
    source = "./irsa_workload_role"
    associated-proxy-role-list = module.irsa_proxy_role1[*].proxy-role-arn
    irsa-workload-role-name = "${var.resource-prefix}-list-bucket-role"
    policy-to-attach = "acme-list-bucket-policy"
    #policy-to-attach = aws_iam_policy.irsa-workload-example-policy.name
}

module "irsa_workload_role2" {
    source = "./irsa_workload_role"
    associated-proxy-role-list = module.irsa_proxy_role2[*].proxy-role-arn
    irsa-workload-role-name = "${var.resource-prefix}-read-bucket-role"
    policy-to-attach = "acme-read-bucket-policy"
    #policy-to-attach = aws_iam_policy.irsa-workload-example-policy.name
}


module "irsa_workload_role3" {
    source = "./irsa_workload_role"
    associated-proxy-role-list = module.irsa_proxy_role2[*].proxy-role-arn
    irsa-workload-role-name = "${var.resource-prefix}-write-bucket-role"
    policy-to-attach = "acme-read-bucket-policy"
    #policy-to-attach = aws_iam_policy.irsa-workload-example-policy.name
}
