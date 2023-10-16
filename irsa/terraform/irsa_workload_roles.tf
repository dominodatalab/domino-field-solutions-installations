

module "irsa_workload_role1" {
    source = "./irsa_workload_role"
    associated-proxy-role-list = module.irsa_proxy_role1[*].proxy-role-arn
    irsa-workload-role-name = "${var.resource-prefix}-project1-role"
    policy-to-attach = "acme-list-bucket-policy"
    #policy-to-attach = aws_iam_policy.irsa-workload-2-policy.name
}

module "irsa_workload_role2" {
    source = "./irsa_workload_role"
    associated-proxy-role-list = module.irsa_proxy_role2[*].proxy-role-arn
    irsa-workload-role-name = "${var.resource-prefix}-project2-role"
    policy-to-attach = "acme-read-bucket-policy"
    #policy-to-attach = aws_iam_policy.irsa-workload-3-policy.name
}


module "irsa_workload_role3" {
    source = "./irsa_workload_role"
    associated-proxy-role-list = module.irsa_proxy_role2[*].proxy-role-arn
    irsa-workload-role-name = "${var.resource-prefix}-project3-role"
    policy-to-attach = "acme-read-bucket-policy"
    #policy-to-attach = aws_iam_policy.irsa-workload-3-policy.name
}


module "irsa_workload_role4" {
    source = "./irsa_workload_role"
    associated-proxy-role-list = module.irsa_proxy_role2[*].proxy-role-arn
    irsa-workload-role-name = "${var.resource-prefix}-project4-role"
    policy-to-attach = "acme-read-bucket-policy"
    #policy-to-attach = aws_iam_policy.irsa-workload-4-policy.name
}