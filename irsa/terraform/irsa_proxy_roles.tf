

module "irsa_proxy_role1" {
    source = "./irsa_proxy_role"
    count = var.proxy-role-count
    irsa-proxy-role-name = "${var.resource-prefix}-proxy-project1-role-${count.index+1}"
    irsa-workload-role-name = "${var.resource-prefix}-project1-role"
    eks-cluster-name = var.eks-cluster-name
}

module "irsa_proxy_role2" {
    source = "./irsa_proxy_role"
    count = var.proxy-role-count
    irsa-proxy-role-name = "${var.resource-prefix}-proxy-project2-role-${count.index+1}"
    irsa-workload-role-name = "${var.resource-prefix}-project2-role"
    eks-cluster-name = var.eks-cluster-name
}

module "irsa_proxy_role3" {
    source = "./irsa_proxy_role"
    count = var.proxy-role-count
    irsa-proxy-role-name = "${var.resource-prefix}-proxy-project3-role-${count.index+1}"
    irsa-workload-role-name = "${var.resource-prefix}-project3-role"
    eks-cluster-name = var.eks-cluster-name
}

module "irsa_proxy_role4" {
    source = "./irsa_proxy_role"
    count = var.proxy-role-count
    irsa-proxy-role-name = "${var.resource-prefix}-proxy-project4-role-${count.index+1}"
    irsa-workload-role-name = "${var.resource-prefix}-project4-role"
    eks-cluster-name = var.eks-cluster-name
}