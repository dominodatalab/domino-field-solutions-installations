
variable "eks-cluster-name" {
    type = string
}

variable "domino-irsa-namespace" {
    type = string
    default = "domino-field"
}

variable "irsa-proxy-role-list" {
    type = list(string)
    default = []
}