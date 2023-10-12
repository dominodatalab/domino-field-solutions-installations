
variable "irsa-proxy-role-name" {
    type = string
    default = "domino-irsa-proxy-irsa-workload-role"
    description = "The string prefix for an IAM proxy role"
}

variable "irsa-workload-role-name" {
    type = string
    default = "irsa-workload-role"
    description = "The name of an associated asset role for an IRSA proxy role"
}

variable "irsa-workload-role-path" {
    type = string
    default = ""
    description = "Optional value for the path to the IRSA workload IAM role. End with a /"
}

variable "eks-cluster-name" {
    type = string
}