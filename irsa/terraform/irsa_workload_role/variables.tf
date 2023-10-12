variable "irsa-workload-role-name" {
    default = "domino-irsa-workload-role"
}

variable "associated-proxy-role-list" {
    type = list(string)
    default = []
}

variable "policy-to-attach" {
    default = ""
}