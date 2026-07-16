variable "aws_region" {
  description = "AWS region for the EKS cluster."
  type        = string
  default     = "ap-southeast-1"
}

variable "cluster_name" {
  description = "EKS cluster name."
  type        = string
  default     = "recsys-eks"
}

variable "node_instance_type" {
  description = "Instance type for EKS managed worker nodes (CPU; MovieLens small needs no GPU)."
  type        = string
  default     = "t3.large"
}

variable "desired_capacity" {
  description = "Desired number of worker nodes (head + at least one Ray worker)."
  type        = number
  default     = 2
}