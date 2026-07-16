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
  description = "Instance type for EKS managed worker nodes (CPU; MovieLens small needs no GPU). Use a free-tier-eligible type if the AWS account restricts launches to free-tier only (run `aws ec2 describe-instance-types --filters Name=free-tier-eligible,Values=true`). m7i-flex.large = 2 vCPU / 8 GiB x86 (same RAM as t3.large) and is free-tier-eligible."
  type        = string
  default     = "m7i-flex.large"
}

variable "desired_capacity" {
  description = "Desired number of worker nodes (head + at least one Ray worker)."
  type        = number
  default     = 2
}