variable "aws_region" {
  description = "AWS region for the EC2 serving deployment."
  type        = string
  default     = "ap-southeast-1"
}

variable "instance_type" {
  description = "EC2 instance type (t3.large recommended for Triton + Qdrant + Gateway)."
  type        = string
  default     = "m7i-flex.large"
}

variable "key_name" {
  description = "Optional SSH key pair name for EC2 access."
  type        = string
  default     = ""
}
