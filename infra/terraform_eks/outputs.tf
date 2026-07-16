output "cluster_endpoint" {
  description = "EKS API server endpoint."
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name (use with aws eks update-kubeconfig)."
  value       = module.eks.cluster_name
}

output "cluster_region" {
  description = "AWS region of the cluster."
  value       = var.aws_region
}

output "update_kubeconfig_command" {
  description = "Run this to point kubectl at the new cluster."
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.aws_region}"
}