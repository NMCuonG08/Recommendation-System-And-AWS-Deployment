output "ec2_public_ip" {
  description = "Public IP address of the EC2 Serving Instance."
  value       = aws_instance.serving.public_ip
}

output "ec2_public_dns" {
  description = "Public DNS of the EC2 Serving Instance."
  value       = aws_instance.serving.public_dns
}

output "api_gateway_url" {
  description = "API Gateway recommendation endpoint URL."
  value       = "http://${aws_instance.serving.public_ip}:8080/recommend"
}

output "mlflow_url" {
  description = "MLflow Tracking Server URL."
  value       = "http://${aws_instance.serving.public_ip}:5000"
}

output "triton_url" {
  description = "Triton Inference Server HTTP URL."
  value       = "http://${aws_instance.serving.public_ip}:8000"
}

output "triton_s3_bucket" {
  description = "S3 Bucket name for Triton Model Repository."
  value       = aws_s3_bucket.triton_repo.bucket
}
