output "kinesis_stream_arn" {
  description = "Kinesis Data Stream ARN (CDC events)."
  value       = aws_kinesis_stream.cdc.arn
}

output "kinesis_stream_name" {
  description = "Kinesis Data Stream name."
  value       = aws_kinesis_stream.cdc.name
}

output "lambda_function_arn" {
  description = "CDC Lambda function ARN."
  value       = aws_lambda_function.cdc.arn
}

output "lambda_function_name" {
  description = "CDC Lambda function name."
  value       = aws_lambda_function.cdc.function_name
}

output "dms_task_arn" {
  description = "DMS replication task ARN (start it manually after one-time SQL)."
  value       = aws_dms_replication_task.cdc.replication_task_arn
}

output "dms_instance_arn" {
  description = "DMS replication instance ARN."
  value       = aws_dms_replication_instance.cdc.replication_instance_arn
}

output "rds_parameter_group_name" {
  description = "Parameter group with rds.logical_replication=1. Attach to recsys-oltp + reboot (manual, one-time)."
  value       = aws_db_parameter_group.cdc.name
}

output "feast_uri_secret_arn" {
  description = "Secrets Manager ARN holding the Feast registry URI (Lambda reads it at runtime)."
  value       = aws_secretsmanager_secret.feast_uri.arn
}

output "start_task_command" {
  description = "Start the DMS CDC task after the one-time SQL setup."
  value       = "aws dms start-replication-task --replication-task-arn ${aws_dms_replication_task.cdc.replication_task_arn} --region ${var.aws_region} --start-replication-task-type start-replication"
}

output "verify_command" {
  description = "Tail the Lambda logs while testing CDC."
  value       = "aws logs tail /aws/lambda/${aws_lambda_function.cdc.function_name} --follow --region ${var.aws_region}"
}