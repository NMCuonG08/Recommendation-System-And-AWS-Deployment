variable "aws_region" {
  description = "AWS region for the CDC stack (Kinesis, DMS, Lambda, Secrets)."
  type        = string
  default     = "ap-southeast-1"
}

variable "rds_endpoint" {
  description = "RDS Postgres endpoint (host only) for the OLTP source. e.g. recsys-oltp.xxx.rds.amazonaws.com"
  type        = string
}

variable "rds_port" {
  description = "RDS Postgres port."
  type        = number
  default     = 5432
}

variable "rds_user" {
  description = "RDS master username."
  type        = string
  default     = "postgres"
}

variable "rds_db_name" {
  description = "RDS database holding the OLTP table (movie_ratings)."
  type        = string
  default     = "postgres"
}

variable "rds_password" {
  description = "RDS master password. SENSITIVE — pass via -var or .tfvars (gitignored), never commit."
  type        = string
  sensitive   = true
}

variable "feast_postgres_uri" {
  description = "Full Feast sql registry URI (RDS registry_feature_store). SENSITIVE. Stored in Secrets Manager; Lambda reads it at runtime."
  type        = string
  sensitive   = true
}

variable "sqs_queue_name" {
  description = "SQS Standard queue name for CDC events."
  type        = string
  default     = "recsys-cdc-queue"
}

variable "kinesis_stream_name" {
  description = "Kinesis Data Stream name for CDC events."
  type        = string
  default     = "recsys-cdc"
}

variable "dms_instance_class" {
  description = "DMS replication instance class. dms.t3.medium is NOT free-tier (~$0.052/h). terraform destroy when idle to stop charges."
  type        = string
  default     = "dms.t3.medium"
}

variable "dms_vpc_security_group_ids" {
  description = "Security group IDs for the DMS replication instance (must reach RDS)."
  type        = list(string)
  default     = []
}

variable "dms_vpc_subnet_ids" {
  description = "Subnet IDs for the DMS replication instance (same VPC as RDS)."
  type        = list(string)
  default     = []
}

variable "lambda_image_uri" {
  description = "ECR image URI for the CDC Lambda (output of infra/scripts/build_push_lambda_cdc.sh)."
  type        = string
}

variable "lambda_memory_mb" {
  description = "Lambda memory (512MB max for account quota)."
  type        = number
  default     = 512
}

variable "lambda_timeout_sec" {
  description = "Lambda timeout."
  type        = number
  default     = 60
}

variable "lambda_batch_size" {
  description = "Kinesis→Lambda batch size (1 for lowest-latency CDC)."
  type        = number
  default     = 1
}

variable "source_table" {
  description = "OLTP table to capture (DMS table mapping)."
  type        = string
  default     = "movie_ratings"
}

variable "source_schema" {
  description = "OLTP schema holding the source table."
  type        = string
  default     = "public"
}