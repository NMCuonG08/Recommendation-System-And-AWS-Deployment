# Real-Time CDC stack (MovieLens): RDS → DMS → Kinesis → Lambda → Feast DynamoDB.
# See docs/03-realtime-cdc.md. Secrets via Secrets Manager (no hardcoding).

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# --------------------------------------------------------------------------- #
# Secrets (RDS password for DMS, Feast registry URI for Lambda)
# --------------------------------------------------------------------------- #
resource "aws_secretsmanager_secret" "rds_password" {
  name        = "recsys-cdc/rds_password"
  description = "RDS master password for the DMS source endpoint."
}

resource "aws_secretsmanager_secret_version" "rds_password" {
  secret_id     = aws_secretsmanager_secret.rds_password.id
  secret_string = var.rds_password
}

resource "aws_secretsmanager_secret" "feast_uri" {
  name        = "recsys-cdc/feast_postgres_uri"
  description = "Feast sql registry URI read by the CDC Lambda at runtime."
}

resource "aws_secretsmanager_secret_version" "feast_uri" {
  secret_id     = aws_secretsmanager_secret.feast_uri.id
  secret_string = var.feast_postgres_uri
}

# --------------------------------------------------------------------------- #
# SQS Standard Queue (Free Tier 1M requests/month)
# --------------------------------------------------------------------------- #
resource "aws_sqs_queue" "cdc" {
  name                      = var.sqs_queue_name
  message_retention_seconds = 86400
  visibility_timeout_seconds = var.lambda_timeout_sec * 6

  tags = { Project = "recsys-cdc" }
}

# --------------------------------------------------------------------------- #
# RDS parameter group: enable logical replication (one-time manual attach + reboot)
# --------------------------------------------------------------------------- #
resource "aws_db_parameter_group" "cdc" {
  name        = "recsys-cdc-logical-repl"
  family      = "postgres16"
  description = "Enable logical replication for DMS CDC."

  parameter {
    name         = "rds.logical_replication"
    value        = "1"
    apply_method = "pending-reboot"
  }
}
# NOTE: this group is NOT attached to the existing RDS automatically (terraform
# does not manage that RDS). After apply, attach it to recsys-oltp via console
# or `aws rds modify-db-instance ... --db-parameter-group-name recsys-cdc-logical-repl`
# then reboot. Run the one-time SQL in docs/03 (pglogical + publication) next.

# --------------------------------------------------------------------------- #
# DMS replication instance + endpoints + task
# --------------------------------------------------------------------------- #
resource "aws_iam_role" "dms_vpc" {
  name = "dms-vpc-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "dms.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "dms_vpc" {
  role       = aws_iam_role.dms_vpc.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonDMSVPCManagementRole"
}

resource "aws_iam_role" "dms_cloudwatch" {
  name = "dms-cloudwatch-logs-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "dms.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "dms_cloudwatch" {
  role       = aws_iam_role.dms_cloudwatch.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonDMSCloudWatchLogsRole"
}

resource "aws_dms_replication_subnet_group" "cdc" {
  replication_subnet_group_id          = "recsys-cdc-subnet-group"
  replication_subnet_group_description = "Subnets for the recsys CDC DMS instance."
  subnet_ids                           = var.dms_vpc_subnet_ids
}

resource "aws_dms_replication_instance" "cdc" {
  replication_instance_id     = "recsys-cdc-repl"
  replication_instance_class  = var.dms_instance_class
  allocated_storage           = 50
  vpc_security_group_ids      = var.dms_vpc_security_group_ids
  replication_subnet_group_id = aws_dms_replication_subnet_group.cdc.id
  publicly_accessible         = false
  auto_minor_version_upgrade  = true
}

resource "aws_dms_endpoint" "source" {
  endpoint_id   = "recsys-cdc-source-rds"
  endpoint_type = "source"
  engine_name   = "aurora-postgresql" # works for RDS Postgres too
  server_name   = var.rds_endpoint
  port          = var.rds_port
  username      = var.rds_user
  password      = var.rds_password
  database_name = var.rds_db_name
  ssl_mode      = "require"

  postgres_settings {
    plugin_name = "pglogical"
  }
}

resource "aws_s3_bucket" "cdc" {
  bucket        = "recsys-cdc-events-${data.aws_caller_identity.current.account_id}"
  force_destroy = true

  tags = { Project = "recsys-cdc" }
}

resource "aws_dms_endpoint" "target" {
  endpoint_id   = "recsys-cdc-target-s3"
  endpoint_type = "target"
  engine_name   = "s3"

  s3_settings {
    bucket_name             = aws_s3_bucket.cdc.id
    service_access_role_arn = aws_iam_role.dms_vpc.arn
    data_format             = "csv"
    timestamp_column_name   = "timestamp"
  }
}

resource "aws_sqs_queue_policy" "cdc" {
  queue_url = aws_sqs_queue.cdc.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "s3.amazonaws.com" }
      Action    = "sqs:SendMessage"
      Resource  = aws_sqs_queue.cdc.arn
      Condition = {
        ArnEquals = { "aws:SourceArn" = aws_s3_bucket.cdc.arn }
      }
    }]
  })
}

resource "aws_s3_bucket_notification" "cdc" {
  bucket = aws_s3_bucket.cdc.id

  queue {
    queue_arn = aws_sqs_queue.cdc.arn
    events    = ["s3:ObjectCreated:*"]
  }
  depends_on = [aws_sqs_queue_policy.cdc]
}

locals {
  table_mappings = jsonencode({
    rules = [
      {
        rule-type      = "selection"
        rule-id        = "1"
        rule-name      = "1"
        object-locator = { schema-name = var.source_schema, table-name = var.source_table }
        rule-action    = "include"
        rule-target    = "table"
      }
    ]
  })
}

resource "aws_dms_replication_task" "cdc" {
  replication_task_id      = "recsys-cdc-task"
  migration_type           = "cdc"
  replication_instance_arn = aws_dms_replication_instance.cdc.replication_instance_arn
  source_endpoint_arn      = aws_dms_endpoint.source.endpoint_arn
  target_endpoint_arn      = aws_dms_endpoint.target.endpoint_arn
  table_mappings           = local.table_mappings
  start_replication_task   = false
}

# --------------------------------------------------------------------------- #
# Lambda (Feast online update on SQS trigger)
# --------------------------------------------------------------------------- #
resource "aws_iam_role" "lambda_exec" {
  name = "recsys-cdc-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_exec" {
  name = "recsys-cdc-lambda-policy"
  role = aws_iam_role.lambda_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
      },
      {
        Effect   = "Allow"
        Action   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"]
        Resource = aws_sqs_queue.cdc.arn
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = [aws_secretsmanager_secret.feast_uri.arn]
      },
      {
        Effect   = "Allow"
        Action   = ["dynamodb:BatchWriteItem", "dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query", "dynamodb:Scan", "dynamodb:DescribeTable", "dynamodb:UpdateItem"]
        Resource = "arn:aws:dynamodb:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:table/*"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/recsys-cdc-lambda"
  retention_in_days = 7
}

resource "aws_lambda_function" "cdc" {
  function_name = "recsys-cdc-lambda"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = var.lambda_image_uri
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_sec
  publish       = false

  environment {
    variables = {
      REGISTRY_PATH_SECRET_ARN = aws_secretsmanager_secret.feast_uri.arn
      FEAST_REPO               = "/var/task"
      FEAST_YAML               = "feature_store.yaml"
    }
  }

  depends_on = [aws_cloudwatch_log_group.lambda, aws_iam_role_policy.lambda_exec]
}

resource "aws_lambda_event_source_mapping" "cdc" {
  event_source_arn = aws_sqs_queue.cdc.arn
  function_name    = aws_lambda_function.cdc.arn
  batch_size       = var.lambda_batch_size
  enabled          = true
}