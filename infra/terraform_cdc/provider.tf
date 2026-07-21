terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state in the shared S3 bucket (same as terraform_eks). Create the
  # bucket once: aws s3api create-bucket --bucket recsys-ops-tfstate --region <r> \
  #   --create-bucket-configuration LocationConstraint=<r>
  backend "s3" {
    bucket = "recsys-ops-tfstate"
    key    = "cdc/terraform.tfstate"
    region = "ap-southeast-1"
  }
}

provider "aws" {
  region = var.aws_region
  # Credentials via ~/.aws/credentials or env vars (AWS_ACCESS_KEY_ID, etc.).
}