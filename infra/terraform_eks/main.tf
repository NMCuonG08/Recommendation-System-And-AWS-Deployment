# EKS cluster for the 007 Item2Vec MLOps stack (KubeRay + in-cluster MLflow).
# Ported + modernized from the reference serving-cluster/terraform_eks: EKS
# module v20, cluster 1.30, CPU t3.large nodes, EBS CSI driver (gp3 PVCs for
# MLflow/Ray). No pod IRSA / CloudWatch — the 007 stack runs MinIO + Postgres
# in-cluster so it needs no AWS data services. See docs/eks-deploy.md.

data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

# ---------------------------------------------------------------------------
# VPC
# ---------------------------------------------------------------------------
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = slice(data.aws_availability_zones.available.names, 0, 2)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
  tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# ---------------------------------------------------------------------------
# EKS cluster + managed node group (CPU)
# ---------------------------------------------------------------------------
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.30"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true
  enable_irsa                    = true

  eks_managed_node_groups = {
    default = {
      instance_types = [var.node_instance_type]
      min_size       = 1
      max_size       = 3
      desired_size   = var.desired_capacity
      disk_size      = 50
    }
  }

  tags = {
    Environment = "dev"
    Project     = "recsys-item2vec"
  }
}

# ---------------------------------------------------------------------------
# EBS CSI driver IRSA role + addon (required for gp3 PVCs on EKS 1.23+).
# ---------------------------------------------------------------------------
resource "aws_iam_role" "ebs_csi" {
  name = "${var.cluster_name}-ebs-csi"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = module.eks.oidc_provider_arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${module.eks.oidc_provider}:sub" = "system:serviceaccount:kube-system:ebs-csi-controller-sa"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ebs_csi" {
  role       = aws_iam_role.ebs_csi.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

resource "aws_eks_addon" "ebs_csi" {
  # addon_version omitted → AWS picks the latest compatible version for the
  # cluster's Kubernetes version. (The `most_recent` argument does not exist on
  # aws_eks_addon in the AWS provider.)
  cluster_name                = module.eks.cluster_name
  addon_name                  = "aws-ebs-csi-driver"
  service_account_role_arn    = aws_iam_role.ebs_csi.arn
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"

  depends_on = [module.eks]
}

# ---------------------------------------------------------------------------
# Optional: IRSA for S3/RDS access if you later swap in-cluster MinIO/Postgres
# for real AWS S3 + RDS (Path B reuse). Not needed for the in-cluster 007 stack.
# To enable, create a ServiceAccount annotated with the role ARN below and a
# policy scoped to your bucket/RDS — see docs/eks-deploy.md "Going real AWS".
# ---------------------------------------------------------------------------