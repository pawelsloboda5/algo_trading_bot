# Infrastructure Module - Claude Context

## Purpose
Infrastructure as Code (IaC) for deploying the trading bot to AWS.

## Directory Structure

```
infrastructure/
├── docker/
│   ├── Dockerfile           # Main application container (TODO)
│   ├── Dockerfile.dev       # Development container (TODO)
│   └── docker-compose.yml   # Local dev stack (TODO)
├── terraform/
│   ├── main.tf              # Root config (TODO)
│   ├── variables.tf         # Input variables (TODO)
│   ├── outputs.tf           # Output values (TODO)
│   ├── provider.tf          # AWS provider (TODO)
│   ├── modules/
│   │   ├── ec2/            # EC2 in Local Zone (TODO)
│   │   ├── networking/     # VPC/subnets (TODO)
│   │   └── security/       # SG, IAM (TODO)
│   └── environments/
│       ├── dev/
│       └── prod/
└── scripts/
    ├── deploy.sh           # Deployment script (TODO)
    └── setup_ec2.sh        # EC2 initialization (TODO)
```

## AWS Architecture

### Target: Chicago Local Zone

- **Region**: `us-east-1` (N. Virginia)
- **Local Zone**: `us-east-1-chi-2a` (Chicago)
- **Purpose**: Sub-millisecond latency to CME Globex

### Why Chicago Local Zone?

CME Globex data center is in Aurora, IL (Chicago suburb). The AWS Chicago Local Zone provides:
- Single-digit millisecond latency to CME
- C6in instances with 200Gbps network
- Direct fiber to exchange

### Instance Types

| Type | vCPU | Memory | Network | Use Case |
|------|------|--------|---------|----------|
| c6i.large | 2 | 4 GB | Up to 12.5 Gbps | Development |
| c6in.xlarge | 4 | 8 GB | Up to 30 Gbps | Paper trading |
| c6in.2xlarge | 8 | 16 GB | Up to 40 Gbps | Production |

### Network Architecture

```
┌─────────────────────────────────────────┐
│              us-east-1                   │
│  ┌─────────────────────────────────┐    │
│  │         VPC (10.0.0.0/16)        │    │
│  │  ┌───────────────────────────┐  │    │
│  │  │  us-east-1-chi-2a (LZ)    │  │    │
│  │  │  Subnet: 10.0.10.0/24     │  │    │
│  │  │  ┌─────────────────────┐  │  │    │
│  │  │  │   EC2 (Trading)     │  │  │    │
│  │  │  └─────────────────────┘  │  │    │
│  │  └───────────────────────────┘  │    │
│  │  ┌───────────────────────────┐  │    │
│  │  │  us-east-1a (AZ)          │  │    │
│  │  │  Subnet: 10.0.1.0/24      │  │    │
│  │  │  ┌─────────────────────┐  │  │    │
│  │  │  │   EC2 (Monitoring)  │  │  │    │
│  │  │  └─────────────────────┘  │  │    │
│  │  └───────────────────────────┘  │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## Docker Strategy

### Development Container
- Python 3.11
- All dependencies installed
- Mounted source code
- Hot reload for development

### Production Container
- Multi-stage build
- Minimal runtime image
- No dev dependencies
- Optimized for size

### IB Gateway Container
- Use official IB Gateway image
- Paper trading: port 4002
- Live trading: port 4001

## Terraform Modules

### networking/
- VPC with DNS support
- Public subnet in Chicago Local Zone
- Internet Gateway
- Route tables

### ec2/
- Launch template with user data
- Auto-recovery enabled
- EBS optimization
- Enhanced networking

### security/
- Security group for trading (IB ports)
- IAM role for EC2
- Secrets Manager for API keys

## Deployment Flow

1. Enable Chicago Local Zone in AWS Console
2. `terraform init`
3. `terraform plan -var-file=environments/dev/terraform.tfvars`
4. `terraform apply`
5. SSH to instance, pull Docker image
6. Start trading containers

## Cost Considerations

- Local Zone instances cost ~10-15% more
- Data transfer within Local Zone is cheaper
- Use Reserved Instances for production
- Spot instances NOT recommended for trading

## Security Requirements

- No public IP in production (use bastion)
- API keys in AWS Secrets Manager
- VPC endpoints for AWS services
- CloudWatch for monitoring

## Status

All infrastructure files are TODO - to be implemented in Phase 6.
