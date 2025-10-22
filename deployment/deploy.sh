#!/bin/bash

# Deployment script for Metrics Engine to AWS ECS
# Usage: ./deploy.sh [environment]

set -e

ENVIRONMENT=${1:-production}
AWS_REGION="ap-south-1"
AWS_ACCOUNT_ID="107456217325"
ECR_REPOSITORY="obsera-metrics-engine"
ECS_CLUSTER="obsera-cluster"
ECS_SERVICE="metrics-engine-service"
TASK_FAMILY="metrics-engine-task"

echo "=========================================="
echo "Deploying Metrics Engine to ECS"
echo "Environment: $ENVIRONMENT"
echo "=========================================="

# Step 1: Build Docker image
echo ""
echo "Step 1: Building Docker image..."
docker build -t ${ECR_REPOSITORY}:latest ..

# Step 2: Tag for ECR
echo ""
echo "Step 2: Tagging image for ECR..."
docker tag ${ECR_REPOSITORY}:latest \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest

docker tag ${ECR_REPOSITORY}:latest \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:$(git rev-parse --short HEAD)

# Step 3: Login to ECR
echo ""
echo "Step 3: Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Step 4: Create ECR repository if it doesn't exist
echo ""
echo "Step 4: Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} --region ${AWS_REGION} 2>/dev/null || \
  aws ecr create-repository --repository-name ${ECR_REPOSITORY} --region ${AWS_REGION}

# Step 5: Push to ECR
echo ""
echo "Step 5: Pushing image to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:$(git rev-parse --short HEAD)

# Step 6: Register task definition
echo ""
echo "Step 6: Registering ECS task definition..."
aws ecs register-task-definition \
  --cli-input-json file://ecs-task-definition.json \
  --region ${AWS_REGION}

# Step 7: Check if service exists
echo ""
echo "Step 7: Checking ECS service..."
SERVICE_EXISTS=$(aws ecs describe-services \
  --cluster ${ECS_CLUSTER} \
  --services ${ECS_SERVICE} \
  --region ${AWS_REGION} \
  --query 'services[0].status' \
  --output text 2>/dev/null || echo "MISSING")

if [ "$SERVICE_EXISTS" = "ACTIVE" ] || [ "$SERVICE_EXISTS" = "DRAINING" ]; then
  # Update existing service
  echo "Updating existing service..."
  aws ecs update-service \
    --cluster ${ECS_CLUSTER} \
    --service ${ECS_SERVICE} \
    --task-definition ${TASK_FAMILY} \
    --force-new-deployment \
    --region ${AWS_REGION}
else
  # Create new service
  echo "Creating new service..."
  aws ecs create-service \
    --cluster ${ECS_CLUSTER} \
    --service-name ${ECS_SERVICE} \
    --task-definition ${TASK_FAMILY} \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-0a8e9b5c3d4f6e7a8],securityGroups=[sg-0a1b2c3d4e5f6g7h8],assignPublicIp=ENABLED}" \
    --region ${AWS_REGION} \
    --enable-service-connect \
    --service-connect-configuration '{
      "enabled": true,
      "namespace": "obsera.local",
      "services": [{
        "portName": "metrics-engine-8001",
        "clientAliases": [{
          "port": 8001,
          "dnsName": "metrics-engine"
        }]
      }]
    }'
fi

# Step 8: Wait for deployment
echo ""
echo "Step 8: Waiting for deployment to complete..."
aws ecs wait services-stable \
  --cluster ${ECS_CLUSTER} \
  --services ${ECS_SERVICE} \
  --region ${AWS_REGION}

# Step 9: Get service info
echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Service Details:"
aws ecs describe-services \
  --cluster ${ECS_CLUSTER} \
  --services ${ECS_SERVICE} \
  --region ${AWS_REGION} \
  --query 'services[0].{Name:serviceName,Status:status,RunningCount:runningCount,DesiredCount:desiredCount,TaskDefinition:taskDefinition}' \
  --output table

echo ""
echo "To view logs:"
echo "  aws logs tail /ecs/metrics-engine --follow --region ${AWS_REGION}"
echo ""
echo "To get task public IP:"
echo "  aws ecs list-tasks --cluster ${ECS_CLUSTER} --service ${ECS_SERVICE} --region ${AWS_REGION}"
echo ""
echo "Health check:"
echo "  curl http://<public-ip>:8001/health"
echo ""
