# Metrics Engine - Deployment Guide

## Prerequisites

1. **AWS CLI configured**
   ```bash
   aws configure
   # Enter Access Key, Secret Key, Region (ap-south-1)
   ```

2. **Docker installed**
   ```bash
   docker --version
   ```

3. **ECS Cluster exists**
   ```bash
   aws ecs describe-clusters --clusters obsera-cluster --region ap-south-1
   ```

---

## Quick Deploy

### Option 1: Automated Script (Recommended)

```bash
cd deployment
./deploy.sh
```

This will:
- ✅ Build Docker image
- ✅ Push to ECR
- ✅ Register task definition
- ✅ Create/update ECS service
- ✅ Wait for deployment

---

### Option 2: GitHub Actions (CI/CD)

1. **Set GitHub Secrets:**
   ```
   AWS_ACCESS_KEY_ID
   AWS_SECRET_ACCESS_KEY
   ```

2. **Push to GitHub:**
   ```bash
   git push origin main
   ```

3. **Deployment auto-triggers!**

Monitor: https://github.com/DendriSys/obsera-metrics-engine/actions

---

### Option 3: Manual Deployment

```bash
# 1. Build image
docker build -t obsera-metrics-engine:latest .

# 2. Tag for ECR
docker tag obsera-metrics-engine:latest \
  339713024809.dkr.ecr.ap-south-1.amazonaws.com/obsera-metrics-engine:latest

# 3. Login to ECR
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin \
  339713024809.dkr.ecr.ap-south-1.amazonaws.com

# 4. Push to ECR
docker push 339713024809.dkr.ecr.ap-south-1.amazonaws.com/obsera-metrics-engine:latest

# 5. Register task definition
aws ecs register-task-definition \
  --cli-input-json file://ecs-task-definition.json \
  --region ap-south-1

# 6. Update service
aws ecs update-service \
  --cluster obsera-cluster \
  --service metrics-engine-service \
  --task-definition metrics-engine-task \
  --force-new-deployment \
  --region ap-south-1
```

---

## Post-Deployment Testing

### 1. Get Service Status

```bash
aws ecs describe-services \
  --cluster obsera-cluster \
  --services metrics-engine-service \
  --region ap-south-1 \
  --query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount}'
```

### 2. Get Task Public IP

```bash
# Get task ARN
TASK_ARN=$(aws ecs list-tasks \
  --cluster obsera-cluster \
  --service metrics-engine-service \
  --region ap-south-1 \
  --query 'taskArns[0]' \
  --output text)

# Get task details
aws ecs describe-tasks \
  --cluster obsera-cluster \
  --tasks $TASK_ARN \
  --region ap-south-1 \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
  --output text

# Get public IP from ENI
ENI_ID=<from above>
aws ec2 describe-network-interfaces \
  --network-interface-ids $ENI_ID \
  --region ap-south-1 \
  --query 'NetworkInterfaces[0].Association.PublicIp' \
  --output text
```

### 3. Test Health Endpoint

```bash
PUBLIC_IP=<from above>
curl http://$PUBLIC_IP:8001/health
```

Expected response:
```json
{
  "status": "healthy",
  "prometheus_connected": true,
  "ollama_connected": true,
  "vector_db_ready": true,
  "version": "1.0.0"
}
```

### 4. View Logs

```bash
# Stream logs
aws logs tail /ecs/metrics-engine --follow --region ap-south-1

# Last 100 lines
aws logs tail /ecs/metrics-engine --since 1h --region ap-south-1
```

---

## Configuration

### Environment Variables

Edit `deployment/ecs-task-definition.json`:

```json
{
  "environment": [
    {"name": "PROMETHEUS_URL", "value": "http://prometheus.obsera.local:9090"},
    {"name": "OLLAMA_URL", "value": "http://ollama.obsera.local:11434"},
    {"name": "VECTOR_STORE_PATH", "value": "/app/vector_store"}
  ]
}
```

### Resource Allocation

Current: `512 CPU`, `2048 MB Memory`

To increase:
```json
{
  "cpu": "1024",
  "memory": "4096"
}
```

---

## Scaling

### Manual Scaling

```bash
aws ecs update-service \
  --cluster obsera-cluster \
  --service metrics-engine-service \
  --desired-count 2 \
  --region ap-south-1
```

### Auto-Scaling (Future)

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/obsera-cluster/metrics-engine-service \
  --min-capacity 1 \
  --max-capacity 5
```

---

## Troubleshooting

### Service won't start

```bash
# Check task status
aws ecs describe-tasks \
  --cluster obsera-cluster \
  --tasks $(aws ecs list-tasks --cluster obsera-cluster --service metrics-engine-service --query 'taskArns[0]' --output text) \
  --region ap-south-1
```

### Check logs for errors

```bash
aws logs tail /ecs/metrics-engine --since 30m --region ap-south-1 | grep ERROR
```

### Container health check failing

```bash
# SSH into task (if enabled)
aws ecs execute-command \
  --cluster obsera-cluster \
  --task $TASK_ARN \
  --container metrics-engine \
  --command "/bin/bash" \
  --interactive
```

---

## Rollback

### To previous version

```bash
# List task definitions
aws ecs list-task-definitions \
  --family-prefix metrics-engine-task \
  --region ap-south-1

# Update to specific version
aws ecs update-service \
  --cluster obsera-cluster \
  --service metrics-engine-service \
  --task-definition metrics-engine-task:1 \
  --region ap-south-1
```

---

## Monitoring

### CloudWatch Metrics

- CPU Utilization
- Memory Utilization
- Network In/Out
- Running Count

Dashboard: https://console.aws.amazon.com/ecs/

### Custom Metrics

The service exposes:
- `/health` - Health check
- `/metrics/stats` - Vector store statistics

---

## Cost Estimation

**Current Configuration:**
- vCPU: 0.5
- Memory: 2 GB
- Cost: ~$15-20/month (running 24/7)

**To reduce cost:**
- Scale to 0 when not in use
- Use Spot instances
- Reduce memory if possible

---

## Next Steps

1. ✅ Deploy service
2. ✅ Verify health
3. ✅ Test endpoints
4. 📊 Set up monitoring
5. 🔒 Configure security groups
6. 🌐 Set up load balancer (optional)

---

**Support:** Check logs in CloudWatch or open GitHub issue
