#!/bin/bash

IMAGE="120569628358.dkr.ecr.ap-northeast-2.amazonaws.com/backend:latest"
CONTAINER_NAME="backend_container"

aws ecr get-login-password --region ap-northeast-2 \
| docker login --username AWS \
  --password-stdin 120569628358.dkr.ecr.ap-northeast-2.amazonaws.com

# ECR에서 최신 이미지 pull
echo "Pulling latest image from ECR"
docker pull $IMAGE

docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true

# 새 컨테이너 실행
echo "Running new container"
docker run -d \
  --name $CONTAINER_NAME \
  -p 8000:8000 \
  $IMAGE
