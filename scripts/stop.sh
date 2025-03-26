#!/bin/bash

# 기존 컨테이너 중지 및 삭제
CONTAINER_NAME="backend_container"

if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
  echo "Stopping running container: $CONTAINER_NAME"
  docker stop $CONTAINER_NAME
  docker rm $CONTAINER_NAME
else
  echo "No running container found for: $CONTAINER_NAME"
fi
