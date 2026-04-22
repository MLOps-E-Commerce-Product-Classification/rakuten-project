#!/bin/bash

echo "AIRFLOW_UID=50000" >> .env
echo "AIRFLOW_GID=0" >> .env
echo "DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)" >> .env

echo "PROJECT_ROOT=$(pwd)" >> .env

echo ".env ready ✅"
