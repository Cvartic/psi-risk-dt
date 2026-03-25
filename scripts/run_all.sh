#!/bin/bash
cd "$(dirname "$0")"

mkdir -p ../results
mkdir -p ../data

#Build and start the docker container
echo "Building and starting the docker container..."
docker-compose -f ../docker/docker-compose.yml up -d --build

#Generate the attack scenarios
echo "Generating attack scenarios..."
docker exec -it psi-risk-dt python scripts/attack_scenarios/scenario_campaigns/scenario_campaigns_generator.py

