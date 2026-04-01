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

#Generate the baseline traffic
echo "Generating baseline traffic..."
docker exec -it psi-risk-dt python scripts/baseline_generator/generate_baseline_traffic.py

#format the generated scenarios
echo "Formatting the generated scenarios..."
docker exec -it psi-risk-dt python scripts/formatter/batch_formatter.py

